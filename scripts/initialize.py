import os
import sys
import pandas as pd
from caveclient import CAVEclient

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))

if project_root not in sys.path:
    sys.path.append(project_root)

from mouse_medi.config import paths

OUTPUT_CSV = os.path.join(getattr(paths, 'DATA_DIR'), 'neuron_info.csv')
PERFORMANCE_CSV = os.path.join(getattr(paths, 'PERFORMANCE_CSV'))

def get_neuron_info():
    print("Loading local data...")
    params_df = pd.read_csv(paths.PARAMS_CSV)
    anatomy_df = pd.read_csv(paths.ANATOMY_CSV)
    units_df = pd.merge(params_df, anatomy_df, on=['session', 'scan_idx', 'unit_id'], how='inner')
    print(f"Loaded {len(units_df)} units.")

    # 1. Query BCM to get root_ids
    client = CAVEclient('minnie65_public')
    print("Querying BCM Table for root IDs...")
    bcm_rows = []
    unique_sessions = units_df[['session', 'scan_idx']].drop_duplicates()

    for idx, (_, row) in enumerate(unique_sessions.iterrows()):
        session = row['session']
        scan_idx = int(row['scan_idx'])

        try:
            bcm_df = client.materialize.query_table(
                'digital_twin_properties_bcm_coreg_auto_phase3_fwd_v2',
                filter_in_dict={'session': [session], 'scan_idx': [scan_idx]}
            )
            if not bcm_df.empty:
                bcm_rows.append(bcm_df[['pt_root_id', 'session', 'scan_idx', 'unit_id']])
        except Exception as e:
            print(f"Error querying session {session}, scan {scan_idx}: {e}")

    if not bcm_rows:
        print("No BCM data found! Exiting.")
        exit(1)

    all_bcm_df = pd.concat(bcm_rows, ignore_index=True)
    print(f"Found {len(all_bcm_df)} BCM matches.")

    # 2. Load cc_max info and deduplicate by root_id
    print("Loading performance stats for deduplication...")
    units_perf = pd.read_csv(PERFORMANCE_CSV)
    
    merged_df = pd.merge(units_df, all_bcm_df, on=['session', 'scan_idx', 'unit_id'], how='left')
    merged_df = pd.merge(merged_df, units_perf[['session', 'scan_idx', 'unit_id', 'cc_max']], on=['session', 'scan_idx', 'unit_id'], how='left')
    merged_df = merged_df.sort_values('cc_max', ascending=False).drop_duplicates('pt_root_id', keep='first')
    merged_df = merged_df.sort_values(['session', 'scan_idx', 'unit_id'])
    
    valid_roots = merged_df['pt_root_id'].dropna().astype(int).unique().tolist()
    print(f"Found {len(valid_roots)} unique root IDs to query for cell types.")

    # 3. Query Nucleus Table to get nucleus IDs
    print("Querying Nucleus Table for nucleus IDs...")
    root_to_nuc = {}
    batch_size = 5000
    total_roots = len(valid_roots)

    for i in range(0, total_roots, batch_size):
        batch_roots = valid_roots[i:i+batch_size]
        try:
            nuc_df = client.materialize.query_table(
                'nucleus_detection_v0',
                filter_in_dict={'pt_root_id': batch_roots},
                select_columns=['pt_root_id', 'id']
            )
            if not nuc_df.empty:
                for _, r in nuc_df.iterrows():
                    root_to_nuc[int(r['pt_root_id'])] = int(r['id'])
        except Exception as e:
            print(f"Error querying nucleus batch {i}: {e}")

    print(f"Mapped {len(root_to_nuc)} roots to nucleus IDs.")

    # 4. Query Cell Types using Nucleus IDs
    valid_nuc_ids = list(set(root_to_nuc.values()))
    total_nucs = len(valid_nuc_ids)
    print(f"Querying Cell Types for {total_nucs} unique nucleus IDs...")

    nuc_to_type = {}
    for i in range(0, total_nucs, batch_size):
        batch_nucs = valid_nuc_ids[i:i+batch_size]
        try:
            type_df = client.materialize.query_table(
                'aibs_metamodel_celltypes_v661',
                filter_in_dict={'target_id': batch_nucs},
                select_columns=['target_id', 'cell_type']
            )
            if not type_df.empty:
                for _, r in type_df.iterrows():
                    nuc_to_type[int(r['target_id'])] = r['cell_type']
            
            try:
                corrections_df = client.materialize.query_table(
                    'aibs_metamodel_celltypes_v661_corrections',
                    filter_in_dict={'target_id': batch_nucs},
                    select_columns=['target_id', 'cell_type']
                )
                if not corrections_df.empty:
                    for _, r in corrections_df.iterrows():
                        nuc_to_type[int(r['target_id'])] = r['cell_type']
            except Exception as e:
                if "Table not found" not in str(e):
                    print(f"Warnings: Error querying corrections in batch {i}: {e}")

        except Exception as e:
            print(f"Error querying cell types batch {i}: {e}")

    def map_cell_type_to_layer(ctype):
        if pd.isna(ctype):
            return None
        ctype = str(ctype)
        if ctype.startswith('23'):
            return 'L23'
        elif ctype.startswith('4'):
            return 'L4'
        elif ctype.startswith('5'):
            return 'L5'
        return None

    def get_layer_and_type(row):
        root = row['pt_root_id']
        if pd.isna(root):
            return None, None
        root = int(root)
        nuc_id = root_to_nuc.get(root)
        if not nuc_id:
            return None, None
        
        ctype = nuc_to_type.get(nuc_id)
        return map_cell_type_to_layer(ctype), ctype

    print("Mapping layers and cell types...")
    layers = []
    types = []
    for idx, row in merged_df.iterrows():
        l, t = get_layer_and_type(row)
        layers.append(l)
        types.append(t)

    merged_df['layer'] = layers
    merged_df['cell_type'] = types

    final_df = merged_df[merged_df['layer'].isin(['L23', 'L4', 'L5'])].copy()
    cols = [c for c in final_df.columns if c not in ['layer', 'cell_type']]
    final_df = final_df[cols + ['layer', 'cell_type']]

    print(f"Final neuron count: {len(final_df)}")
    if not final_df.empty:
        print("Layer counts:")
        print(final_df['layer'].value_counts())

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved to {OUTPUT_CSV}")

def main():
    if os.path.exists(OUTPUT_CSV):
        print(f"[SKIP] Neuron info found at: {OUTPUT_CSV}")
    else:
        print(f"[MISSING] Neuron info not found at: {OUTPUT_CSV}")
        get_neuron_info()
    
    print("Neuron info initialization completed.")

if __name__ == "__main__":
    main()