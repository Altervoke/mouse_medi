import os
import sys
import gc
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import imageio
from PIL import Image
import matplotlib
matplotlib.use('Agg')

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
fnn_root = os.path.join(project_root, 'fnn')

if project_root not in sys.path:
    sys.path.append(project_root)
if fnn_root not in sys.path:
    sys.path.append(fnn_root)

from medi_pipeline.config import paths
from fnn import microns

EXEC_ORDER = os.path.join(getattr(paths, 'DATA_DIR'), 'execution_order.csv')
MEDI_FEAT = os.path.join(getattr(paths, 'DATA_DIR'), 'medi_features.csv')
GRATING_FEAT = os.path.join(getattr(paths, 'DATA_DIR'), 'grating_features.csv')
NATURAL_FEAT = os.path.join(getattr(paths, 'DATA_DIR'), 'natural_max_features.csv')

UNITS_CSV = os.path.join(fnn_root, 'data', 'microns_digital_twin', 'properties', 'responses', 'units.csv')
STIM_MMAP_PATH = os.path.join(fnn_root, 'data', 'microns_digital_twin', 'properties', 'responses', 'stimulus.npy')

OUTPUT_CSV = os.path.join(getattr(paths, 'DATA_DIR'), 'baseline_comparisons.csv')

def extract_model_response_batch(model, videos_uint8, device='cuda'):
    batch_input = np.transpose(videos_uint8, (1, 0, 2, 3))[..., None]
    
    with torch.no_grad():
        gen = model.generate_response(batch_input, reset=True)
        all_resps = []
        for resp in gen:
            all_resps.append(resp)
            
    all_resps = np.stack(all_resps, axis=0)
    if all_resps.shape[0] > 10:
        mean_resp = np.mean(all_resps[10:], axis=0)
    else:
        mean_resp = np.mean(all_resps, axis=0)
    return mean_resp

def load_video(path):
    if not os.path.exists(path): return None
    try:
        vid = imageio.mimread(path)
        if len(vid) == 0: return None
        v_arr = np.array(vid)
        if v_arr.ndim == 4:
            v_arr = np.mean(v_arr[..., :3], axis=3).astype(np.uint8)
        return v_arr
    except: return None
    
def load_image(path):
    if not os.path.exists(path): return None
    try:
        img = Image.open(path).convert('L')
        v_arr = np.array(img)
        return np.stack([v_arr]*60, axis=0)
    except: return None

def construct_natural_max_features():
    if os.path.exists(NATURAL_FEAT):
        return
    print(f"Generating {NATURAL_FEAT}...")
    import numpy as np
    import pandas as pd
    
    df_medi = pd.read_csv(MEDI_FEAT)
    df_units = pd.read_csv(UNITS_CSV)
    
    merged = pd.merge(
        df_medi[['session', 'scan_idx', 'unit_id', 'readout_id', 'resp']],
        df_units.reset_index().rename(columns={'index': 'neuron_idx'}),
        on=['session', 'scan_idx', 'unit_id'],
        how='inner'
    )
    if merged.empty:
        return
        
    neuron_indices = merged['neuron_idx'].values
    session_vals = merged['session'].values
    scan_idx_vals = merged['scan_idx'].values
    unit_id_vals = merged['unit_id'].values
    readout_id_vals = merged['readout_id'].values
    
    responses_path = os.path.join(fnn_root, 'data', 'microns_digital_twin', 'properties', 'responses', 'responses.npy')
    responses = np.load(responses_path, mmap_mode='r')
    
    window_size = 60
    n_neurons = len(neuron_indices)
    natural_max = np.zeros(n_neurons)
    start_frames = np.zeros(n_neurons, dtype=int)
    end_frames = np.zeros(n_neurons, dtype=int)
    
    for i, neuron_idx in enumerate(neuron_indices):
        row = np.array(responses[neuron_idx, :], dtype=np.float32)
        chunk_size = 300
        n_frames = len(row)
        n_chunks = n_frames // chunk_size
        
        if n_chunks == 0 or n_frames < window_size:
            natural_max[i] = np.mean(row)
            start_frames[i] = 0
            end_frames[i] = n_frames - 1
        else:
            row_chunks = row[:n_chunks * chunk_size].reshape(n_chunks, chunk_size)
            cumsum = np.zeros((n_chunks, chunk_size + 1), dtype=np.float32)
            np.cumsum(row_chunks, axis=1, out=cumsum[:, 1:])
            window_sums = cumsum[:, window_size:] - cumsum[:, :-window_size]
            max_idx = np.argmax(window_sums)
            chunk_idx = max_idx // window_sums.shape[1]
            frame_in_chunk = max_idx % window_sums.shape[1]
            natural_max[i] = window_sums[chunk_idx, frame_in_chunk] / window_size
            start_frames[i] = chunk_idx * chunk_size + frame_in_chunk
            end_frames[i] = start_frames[i] + window_size - 1
        
    out_df = pd.DataFrame({
        'session': session_vals,
        'scan_idx': scan_idx_vals,
        'unit_id': unit_id_vals,
        'readout_id': readout_id_vals,
        'start_frame': start_frames,
        'end_frame': end_frames
    })
    out_df.to_csv(NATURAL_FEAT, index=False)
    print(f"Saved {NATURAL_FEAT}")


def main():
    construct_natural_max_features()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    df_order = pd.read_csv(EXEC_ORDER)
    df_medi = pd.read_csv(MEDI_FEAT)
    df_grat = pd.read_csv(GRATING_FEAT)
    df_nat = pd.read_csv(NATURAL_FEAT)
    df_units = pd.read_csv(UNITS_CSV)
    
    medi_map = df_medi.set_index(['session', 'scan_idx', 'readout_id']).to_dict('index')
    grat_map = df_grat.set_index(['session', 'scan_idx', 'readout_id']).to_dict('index')
    nat_map = df_nat.set_index(['session', 'scan_idx', 'readout_id']).to_dict('index')
    
    stim_mmap = np.load(STIM_MMAP_PATH, mmap_mode='r')
    
    idx_map = {}
    for i, r in df_units.iterrows():
        idx_map[(int(r['session']), int(r['scan_idx']), int(r['unit_id']))] = i
        
    existing_keys = set()
    results_records = []
    if os.path.exists(OUTPUT_CSV):
        try:
            df_existing = pd.read_csv(OUTPUT_CSV)
            if all(col in df_existing.columns for col in ['session', 'scan_idx', 'readout_id']):
                for _, r in df_existing.iterrows():
                    existing_keys.add((int(r['session']), int(r['scan_idx']), int(r['readout_id'])))
                results_records = df_existing.to_dict('records')
                print(f"Loaded {len(existing_keys)} existing records from {OUTPUT_CSV}. Will skip them.")
        except Exception as e:
            print(f"Warning: Could not read existing output file: {e}")
    
    valid_targets = []
    
    for _, row in df_order.iterrows():
        key = (int(row.session), int(row.scan_idx), int(row.readout_id))
        if key in existing_keys: continue
        if key not in medi_map: continue
        if key not in grat_map: continue
        if key not in nat_map: continue
        
        unit_key = (int(row.session), int(row.scan_idx), int(row.unit_id))
        if unit_key not in idx_map: continue
        
        g_path = os.path.join(getattr(paths, 'RESULTS_DIR'), 'Gabor', row.brain_area, row.layer, f"{int(row.session)}_{int(row.scan_idx)}_r{int(row.readout_id)}.png")
        if not os.path.exists(g_path): continue
          
        m_path = os.path.join(getattr(paths, 'RESULTS_DIR'), 'MESI', row.brain_area, row.layer, f"{int(row.session)}_{int(row.scan_idx)}_r{int(row.readout_id)}.png")
        if not os.path.exists(m_path): continue
        
        valid_targets.append(row)
        
    if len(valid_targets) == 0:
        print("No valid target found!")
        return
        
    print(f"Found {len(valid_targets)} complete neuro targets!")
    
    valid_targets_df = pd.DataFrame(valid_targets)
    groups = valid_targets_df.groupby(['session', 'scan_idx'])
    
    NEURONS_PER_BATCH = 10
    
    for (s, sc), group in groups:
        s, sc = int(s), int(sc)
        print(f"\n================ Loading Model Session {s}, Scan {sc} ({len(group)} neurons) ================")
        if device == 'cuda': torch.cuda.empty_cache()
        model, ids_df = microns.scan(session=s, scan_idx=sc, directory=getattr(paths, 'SRC_DIR'))
        model = model.to(device).eval()
        for p in model.parameters(): p.requires_grad = False
        
        if 'readout_id' in ids_df.columns: rids = ids_df['readout_id'].values
        else: rids = ids_df.index.values
        rid_to_idx = {r: j for j, r in enumerate(rids)}
        
        for i in tqdm(range(0, len(group), NEURONS_PER_BATCH), desc=f'Session {s} Scan {sc}'):
            chunk = group.iloc[i:i+NEURONS_PER_BATCH]
            videos_to_run = []
            meta_info = []
            
            for _, n in chunk.iterrows():
                rid, uid = int(n.readout_id), int(n.unit_id)
                area, layer = n.brain_area, n.layer
                model_unit_idx = rid_to_idx[rid]
                
                gab_path = os.path.join(getattr(paths, 'RESULTS_DIR'), 'Gabor', area, layer, f"{s}_{sc}_r{rid}.png")
                gabor_vid = load_image(gab_path)
                
                mesi_path = os.path.join(getattr(paths, 'RESULTS_DIR'), 'MESI', area, layer, f"{s}_{sc}_r{rid}.png")
                mesi_vid = load_image(mesi_path)
                
                n_row = nat_map[(s, sc, rid)]
                st, ed = int(n_row['start_frame']), int(n_row['end_frame'])
                nat_vid = stim_mmap[st:ed+1].astype(np.uint8)
                
                videos_to_run.extend([gabor_vid, mesi_vid, nat_vid])
                
                g_row = grat_map[(s, sc, rid)]
                resp_cols = [f'resp_{x}' for x in range(16)]
                g_max = np.max([g_row[c] for c in resp_cols])
                
                medi_val = medi_map[(s, sc, rid)]['resp']
                
                meta_info.append({
                    'session': s, 'scan_idx': sc, 'unit_id': uid, 'readout_id': rid,
                    'brain_area': area, 'layer': layer, 'model_unit_idx': model_unit_idx,
                    'g_max': g_max, 'medi_val': medi_val
                })
                
            batch_videos = np.stack(videos_to_run, axis=0)
            batch_resps = extract_model_response_batch(model, batch_videos, device)
            
            for v_idx, meta in enumerate(meta_info):
                base_idx = v_idx * 3
                model_idx = meta['model_unit_idx']
                
                gabor_resp = float(batch_resps[base_idx + 0, model_idx])
                mesi_resp = float(batch_resps[base_idx + 1, model_idx])
                nat_resp = float(batch_resps[base_idx + 2, model_idx])
                
                results_records.append({
                    'session': meta['session'], 'scan_idx': meta['scan_idx'], 
                    'unit_id': meta['unit_id'], 'readout_id': meta['readout_id'],
                    'brain_area': meta['brain_area'], 'layer': meta['layer'],
                    'MEDI': meta['medi_val'], 'Gabor': gabor_resp, 'MESI': mesi_resp, 
                    'Grating': meta['g_max'], 'Natural': nat_resp
                })
            pd.DataFrame(results_records).to_csv(OUTPUT_CSV, index=False)
                
        del model
        gc.collect()
        if device == "cuda": torch.cuda.empty_cache()

    print(f"Done processing all groups.")
    
if __name__ == "__main__":
    main()
