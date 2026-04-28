import os
import sys
import pandas as pd
import json
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))

if project_root not in sys.path:
    sys.path.append(project_root)

from medi_pipeline.config import paths
from medi_pipeline.generation.generator import load_foundation_model, generate_gabor_func, generate_mesi_func, load_vae

ORDER_CSV = os.path.join(getattr(paths, 'DATA_DIR'), 'execution_order.csv')

def get_progress_file():
    base = os.path.join(getattr(paths, 'RESULTS_DIR'), 'Gabor')
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, 'progress_status.json')

def get_mesi_progress_file():
    base = os.path.join(getattr(paths, 'RESULTS_DIR'), 'MESI')
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, 'progress_status.json')

def load_progress_set():
    progress_file = get_progress_file()
    completed = set()
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                data = json.load(f)
                for item in data.get('completed', []):
                    if len(item) >= 3:
                        completed.add(f"{int(item[0])}_{int(item[1])}_{int(item[2])}")
        except Exception as e:
            print(f"Warning: Failed to parse progress file: {e}")
            
    print(f"Loaded {len(completed)} completed entries from {progress_file}.")
    return completed

def load_mesi_progress_set():
    progress_file = get_mesi_progress_file()
    completed = set()
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                data = json.load(f)
                for item in data.get('completed', []):
                    if len(item) >= 3:
                        completed.add(f"{int(item[0])}_{int(item[1])}_{int(item[2])}")
        except Exception as e:
            print(f"Warning: Failed to parse MESI progress file: {e}")
    return completed

def save_progress_append(session, scan, rid):
    progress_file = get_progress_file()
    current_data = {'completed': []}
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                current_data = json.load(f)
        except:
            pass
    
    current_data['completed'].append([int(session), int(scan), int(rid)])
    
    temp_file = progress_file + ".tmp"
    with open(temp_file, 'w') as f:
        json.dump(current_data, f)
    os.replace(temp_file, progress_file)

def save_mesi_progress_append(session, scan, rid):
    progress_file = get_mesi_progress_file()
    current_data = {'completed': []}
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                current_data = json.load(f)
        except:
            pass
    
    current_data['completed'].append([int(session), int(scan), int(rid)])
    
    temp_file = progress_file + ".tmp"
    with open(temp_file, 'w') as f:
        json.dump(current_data, f)
    os.replace(temp_file, progress_file)

import argparse

def main():
    if not os.path.exists(ORDER_CSV):
        print(f"Order file not found at {ORDER_CSV}. Please run run_generation.py to generate it.")
        return

    df_exec = pd.read_csv(ORDER_CSV)
    print(f"Total entries in execution plan: {len(df_exec)}")

    completed_keys = load_progress_set()
    completed_mesi_keys = load_mesi_progress_set()

    df_exec['key'] = df_exec.apply(lambda x: f"{int(x['session'])}_{int(x['scan_idx'])}_{int(x['readout_id'])}", axis=1)
    df_todo = df_exec[~df_exec['key'].isin(completed_keys)]
    
    queue = df_todo.to_dict('records')
    print(f"Remaining tasks: {len(queue)}")
    
    if not queue:
        print(f"All tasks completed!")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    current_model_key = None
    model = None
    vae = None
    
    for i, n in enumerate(queue):
        s = int(n['session'])
        sc = int(n['scan_idx'])
        rid = int(n['readout_id'])
        area = n['brain_area']
        layer = n['layer']
        
        if current_model_key != (s, sc):
            model = None
            if device == 'cuda':
                torch.cuda.empty_cache()
            
            try:
                model = load_foundation_model(s, sc, device=device)
                current_model_key = (s, sc)
            except Exception as e:
                print(f"Failed to load model {s}/{sc}: {e}")
                continue
        
        print(f"\n=============================================")
        print(f"[{i+1}/{len(queue)}] Processing {area}/{layer} (ID:{rid})")
        print(f"=============================================")
        
        try:
            # Check if MESI prior exists
            base_mesi = os.path.join(getattr(paths, 'RESULTS_DIR'), 'MESI')
            mesi_path = os.path.join(base_mesi, area, layer, f"{s}_{sc}_r{rid}.png")
            
            if not os.path.exists(mesi_path) or f"{s}_{sc}_{rid}" not in completed_mesi_keys:
                print(f"   [!] MESI prior missing for {s}_{sc}_r{rid}. Fast-forwarding generation of next 10 MESI...")
                if vae is None:
                    vae = load_vae(device=device)
                
                lookahead = queue[i:i+10]
                
                for idx_ahead, n_ahead in enumerate(lookahead):
                    s_ahead = int(n_ahead['session'])
                    sc_ahead = int(n_ahead['scan_idx'])
                    rid_ahead = int(n_ahead['readout_id'])
                    area_ahead = n_ahead['brain_area']
                    layer_ahead = n_ahead['layer']
                    
                    ahead_mesi_path = os.path.join(base_mesi, area_ahead, layer_ahead, f"{s_ahead}_{sc_ahead}_r{rid_ahead}.png")
                    key_ahead = f"{s_ahead}_{sc_ahead}_{rid_ahead}"
                    
                    if not os.path.exists(ahead_mesi_path) or key_ahead not in completed_mesi_keys:
                        if current_model_key != (s_ahead, sc_ahead):
                            if device == 'cuda': torch.cuda.empty_cache()
                            model = load_foundation_model(s_ahead, sc_ahead, device=device)
                            current_model_key = (s_ahead, sc_ahead)
                            
                        generate_mesi_func(
                            session=s_ahead,
                            scan=sc_ahead,
                            readout_id=rid_ahead,
                            brain_area=area_ahead,
                            layer=layer_ahead,
                            model=model,
                            vae=vae,
                            device=device
                        )
                        save_mesi_progress_append(s_ahead, sc_ahead, rid_ahead)
                        completed_mesi_keys.add(key_ahead)
                
                if current_model_key != (s, sc):
                    if device == 'cuda': torch.cuda.empty_cache()
                    model = load_foundation_model(s, sc, device=device)
                    current_model_key = (s, sc)
                
                print(f"   [✓] Fast-forwarded MESI constraint generation!")
        
            generate_gabor_func(
                session=s,
                scan=sc,
                readout_id=rid,
                brain_area=area,
                layer=layer,
                model=model,
                device=device
            )
            save_progress_append(s, sc, rid)
        except Exception as e:
            print(f"Failed generation for {rid}: {e}")

if __name__ == "__main__":
    main()
