import os
import sys
import pandas as pd
import numpy as np
import random
import json
import torch
from collections import defaultdict

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))

if project_root not in sys.path:
    sys.path.append(project_root)

from medi_pipeline.config import paths
from medi_pipeline.generation.generator import generate_mesi_func, generate_medi_func, load_foundation_model, load_vae

ORDER_CSV = os.path.join(getattr(paths, 'DATA_DIR'), 'execution_order.csv')
NEURON_CSV = os.path.join(getattr(paths, 'DATA_DIR'), 'neuron_info.csv')

def get_progress_file(is_mesi=False):
    if is_mesi:
        base = os.path.join(getattr(paths, 'RESULTS_DIR'), 'MESI')
    else:
        base = getattr(paths, 'RESULTS_MEDI_DIR')
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, 'progress_status.json')

def load_progress_set(is_mesi=False):
    progress_file = get_progress_file(is_mesi)
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

def save_progress_append(session, scan, rid, is_mesi=False):
    progress_file = get_progress_file(is_mesi)
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

def generate_execution_order_csv(df):
    print("Generating execution order...")
    random.seed(42)
    np.random.seed(42)
    
    buckets = defaultdict(list)
    
    for _, row in df.iterrows():
        area = str(row['brain_area'])
        layer = str(row['layer'])
        buckets[(area, layer)].append(row.to_dict())
        
    cycle_keys = sorted(buckets.keys())
    print(f"Cycling through {len(cycle_keys)} groups: {cycle_keys}")
    
    for k in buckets:
        random.shuffle(buckets[k])
        
    execution_list = []
    
    while True:
        neurons_popped = 0
        
        for k in cycle_keys:
            if buckets[k]:
                neuron = buckets[k].pop(0)
                execution_list.append(neuron)
                neurons_popped += 1
                
        if neurons_popped == 0:
            break
            
    df_order = pd.DataFrame(execution_list)
    cols = ['session', 'scan_idx', 'readout_id', 'brain_area', 'layer', 'unit_id']
    save_cols = [c for c in cols if c in df_order.columns]
    
    os.makedirs(os.path.dirname(ORDER_CSV), exist_ok=True)
    df_order[save_cols].to_csv(ORDER_CSV, index=False)
    print(f"Saved strict round-robin order to {ORDER_CSV} ({len(df_order)} neurons)")

import argparse

def main():
    parser = argparse.ArgumentParser(description="Run Generation Pipeline")
    parser.add_argument("--mode", type=str, choices=["medi", "mesi"], default="medi", help="Which generation to run (medi or mesi)")
    args = parser.parse_args()

    if not os.path.exists(ORDER_CSV):
        print(f"Order file not found at {ORDER_CSV}. Generating...")

        df_source = pd.read_csv(NEURON_CSV)
        required = ['session', 'scan_idx', 'readout_id', 'brain_area', 'layer']
        df_source = df_source.dropna(subset=required)
        
        generate_execution_order_csv(df_source)
    else:
        print(f"Found existing execution order at {ORDER_CSV}")

    df_exec = pd.read_csv(ORDER_CSV)
    print(f"Total entries in execution plan: {len(df_exec)}")

    completed_keys_medi = load_progress_set(is_mesi=False) if args.mode == 'medi' else set()
    completed_keys_mesi = load_progress_set(is_mesi=True) if args.mode == 'mesi' else set()

    df_exec['key'] = df_exec.apply(lambda x: f"{int(x['session'])}_{int(x['scan_idx'])}_{int(x['readout_id'])}", axis=1)
    
    if args.mode == 'medi':
        df_todo = df_exec[~df_exec['key'].isin(completed_keys_medi)]
    else:
        df_todo = df_exec[~df_exec['key'].isin(completed_keys_mesi)]
    
    queue = df_todo.to_dict('records')
    print(f"Remaining tasks: {len(queue)}")
    
    if not queue:
        print(f"All tasks completed!")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    try:
        vae = load_vae(device=device)
    except Exception as e:
        print(f"Failed to load VAE: {e}")
        return

    current_model_key = None
    model = None
    total_generated = 0
    
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
        
        print(f"[{i+1}/{len(queue)}] {area}/{layer} (ID:{rid})")
        
        todo_medi = args.mode == 'medi' and (n['key'] not in completed_keys_medi)
        todo_mesi = args.mode == 'mesi' and (n['key'] not in completed_keys_mesi)
        
        try:
            if todo_medi:
                generate_medi_func(
                    session=s,
                    scan=sc,
                    readout_id=rid,
                    brain_area=area,
                    layer=layer,
                    model=model,
                    vae=vae,
                    device=device
                )
                save_progress_append(s, sc, rid, is_mesi=False)
                
            if todo_mesi:
                generate_mesi_func(
                    session=s,
                    scan=sc,
                    readout_id=rid,
                    brain_area=area,
                    layer=layer,
                    model=model,
                    vae=vae,
                    device=device
                )
                save_progress_append(s, sc, rid, is_mesi=True)
            
            total_generated += 1
            
        except KeyboardInterrupt:
            print("\nInterrupted.")
            sys.exit(0)
        except Exception as e:
            print(f"Error on {rid}: {e}")
            
    print("\nJob Done.")

if __name__ == "__main__":
    main()
