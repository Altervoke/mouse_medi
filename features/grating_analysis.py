import os
import sys
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from scipy.optimize import least_squares

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
fnn_root = os.path.join(project_root, 'fnn')

if project_root not in sys.path:
    sys.path.append(project_root)
if fnn_root not in sys.path:
    sys.path.append(fnn_root)

from medi_pipeline.config import paths
from fnn import microns

INPUT_CSV = os.path.join(getattr(paths, 'DATA_DIR'), 'neuron_info.csv')
OUTPUT_CSV = os.path.join(getattr(paths, 'DATA_DIR'), 'grating_features.csv')
SRC_DIR = getattr(paths, 'SRC_DIR')

def generate_drifting_grating_stimulus(num_directions=16, num_frames=60, height=144, width=256, 
                                       spatial_frequency=0.04, temporal_frequency=2.0, fps=30, contrast=1.0):
    stimulus_sequences = []
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    
    for dir_idx in range(num_directions):
        theta_deg = dir_idx * (360.0 / num_directions)
        theta_rad = -np.deg2rad(theta_deg)
        
        kx = spatial_frequency * np.cos(theta_rad)
        ky = spatial_frequency * np.sin(theta_rad)
        
        sequence = []
        for t in range(num_frames):
            time_sec = t / fps
            phase = 2 * np.pi * temporal_frequency * time_sec
            grating = 0.5 + 0.5 * contrast * np.cos(2 * np.pi * (kx * x + ky * y) - phase)
            sequence.append(grating)
            
        stimulus_sequences.append(np.array(sequence))
        
    return np.array(stimulus_sequences)

def get_stimulus_response_batch(model, stimuli):
    N_dirs, T, H, W = stimuli.shape
    
    stim_uint8 = (np.clip(stimuli, 0, 1) * 255).astype(np.uint8)
    batch_input = np.transpose(stim_uint8, (1, 0, 2, 3))[..., None]
    
    with torch.no_grad():
        gen = model.generate_response(batch_input, reset=True)
        all_frame_resps = []
        for resp in gen:
            all_frame_resps.append(resp)
            
    all_frame_resps = np.stack(all_frame_resps, axis=0)
    
    if T > 10:
        mean_resps = np.mean(all_frame_resps[10:], axis=0)
    else:
        mean_resps = np.mean(all_frame_resps, axis=0)
        
    return mean_resps

def von_mises_double(theta, mu, kappa, alpha, beta, gamma):
    return (alpha * np.exp(kappa * np.cos(theta - mu)) + 
            beta * np.exp(kappa * np.cos(theta - mu + np.pi)) + gamma)

def fit_tuning_curve(directions, responses):
    if len(responses) == 0: return None
    
    max_idx = np.argmax(responses)
    mu_guess = directions[max_idx]
    gamma_guess = np.min(responses)
    if gamma_guess < 0: gamma_guess = 0
    
    amp_guess = np.max(responses) - gamma_guess
    if amp_guess < 1e-5: amp_guess = 1.0
    
    x0 = [mu_guess, 1.0, amp_guess, amp_guess/2, gamma_guess]
    
    lower_bounds = [-np.inf, 0, 0, 0, -np.inf]
    upper_bounds = [np.inf, 50, np.inf, np.inf, np.inf]
    
    def residuals(params, x_data, y_data):
        mu, kappa, alpha, beta, gamma = params
        model = (alpha * np.exp(kappa * np.cos(x_data - mu)) + 
                 beta * np.exp(kappa * np.cos(x_data - mu + np.pi)) + gamma)
        return model - y_data

    try:
        res = least_squares(residuals, x0, args=(directions, responses),bounds=(lower_bounds, upper_bounds), max_nfev=5000)
        popt = res.x
    except Exception:
        return None
        
    return popt

def calculate_indices(responses, directions_rad):
    r_k = responses
    theta_k = directions_rad
    sum_r = np.sum(r_k) + 1e-12
    
    gOSI = np.abs(np.sum(r_k * np.exp(1j * 2 * theta_k))) / sum_r
    gDSI = np.abs(np.sum(r_k * np.exp(1j * theta_k))) / sum_r
    
    best_idx = np.argmax(r_k)
    
    num_dirs = len(theta_k)
    step = num_dirs // 4 
    
    oppo_idx = (best_idx + num_dirs // 2) % num_dirs
    ortho1_idx = (best_idx + step) % num_dirs
    ortho2_idx = (best_idx - step) % num_dirs
    
    R_pref = r_k[best_idx]
    R_oppo = r_k[oppo_idx]
    R_ortho = (r_k[ortho1_idx] + r_k[ortho2_idx]) / 2.0
    
    OSI = (R_pref - R_ortho) / (R_pref + R_ortho + 1e-12)
    DSI = (R_pref - R_oppo) / (R_pref + R_oppo + 1e-12)
    
    return OSI, DSI, gOSI, gDSI

def process_single_unit(args):
    idx, tuning_curve, tf, sf, directions_rad = args
    y_data = tuning_curve
    fit_params = fit_tuning_curve(directions_rad, y_data)
    osi, dsi, gosi, gdsi = calculate_indices(y_data, directions_rad)
    
    res = {
        'idx': idx,
        'y_data': y_data,
        'metrics': {
            'pref_tf': tf,
            'pref_sf': sf,
            'OSI': osi, 'DSI': dsi, 'gOSI': gosi, 'gDSI': gdsi
        }
    }
    
    if fit_params is not None:
        mu, kappa, alpha, beta, gamma = fit_params
        pref_dir_deg = np.rad2deg(mu % (2*np.pi))
        pref_ori_deg = np.rad2deg((mu % (2*np.pi)) % np.pi)
        res['metrics']['pref_dir'] = pref_dir_deg
        res['metrics']['pref_ori'] = pref_ori_deg
    else:
        best_dir_idx = np.argmax(y_data)
        pd_rad = directions_rad[best_dir_idx]
        res['metrics']['pref_dir'] = np.rad2deg(pd_rad)
        res['metrics']['pref_ori'] = np.rad2deg(pd_rad % np.pi)
    return res

def main():
    print("Loading Neuron Info...")
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return
        
    df = pd.read_csv(INPUT_CSV)
    
    existing_keys = set()
    if os.path.exists(OUTPUT_CSV):
        try:
            existing_df = pd.read_csv(OUTPUT_CSV)
            if all(col in existing_df.columns for col in ['session', 'scan_idx', 'readout_id']):
                for _, r in existing_df.iterrows():
                    existing_keys.add((int(r['session']), int(r['scan_idx']), int(r['readout_id'])))
                print(f"Found {len(existing_keys)} existing records in {OUTPUT_CSV}. Will skip them.")
        except Exception as e:
            print(f"Warning: Could not read existing output file: {e}")

    groups = df.groupby(['session', 'scan_idx'])
    
    results = []
    if existing_keys:
        results = existing_df.to_dict('records')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Loading models from: {SRC_DIR}")

    for (session, scan), group_df in groups:
        session_val = int(session)
        scan_val = int(scan)
        
        mask = group_df.apply(lambda r: (session_val, scan_val, int(r['readout_id'])) not in existing_keys, axis=1)
        current_group_df = group_df[mask]
        
        if current_group_df.empty:
            print(f"\nSkipping Session {session_val}, Scan {scan_val} (All {len(group_df)} neurons already processed).")
            continue
            
        print(f"\nProcessing Session {session_val}, Scan {scan_val} ({len(current_group_df)}/{len(group_df)} neurons)...")
        
        try:
            model, neuron_ids_df = microns.scan(
                session=int(session), 
                scan_idx=int(scan), 
                directory=SRC_DIR
            )
            model = model.to(device).eval()
            for p in model.parameters(): p.requires_grad = False
        except Exception as e:
            print(f"Failed to load model {session}/{scan}: {e}")
            continue
            
        if 'readout_id' in neuron_ids_df.columns:
            model_rids = neuron_ids_df['readout_id'].values
        else:
            model_rids = neuron_ids_df.index.values
            
        rid_to_idx = {rid: i for i, rid in enumerate(model_rids)}
        
        target_indices = [rid_to_idx[rid] for rid in current_group_df['readout_id'] if rid in rid_to_idx]
        if not target_indices:
            continue
            
        num_units = len(model_rids)
        
        tfs = [0.5, 1.0, 2.0, 4.0, 8.0]
        sfs = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32]
        
        print(f"  Sweeping 30 combinations for {len(target_indices)} units...")
        
        best_max_resp = np.full(num_units, -np.inf)
        best_tuning_curves = np.zeros((num_units, 16))
        best_tf = np.zeros(num_units)
        best_sf = np.zeros(num_units)
        
        for tf in tqdm(tfs, desc="TFs"):
            for sf in tqdm(sfs, desc="SFs", leave=False):
                stim = generate_drifting_grating_stimulus(
                    temporal_frequency=tf, spatial_frequency=sf,
                    num_directions=16, num_frames=60, contrast=1.0
                )
                resps_all = get_stimulus_response_batch(model, stim)
                
                max_resp_current = np.max(resps_all, axis=0)
                better_mask = (max_resp_current > best_max_resp)
                
                best_max_resp[better_mask] = max_resp_current[better_mask]
                best_tuning_curves[better_mask] = resps_all[:, better_mask].T
                best_tf[better_mask] = tf
                best_sf[better_mask] = sf

        print("Calculating Metrics...")
        final_metrics = {}
        final_responses = {}
        directions_rad = np.linspace(0, 2*np.pi, 16, endpoint=False)
        
        from concurrent.futures import ProcessPoolExecutor
        
        task_args = [
            (i, best_tuning_curves[i], best_tf[i], best_sf[i], directions_rad)
            for i in target_indices
        ]

        with ProcessPoolExecutor() as executor:
            unit_results = list(tqdm(executor.map(process_single_unit, task_args), 
                                    total=len(target_indices), desc="Processing Metrics"))
            
        for res in unit_results:
            idx = res['idx']
            final_metrics[idx] = res['metrics']
            final_responses[idx] = res['y_data']

        for _, row in current_group_df.iterrows():
            rid = int(row['readout_id'])
            
            if rid in rid_to_idx:
                idx = rid_to_idx[rid]
                if idx in final_metrics:
                    m = final_metrics[idx]
                    record = {
                        'session': int(session),
                        'scan_idx': int(scan),
                        'unit_id': int(row['unit_id']),
                        'readout_id': rid,
                        'pref_tf': m['pref_tf'],
                        'pref_sf': m['pref_sf'],
                        'pref_ori': m['pref_ori'],
                        'pref_dir': m['pref_dir'],
                        'OSI': m['OSI'],
                        'DSI': m['DSI'],
                        'gOSI': m['gOSI'],
                        'gDSI': m['gDSI']
                    }
                    
                    if idx in final_responses:
                        for i, val in enumerate(final_responses[idx]):
                            record[f'resp_{i}'] = val

                    results.append(record)

    if results:
        df_out = pd.DataFrame(results)
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        
        base_cols = ['session', 'scan_idx', 'unit_id', 'readout_id', 
                     'pref_tf', 'pref_sf', 'pref_ori', 'pref_dir', 
                     'OSI', 'DSI', 'gOSI', 'gDSI']
        resp_cols = [f'resp_{i}' for i in range(16)]
        cols = base_cols + resp_cols
        cols_final = [c for c in cols if c in df_out.columns]
        df_out = df_out[cols_final]
        
        float_cols = ['pref_tf', 'pref_sf', 'pref_ori', 'pref_dir', 'OSI', 'DSI', 'gOSI', 'gDSI'] + resp_cols

        for c in float_cols:
            if c in df_out.columns:
                df_out[c] = df_out[c].round(10)
        
        df_out.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved results for {len(df_out)} neurons to {OUTPUT_CSV}")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
