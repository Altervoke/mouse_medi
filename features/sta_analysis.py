import os
import sys
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
    
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
OUTPUT_CSV = os.path.join(getattr(paths, 'DATA_DIR'), 'sta_features.csv')
SRC_DIR = getattr(paths, 'SRC_DIR')

class FlashingGaussianDots:
    def __init__(self, resolution=(144, 256), grid_size=(36, 64), sigma=6.0, duration_frames=9, seed=42):
        self.resolution = resolution
        self.grid_size = grid_size
        self.sigma = sigma
        self.duration_frames = duration_frames
        self.background = 128
        self.rng = np.random.default_rng(seed)
        
        h, w = resolution
        rows, cols = grid_size
        y_centers = np.linspace(h / (2 * rows), h - h / (2 * rows), rows)
        x_centers = np.linspace(w / (2 * cols), w - w / (2 * cols), cols)
        self.positions = np.array([(y, x) for y in y_centers for x in x_centers])

    def generate_single(self, idx):
        h, w = self.resolution
        n_white = len(self.positions)
        if idx < n_white:
            pos = self.positions[idx]
            pol = 1
        else:
            pos = self.positions[idx - n_white]
            pol = -1
            
        y, x = np.ogrid[:h, :w]
        cy, cx = pos
        
        blob = np.exp(-((y - cy)**2 + (x - cx)**2) / (2 * self.sigma**2))
        
        if pol > 0:
            frame = self.background + 127 * blob
        else:
            frame = self.background - 128 * blob
        
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        seq = np.tile(frame[None], (self.duration_frames, 1, 1))
        
        return seq, (frame.astype(np.float32) - 128.0)

    def get_num_stimuli(self):
        return 2 * len(self.positions)


def gaussian_2d(coords, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = coords
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2 + 1e-9) + (np.sin(theta)**2)/(2*sigma_y**2 + 1e-9)
    b = -(np.sin(2*theta))/(4*sigma_x**2 + 1e-9) + (np.sin(2*theta))/(4*sigma_y**2 + 1e-9)
    c = (np.sin(theta)**2)/(2*sigma_x**2 + 1e-9) + (np.cos(theta)**2)/(2*sigma_y**2 + 1e-9)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g.ravel()

def fit_2d_gaussian(active_map):
    H_map, W_map = active_map.shape
    x = np.linspace(0, W_map-1, W_map)
    y = np.linspace(0, H_map-1, H_map)
    x_grid, y_grid = np.meshgrid(x, y)
    
    idx_max = np.argmax(active_map)
    y_max, x_max = np.unravel_index(idx_max, active_map.shape)
    amp_max = active_map[y_max, x_max]
    
    n_active = np.sum(active_map > (active_map.max() * 0.5))
    guess_sigma = np.sqrt(n_active / np.pi) if n_active > 0 else 5.0
    guess_sigma = np.clip(guess_sigma, 2.0, W_map/4.0)

    initial_guess = (amp_max, x_max, y_max, guess_sigma, guess_sigma, 0, np.mean(active_map))
    bounds = ([0, 0, 0, 1.0, 1.0, -np.pi, 0], 
              [max(1.0, amp_max*2), W_map, H_map, W_map, H_map, np.pi, max(1.0, amp_max)])
    
    try:
        popt, _ = curve_fit(gaussian_2d, (x_grid, y_grid), active_map.ravel(), p0=initial_guess,
                            bounds=bounds)
        return {
            'cx': float(popt[1]), 'cy': float(popt[2]), 
            'sigma_x': float(popt[3]), 'sigma_y': float(popt[4]),
            'theta': float(popt[5]), 'amp': float(popt[0])
        }
    except Exception:
        return None

def estimate_rf_from_map(rf_map):
    base_map = np.abs(rf_map - np.median(rf_map))
    H_vid, W_vid = base_map.shape
    sigma = 1.0 * (W_vid / 64.0)
    active_map = gaussian_filter(base_map, sigma=sigma)
    gauss_res = fit_2d_gaussian(active_map)
    return gauss_res

def calculate_sta(model, stimulus_gen):
    N_stim = stimulus_gen.get_num_stimuli()
    BATCH_SIZE = 10
    BURNIN_FRAMES = 10
    
    # Determine Output Size (N_neurons) by running one dummy batch
    print(f"Initializing STA (Total {N_stim} stimuli)...")
    
    burnin_seq = np.full((BURNIN_FRAMES, 144, 256, 1), 128, dtype=np.uint8)
    model.eval()
    dummy_seq, _ = stimulus_gen.generate_single(0)
    dummy_seq = dummy_seq[None, ..., None]
    
    with torch.no_grad():
        batch_burnin = np.tile(burnin_seq[None], (1, 1, 1, 1, 1))
        batch = np.concatenate([batch_burnin, dummy_seq], axis=1)
        batch_t = batch.transpose(1, 0, 2, 3, 4)
        
        gen = model.generate_response(batch_t, reset=True)
        out = None
        for outputs in gen:
            out = outputs
        
        if hasattr(out, "cpu"): out = out.cpu().numpy()
        N_neurons = out.shape[1]
        
    H, W = stimulus_gen.resolution
    
    # Initialize STA Accumulators
    sta_accum = np.zeros((N_neurons, H, W), dtype=np.float32)
    count_accum = np.zeros(N_neurons, dtype=np.float32)
    
    with torch.no_grad():
        for i in tqdm(range(0, N_stim, BATCH_SIZE), desc="STA Stream"):
            current_batch_size = min(BATCH_SIZE, N_stim - i)
            
            # 1. Generate Batch Data
            batch_seqs = []
            frames_mat = np.zeros((current_batch_size, H*W), dtype=np.float32)
            
            for k in range(current_batch_size):
                idx = i + k
                seq, frame_ref = stimulus_gen.generate_single(idx)
                batch_seqs.append(seq)
                frames_mat[k, :] = frame_ref.ravel()
            
            # 2. Run Model Inference
            input_stim = np.stack(batch_seqs)
            input_stim = input_stim[..., None]
            batch_burnin = np.tile(burnin_seq[None], (current_batch_size, 1, 1, 1, 1))
            batch = np.concatenate([batch_burnin, input_stim], axis=1)
            batch_t = batch.transpose(1, 0, 2, 3, 4)
            
            b_resps = []
            gen_resp = model.generate_response(batch_t, reset=True)
            for t, outputs in enumerate(gen_resp):
                if t >= BURNIN_FRAMES:
                    o = outputs.cpu().numpy() if hasattr(outputs, "cpu") else outputs
                    b_resps.append(o)

            mean_resp = np.mean(b_resps, axis=0).astype(np.float32)
            
            # 3. Update STA Accumulators
            N_CHUNK = 500
            resp_T = mean_resp.T
            
            for nc in range(0, N_neurons, N_CHUNK):
                n_end = min(nc + N_CHUNK, N_neurons)
                r_chunk = resp_T[nc:n_end, :] 
                update_chunk = np.dot(r_chunk, frames_mat)
                sta_accum[nc:n_end] += update_chunk.reshape(-1, H, W)
            
            count_accum += mean_resp.sum(axis=0)
    
    count_accum[count_accum == 0] = 1e-8
    sta_accum /= count_accum[:, None, None]
    
    return sta_accum

def calculate_ssi(map):
    """
    Compute Spatial Selectivity Index for a receptive field map.
    
    Higher SSI = more spatially selective (tighter RF)
    Lower SSI = less selective (broader RF)
    """
    h, w = map.shape
    
    size = max(h, w)
    x_coords = np.linspace(0, (w-1)/size, w)
    y_coords = np.linspace(0, (h-1)/size, h)
    x, y = np.meshgrid(x_coords, y_coords, indexing='xy')
    
    rf_abs = np.abs(map)
    sum_val = rf_abs.sum()
    if sum_val == 0:
        return np.nan

    z = rf_abs / sum_val
    mu_x = (z * x).sum()
    mu_y = (z * y).sum()
    x_c = x - mu_x
    y_c = y - mu_y
    
    cov_xx = (z * x_c * x_c).sum()
    cov_xy = (z * x_c * y_c).sum()
    cov_yy = (z * y_c * y_c).sum()
    det_cov = cov_xx * cov_yy - cov_xy**2
    
    if det_cov <= 0:
        return np.nan
    
    return -np.log(det_cov)

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
            required_cols = ['session', 'scan_idx', 'readout_id']
            if all(col in existing_df.columns for col in required_cols):
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
    
    feature_keys = ['rf_x', 'rf_y', 'rf_sigma_x', 'rf_sigma_y', 'rf_theta', 'SSI']
    
    print("Initializing Stimulus Generator...")
    gen = FlashingGaussianDots(resolution=(144, 256), grid_size=(36, 64))
    
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
                session=session_val, 
                scan_idx=scan_val, 
                directory=SRC_DIR
            )
            model = model.to(device).eval()
            for p in model.parameters(): p.requires_grad = False
        except Exception as e:
            print(f"Failed to load model {session_val}/{scan_val}: {e}")
            continue
            
        if 'readout_id' in neuron_ids_df.columns:
            model_rids = neuron_ids_df['readout_id'].values
        else:
            model_rids = neuron_ids_df.index.values
            
        rid_to_idx = {rid: i for i, rid in enumerate(model_rids)}
        
        sta_maps = calculate_sta(model, gen)
        
        for _, row in tqdm(current_group_df.iterrows(), total=len(current_group_df), desc="Fitting RFs"):
            rid = int(row['readout_id'])
            
            if rid not in rid_to_idx: continue
            
            idx = rid_to_idx[rid]
            rf_map = sta_maps[idx]
            
            fit = estimate_rf_from_map(rf_map)
            ssi_val = calculate_ssi(rf_map)
            
            res_entry = {
                'session': session_val,
                'scan_idx': scan_val,
                'unit_id': int(row['unit_id']),
                'readout_id': rid
            }
            
            if fit:
                W_2, H_2 = 128.0, 72.0
                res_entry['rf_x'] = (fit['cx'] / W_2) - 1.0
                res_entry['rf_y'] = (fit['cy'] / H_2) - 1.0
                res_entry['rf_sigma_x'] = fit['sigma_x']
                res_entry['rf_sigma_y'] = fit['sigma_y']
                res_entry['rf_theta'] = fit['theta']
                res_entry['SSI'] = ssi_val
            else:
                for k in feature_keys: 
                    if k == 'SSI':
                        res_entry[k] = ssi_val
                    else:
                        res_entry[k] = np.nan
                
            results.append(res_entry)

    if results:
        df_out = pd.DataFrame(results)
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        cols = ['session', 'scan_idx', 'unit_id', 'readout_id'] + feature_keys
        df_out = df_out[cols]
        
        for c in feature_keys:
            if c in df_out.columns:
                df_out[c] = df_out[c].round(10)
                
        df_out.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved results for {len(df_out)} neurons to {OUTPUT_CSV}")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
