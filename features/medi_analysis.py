import os
import sys
import glob
import numpy as np
import pandas as pd
import torch
import imageio
import cv2
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
fnn_root = os.path.join(project_root, 'fnn')

if project_root not in sys.path:
    sys.path.append(project_root)
if fnn_root not in sys.path:
    sys.path.append(fnn_root)

from medi_pipeline.config import paths
from medi_pipeline.features.grating_analysis import fit_tuning_curve, calculate_indices
from medi_pipeline.features.sta_analysis import fit_2d_gaussian, calculate_ssi
from fnn import microns

RESULTS_MEDI_DIR = getattr(paths, 'RESULTS_MEDI_DIR')
DATA_DIR = getattr(paths, 'DATA_DIR')
FEATURES_CSV = os.path.join(DATA_DIR, 'medi_features.csv')
GRATING_CSV = os.path.join(DATA_DIR, 'grating_features.csv')
SRC_DIR = getattr(paths, 'SRC_DIR')

def estimate_rf_from_mei(video):
    v_min, v_max = video.min(), video.max()
    base_map = (np.max(video, axis=0) - np.min(video, axis=0)) / (v_max - v_min + 1e-9)
    H_vid, W_vid = base_map.shape
    sigma = 1.0 * (W_vid / 64.0)
    active_map = gaussian_filter(base_map, sigma=sigma)
    gauss_res = fit_2d_gaussian(active_map)
    return gauss_res, active_map

def estimate_direction_3d_fft(video):
    T, H, W = video.shape
    v_fft = np.fft.fftshift(np.fft.fftn(video))
    mag = np.abs(v_fft)**2
    wt = np.fft.fftshift(np.fft.fftfreq(T, d=1/30))
    ky = np.fft.fftshift(np.fft.fftfreq(H))
    kx = np.fft.fftshift(np.fft.fftfreq(W))
    WT, KY, KX = np.meshgrid(wt, ky, kx, indexing='ij')
    k_mag = np.sqrt(KX**2 + KY**2)
    mask = (np.abs(WT) > 0) & (k_mag >= 0.01) & (k_mag <= 0.32)
    angs = np.rad2deg(np.arctan2(KY[mask]*WT[mask], -KX[mask]*WT[mask])) % 360
    h, _ = np.histogram(angs, bins=np.arange(361), weights=mag[mask])
    return gaussian_filter(h, sigma=20, mode='wrap')

def estimate_sftf(video):
    T, H, W = video.shape
    v_fft = np.fft.fftshift(np.fft.fftn(video))
    mag = np.abs(v_fft)**2
    wt = np.fft.fftshift(np.fft.fftfreq(T, d=1/30))
    ky = np.fft.fftshift(np.fft.fftfreq(H))
    kx = np.fft.fftshift(np.fft.fftfreq(W))
    WT, KY, KX = np.meshgrid(wt, ky, kx, indexing='ij')
    k_mag = np.sqrt(KX**2 + KY**2)
    mask = (np.abs(WT) > 0) & (k_mag >= 0.01) & (k_mag <= 0.32)
    masked_mag = mag.copy()
    masked_mag[~mask] = 0
    t_p, y_p, x_p = np.unravel_index(np.argmax(masked_mag), mag.shape)
    tf_peak = np.abs(WT[t_p, y_p, x_p])
    sf_peak = k_mag[t_p, y_p, x_p]
    return float(sf_peak), float(tf_peak)

def estimate_insep(video):
    T, H, W = video.shape
    X = video.reshape(T, H * W)
    X = X - np.mean(X, axis=0)
    from scipy.linalg import svd
    _, s, _ = svd(X, full_matrices=False)
    STII = 1.0 - (s[0]**2 / (np.sum(s**2) + 1e-9))
    return float(STII)

def load_mei_video(video_path):
    try:
        reader = imageio.get_reader(video_path)
        frames = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) if f.ndim==3 else f, (256, 144)) for f in reader]
        return np.stack(frames).astype(np.float32) / 255.0
    except Exception as e:
        print(f"Error loading {video_path}: {e}")
        return None

def get_model_responses_batch(model, videos_list, readout_id):
    if not videos_list:
        return []
    
    processed_videos = []
    for v in videos_list:
        if v.dtype != np.uint8:
            v = (v * 255.0).clip(0, 255).astype(np.uint8)
        if v.ndim == 3:
            v = v[..., None]
        processed_videos.append(v)
    
    batch_videos = np.stack(processed_videos, axis=1) 
    gen = model.generate_response(batch_videos, reset=True)
    
    all_resps = []
    for resp in gen:
        all_resps.append(resp[:, readout_id])
            
    all_resps = np.array(all_resps)
    
    results = []
    for n in range(all_resps.shape[1]):
        res = all_resps[:, n]
        if len(res) > 10:
            results.append(np.mean(res[10:]))
        else:
            results.append(np.mean(res))
    return results

def main():
    if os.path.exists(FEATURES_CSV):
        features_df = pd.read_csv(FEATURES_CSV)
        processed_ids = set(zip(features_df['session'], features_df['scan_idx'], features_df['readout_id']))
    else:
        features_df = pd.DataFrame()
        processed_ids = set()
        
    if os.path.exists(GRATING_CSV):
        grating_df = pd.read_csv(GRATING_CSV)
    else:
        print(f"Warning: {GRATING_CSV} not found. Grating correlation will be skipped.")
        grating_df = None

    gif_paths = glob.glob(os.path.join(RESULTS_MEDI_DIR, "*", "*", "*.gif"))
    
    if not gif_paths:
        print("No GIFs found to process.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    new_records = []
    
    for gif_path in tqdm(gif_paths, desc="Analyzing MEDI"):
        filename = os.path.basename(gif_path)
        try:
            parts = filename.replace(".gif", "").split("_")
            session = int(parts[0])
            scan_idx = int(parts[1])
            readout_id = int(parts[-1][1:])
        except Exception:
            print(f"Skipping {filename}: could not parse session, scan_idx, or readout_id")
            continue
            
        if (session, scan_idx, readout_id) in processed_ids:
            continue
            
        model, scan_neurons = microns.scan(session=session, scan_idx=scan_idx, directory=SRC_DIR)
        model.to(device).eval()
        rid_to_idx = {rid: i for i, rid in enumerate(scan_neurons.index)}
            
        if readout_id not in rid_to_idx:
            print(f"Skipping {readout_id}: not found in model")
            continue
            
        model_idx = rid_to_idx[readout_id]
        unit_id = scan_neurons.iloc[model_idx].get('unit_id', np.nan)
            
        video = load_mei_video(gif_path)
        if video is None:
            continue
            
        record = {
            'session': session,
            'scan_idx': scan_idx,
            'unit_id': unit_id,
            'readout_id': readout_id
        }
        
        # 1. 3D-FFT Analysis
        hist_fft = estimate_direction_3d_fft(video)
        directions_rad = np.deg2rad(np.arange(360))
        osi, dsi, gosi, gdsi = calculate_indices(hist_fft, directions_rad)
        
        fit_params = fit_tuning_curve(directions_rad, hist_fft)
        if fit_params is not None:
            mu, kappa, alpha, beta, gamma = fit_params
            pref_dir_deg = np.rad2deg(mu % (2*np.pi))
            mei_pref_dir = pref_dir_deg
            mei_pref_ori = float(pref_dir_deg % 180)
        else:
            best_dir_idx = np.argmax(hist_fft)
            mei_pref_dir = float(best_dir_idx)
            mei_pref_ori = float(best_dir_idx % 180)
        
        mei_sf, mei_tf = estimate_sftf(video)
        insep_val = estimate_insep(video)
        
        record.update({
            'pref_tf': mei_tf,
            'pref_sf': mei_sf,
            'pref_ori': mei_pref_ori,
            'pref_dir': mei_pref_dir,
            'OSI': osi,
            'DSI': dsi,
            'gOSI': gosi,
            'gDSI': gdsi,
            'STII': insep_val
        })
        
        # 2. Grating Correlation
        if grating_df is not None:
            match = grating_df[(grating_df['session'] == session) & 
                               (grating_df['scan_idx'] == scan_idx) & 
                               (grating_df['readout_id'] == readout_id)]
            if not match.empty:
                g_keys = [f'resp_{i}' for i in range(16)]
                if all(k in match.columns for k in g_keys):
                    y_g = np.array([float(match.iloc[0][k]) for k in g_keys])
                    x = np.linspace(0, 360, 16, endpoint=False)
                    y_g_smooth = np.interp(np.linspace(0, 360, 360, endpoint=False), x, y_g)
                    
                    h_norm = hist_fft / (np.max(hist_fft) + 1e-9)
                    g_norm = y_g_smooth / (np.max(y_g_smooth) + 1e-9)
                    record['curve_corr'], _ = pearsonr(h_norm, g_norm)
                else:
                    record['curve_corr'] = np.nan
            else:
                record['curve_corr'] = np.nan
        else:
            record['curve_corr'] = np.nan
            
        # 3. RF Fitting
        rf_fit, rf_map_processed = estimate_rf_from_mei(video)
        half_w, half_h = 128, 72
        ssi_val = calculate_ssi(rf_map_processed)
        
        if rf_fit:
            record.update({
                'rf_x': (rf_fit['cx'] / half_w) - 1.0,
                'rf_y': (rf_fit['cy'] / half_h) - 1.0,
                'rf_sigma_x': rf_fit['sigma_x'],
                'rf_sigma_y': rf_fit['sigma_y'],
                'rf_theta': rf_fit['theta'],
                'SSI': ssi_val
            })
        else:
            record.update({
                'rf_x': np.nan, 'rf_y': np.nan,
                'rf_sigma_x': np.nan, 'rf_sigma_y': np.nan,
                'rf_theta': np.nan, 'SSI': ssi_val
            })
            
        # 4. Model Inference
        resps = get_model_responses_batch(model, [video], model_idx)
        record['resp'] = resps[0]
                
        new_records.append(record)
        processed_ids.add((session, scan_idx, readout_id))
        
        if len(new_records) >= 10:
            new_df = pd.DataFrame(new_records)
            features_df = pd.concat([features_df, new_df], ignore_index=True)
            features_df.to_csv(FEATURES_CSV, index=False)
            new_records = []
            
    if new_records:
        new_df = pd.DataFrame(new_records)
        features_df = pd.concat([features_df, new_df], ignore_index=True)
        features_df.to_csv(FEATURES_CSV, index=False)
        
    print(f"Analysis complete. Features saved to {FEATURES_CSV}")

if __name__ == "__main__":
    main()
