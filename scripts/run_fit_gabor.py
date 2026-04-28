import os
import sys
import glob
import pandas as pd
import numpy as np
import imageio.v2 as imageio
from scipy.optimize import least_squares
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))

if project_root not in sys.path:
    sys.path.append(project_root)

from medi_pipeline.config import paths

def gabor_function(p, x, y):
    amp, x0, y0, sig_x, sig_y, th, Lam, psi, off = p
    xt = (x - x0)*np.cos(th) + (y - y0)*np.sin(th)
    yt = -(x - x0)*np.sin(th) + (y - y0)*np.cos(th)
    gb = np.exp(-0.5*(xt**2/sig_x**2 + yt**2/sig_y**2)) * np.cos(2*np.pi/Lam * xt + psi)
    return amp * gb + off

def fit_gabor(img_path, image_size=(144, 256)):
    try:
        img = imageio.imread(img_path)
        if len(img.shape) == 3: img = np.mean(img, axis=-1)
        data = img.astype(np.float32)
        if data.max() == data.min():
            return 0.0
            
        data = (data - data.min()) / (data.max() - data.min() + 1e-8)
        
        H, W = image_size
        y_vec = np.linspace(-1, 1, H)
        x_vec = np.linspace(-1, 1, W)
        y, x = np.meshgrid(y_vec, x_vec, indexing='ij')
        
        y0_guess = y_vec[np.unravel_index(np.argmax(np.abs(data)), data.shape)[0]]
        x0_guess = x_vec[np.unravel_index(np.argmax(np.abs(data)), data.shape)[1]]
        
        p0 = [1.0, x0_guess, y0_guess, 0.15, 0.15, 0.0, 0.5, 0.0, float(np.mean(data))]
        
        def residual(p):
            return (gabor_function(p, x, y) - data).ravel()
            
        bounds = (
            [0, -1, -1, 0.05, 0.05, -np.pi, 0.1, -np.pi, -1],
            [5,  1,  1, 1.0,  1.0,  np.pi, 2.0,  np.pi,  1]
        )
        res = least_squares(residual, p0, bounds=bounds, max_nfev=150)
        
        sse = np.sum(res.fun**2)
        tss = np.sum((data - data.mean())**2)
        if tss == 0:
            return 0.0
        r2 = 1.0 - (sse / tss)
        
        return float(r2)
    except Exception as e:
        return np.nan

def parse_filename(fp):
    fname = os.path.basename(fp)
    name_no_ext = os.path.splitext(fname)[0]
    parts = name_no_ext.split('_')
    if len(parts) >= 3:
        session = int(parts[0])
        scan_idx = int(parts[1])
        readout_str = parts[2]
        if readout_str.startswith('r'):
            readout_id = int(readout_str[1:])
        else:
            readout_id = int(readout_str)
        return f"{session}_{scan_idx}_{readout_id}", session, scan_idx, readout_id
    return None, None, None, None
    
def process_file_wrapper(img_path):
    r2 = fit_gabor(img_path)
    id_str, session, scan_idx, readout_id = parse_filename(img_path)
    if id_str:
        return {
            'session': session,
            'scan_idx': scan_idx,
            'readout_id': readout_id,
            'r2': r2
        }
    return None

def main():
    search_dir = os.path.join(paths.RESULTS_DIR, 'MESI', 'V1')
    img_files = glob.glob(os.path.join(search_dir, '**/*.png'), recursive=True)
    print(f"Found {len(img_files)} MESI in {search_dir}")
    
    out_csv = os.path.join(paths.DATA_DIR, 'gabor_fit.csv')
    
    processed_ids = set()
    if os.path.exists(out_csv):
        df_existing = pd.read_csv(out_csv)
        if not df_existing.empty:
            for _, row in df_existing.iterrows():
                processed_ids.add(f"{int(row['session'])}_{int(row['scan_idx'])}_{int(row['readout_id'])}")
        print(f"Loaded {len(processed_ids)} previously processed results from {out_csv}.")
    else:
        pd.DataFrame(columns=['session', 'scan_idx', 'readout_id', 'r2']).to_csv(out_csv, index=False)
        
    filtered_files = []
    for fp in img_files:
        id_str, _, _, _ = parse_filename(fp)
        if id_str and id_str not in processed_ids:
            filtered_files.append(fp)
            
    print(f"Remaining MESI to process: {len(filtered_files)}")
    if len(filtered_files) == 0:
        print("All done!")
        return
        
    workers = min(cpu_count(), 14)
    results_batch = []
    
    with Pool(workers) as pool:
        pbar = tqdm(pool.imap_unordered(process_file_wrapper, filtered_files), total=len(filtered_files), desc="Fitting MESI")
        for idx, res in enumerate(pbar):
            if res:
                results_batch.append(res)
                
            if len(results_batch) % 10 == 0 or (idx + 1) == len(filtered_files):
                df_batch = pd.DataFrame(results_batch)
                df_batch.to_csv(out_csv, mode='a', header=False, index=False)
                results_batch = []
            
    print(f"Saved completed results to {out_csv}")
    
if __name__ == "__main__":
    main()
