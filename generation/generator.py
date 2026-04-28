import os
import sys
import imageio
from diffusers import AutoencoderTiny

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
fnn_root = os.path.join(project_root, 'fnn')

if project_root not in sys.path:
    sys.path.append(project_root)
if fnn_root not in sys.path:
    sys.path.append(fnn_root)

from medi_pipeline.config import paths
from medi_pipeline.generation.optimizer import generate_medi, generate_mesi, generate_pixel_medi, generate_gabor
from fnn import microns

SRC_DIR = getattr(paths, 'SRC_DIR')
TAESD_PATH = getattr(paths, 'TAESD_PATH')

def load_foundation_model(session, scan_idx, device='cuda'):
    print(f"Loading Foundation Model (Session {session}, Scan {scan_idx})...")
    model, neuron_ids = microns.scan(
        session=session, 
        scan_idx=scan_idx, 
        directory=SRC_DIR
    )
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model

def load_vae(device='cuda'):
    print(f"Loading VAE from {TAESD_PATH}...")
    vae = AutoencoderTiny.from_pretrained(
        TAESD_PATH, 
        local_files_only=True
    )
    vae.to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    return vae

def generate_mesi_func(session, scan, readout_id, brain_area, layer, model, vae, device='cuda'):
    base_results = os.path.join(getattr(paths, 'RESULTS_DIR'), 'MESI')
    filename = f"{session}_{scan}_r{readout_id}.png"
    
    out_dir = os.path.join(base_results, brain_area, layer)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    
    if os.path.exists(out_path):
        print(f"MESI already exists at {out_path}, skipping generation.")
        return out_path
    
    print(f"Generating MESI for {brain_area}/{layer} - Session {session} Scan {scan} Readout {readout_id}...")
    final_img = generate_mesi(
        vae=vae,
        model=model,
        readout_index=readout_id,
        N_frames=20,
        iterations=10,
        lr=10.0,
        seed=42,
        device=device
    )
    imageio.imwrite(out_path, final_img)
    print(f"Saved MESI to {out_path}")
    
    return out_path

def generate_medi_func(session, scan, readout_id, brain_area, layer, model, vae, device='cuda'):
    base_results = getattr(paths, 'RESULTS_MEDI_DIR')
    filename = f"{session}_{scan}_r{readout_id}.gif"
    
    out_dir = os.path.join(base_results, brain_area, layer)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    
    if os.path.exists(out_path):
        print(f"MEDI already exists at {out_path}, skipping generation.")
        return out_path
    
    print(f"Generating MEDI for {brain_area}/{layer} - Session {session} Scan {scan} Readout {readout_id}...")
    final_vid = generate_medi(
        vae=vae,
        model=model,
        readout_index=readout_id,
        total_frames=60,
        chunk_size=15,
        chunk_overlap=8,
        iterations=20,
        lr=10.0,
        seed=42,
        device=device
    )
    imageio.mimsave(out_path, final_vid, fps=30, loop=0)
    print(f"Saved MEDI to {out_path}")
    
    return out_path

def generate_pixel_medi_func(session, scan, readout_id, brain_area, layer, model, device='cuda'):
    base_results = os.path.join(getattr(paths, 'RESULTS_DIR'), 'MEDI_pixel')
    filename = f"{session}_{scan}_r{readout_id}.gif"
    
    out_dir = os.path.join(base_results, brain_area, layer)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    
    if os.path.exists(out_path):
        print(f"Pixel MEDI already exists at {out_path}, skipping generation.")
        return out_path
    
    print(f"Generating PIXEL MEDI for {session} {scan} r{readout_id}...")
    final_vid = generate_pixel_medi(
        model=model,
        readout_index=readout_id,
        total_frames=60,
        chunk_size=15,
        chunk_overlap=8,
        iterations=20,
        lr=10.0,
        seed=42,
        device=device
    )
    imageio.mimsave(out_path, final_vid, fps=30, loop=0)
    print(f"Saved Pixel MEDI to {out_path}")
    
    return out_path

def generate_gabor_func(session, scan, readout_id, brain_area, layer, model, device='cuda'):
    import imageio.v2 as imageio
    base_results = os.path.join(getattr(paths, 'RESULTS_DIR'), 'Gabor')
    filename = f"{session}_{scan}_r{readout_id}.png"
    
    out_dir = os.path.join(base_results, brain_area, layer)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    
    if os.path.exists(out_path):
        print(f"Gabor already exists at {out_path}, skipping generation.")
        return out_path
        
    base_mesi = os.path.join(getattr(paths, 'RESULTS_DIR'), 'MESI')
    mesi_path = os.path.join(base_mesi, brain_area, layer, f"{session}_{scan}_r{readout_id}.png")
    
    print(f"Generating Gabor for {session} {scan} r{readout_id}...")
    final_img = generate_gabor(
        model=model,
        readout_index=readout_id,
        image_size=(144, 256),
        N_frames=20,
        iterations=100,
        lr=0.1,
        fixed_std=0.1,
        seed=42,
        device=device,
        mesi_prior_path=mesi_path
    )
    imageio.imwrite(out_path, final_img)
    print(f"Saved Gabor to {out_path}")
    
    return out_path
