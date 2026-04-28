import os
import math
import random
import numpy as np
import torch
import imageio.v2 as imageio
import torch.nn.functional as F
from tqdm import tqdm
from scipy.optimize import least_squares

VAE_SCALING_FACTOR = 1.0
LATENT_CHANNELS = 4
LATENT_H = 36
LATENT_W = 64

def apply_robust_transforms(img_batch, jitter_range=4, scale_range=0.05):
    T, C, H, W = img_batch.shape
    device = img_batch.device
    
    # 1. Random Scale (Zoom)
    scale = 1.0 + (torch.rand(1, device=device) * 2 - 1) * scale_range
    
    # 2. Random Translation (Jitter)
    dx = torch.randint(-jitter_range, jitter_range + 1, (1,), device=device).item()
    dy = torch.randint(-jitter_range, jitter_range + 1, (1,), device=device).item()
    theta = torch.eye(2, 3, device=device).unsqueeze(0).repeat(T, 1, 1) 
    
    theta[:, 0, 0] = 1.0 / scale
    theta[:, 1, 1] = 1.0 / scale
    theta[:, 0, 2] = -dx / (W / 2.0)
    theta[:, 1, 2] = -dy / (H / 2.0)
    
    grid = F.affine_grid(theta, img_batch.size(), align_corners=False)
    img_transformed = F.grid_sample(img_batch, grid, padding_mode='reflection', align_corners=False)
    
    # 3. Gaussian Blur (Spatial)
    k_size = 5
    sigma = 1.0
    x_coord = torch.arange(k_size, device=device).float() - k_size // 2
    gaussian_1d = torch.exp(-x_coord**2 / (2 * sigma**2))
    gaussian_1d = gaussian_1d / gaussian_1d.sum()
    gaussian_2d = gaussian_1d.unsqueeze(0) * gaussian_1d.unsqueeze(1)
    gaussian_2d = gaussian_2d.expand(C, 1, k_size, k_size)
    
    img_blurred = F.conv2d(img_transformed, gaussian_2d, padding=k_size//2, groups=C)
    
    return img_blurred

def generate_mesi(
    vae,
    model,
    readout_index,
    N_frames=20,
    iterations=10,
    lr=10.0,
    seed=42,
    device='cuda'
):
    vae.enable_gradient_checkpointing()
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False

    latent = torch.randn((1, LATENT_CHANNELS, 1, LATENT_H, LATENT_W), device=device) * 0.02 + 0.5
    latent.requires_grad_(True)
    
    optimizer = torch.optim.SGD([latent], lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations, eta_min=lr*0.1)
    
    pbar = tqdm(range(iterations), desc="Static Optimization")
    for it in pbar:
        optimizer.zero_grad()
        
        noise = torch.randn_like(latent) * 0.05
        noisy_latent = latent + noise
        
        scaled_latent = (1.0/VAE_SCALING_FACTOR) * noisy_latent.squeeze(0).permute(1,0,2,3)
        lat_sc = torch.clamp(scaled_latent, -10, 10)
        
        vae.float()
        decoded = torch.utils.checkpoint.checkpoint(vae.decode, lat_sc, use_reentrant=False).sample
        
        dec_sz = F.interpolate(decoded, (144, 256), mode='bilinear', align_corners=False)
        dec_gray = (0.299*dec_sz[:,0:1] + 0.587*dec_sz[:,1:2] + 0.114*dec_sz[:,2:3])
        dec_clamped = torch.clamp(dec_gray, -1, 1)
        dec_robust = apply_robust_transforms(dec_clamped, jitter_range=4) 
        model_in_single = torch.clamp((dec_robust+1.0)*0.5, 0.0, 1.0)
        
        model.reset()
        model_in = model_in_single.repeat(N_frames, 1, 1, 1)
        
        dummy_p = torch.zeros(N_frames, 2, device=device)
        dummy_m = torch.zeros(N_frames, 2, device=device)
        
        resps = []
        for t in range(N_frames):
            resps.append(model(model_in[t:t+1], dummy_p[t:t+1], dummy_m[t:t+1]))
        resp = torch.cat(resps, dim=0)
        
        target_resp = resp[:, readout_index].mean()
        
        safe_resp = torch.relu(target_resp) + 1e-4
        loss_log_resp = -torch.log(safe_resp) * 3.0
        
        d_lat_y = latent[:,:,:,1:,:] - latent[:,:,:,:-1,:]
        d_lat_x = latent[:,:,:,:,1:] - latent[:,:,:,:,:-1]
        loss_spatial = (torch.mean(d_lat_y**2) + torch.mean(d_lat_x**2)) * 500.0
        
        total_loss = loss_log_resp + loss_spatial
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_([latent], 0.1) 
        optimizer.step()
        scheduler.step()
        latent.data.clamp_(-5.0, 5.0)
    
    with torch.no_grad():
        final_latent = (1.0/VAE_SCALING_FACTOR) * latent.squeeze(0).permute(1,0,2,3)
        vae.float()
        pic = vae.decode(final_latent).sample
        
        pic_144 = F.interpolate(pic, size=(144, 256), mode='bilinear', align_corners=False)
        pic_gray = (0.299*pic_144[:,0:1] + 0.587*pic_144[:,1:2] + 0.114*pic_144[:,2:3])
        pic_clamped = torch.clamp(pic_gray, -1, 1)
        pic_final = torch.clamp((pic_clamped + 1.0) * 0.5, 0, 1)
        pic_numpy = pic_final.permute(0, 2, 3, 1).cpu().numpy()
        if pic_numpy.shape[-1] == 1:
            pic_numpy = pic_numpy.squeeze(-1)
        pic_uint8 = (pic_numpy * 255).astype(np.uint8)
        
    return pic_uint8[0]

def generate_pixel_medi(
    model,
    readout_index,
    total_frames=60,
    chunk_size=15,
    chunk_overlap=8,
    iterations=20,
    lr=10.0,
    seed=42,
    device='cuda'
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False

    step = chunk_size - chunk_overlap
    chunks = []
    start = 0
    while start < total_frames:
        end = start + chunk_size
        chunks.append((start, end))
        if end >= total_frames:
            break
        start += step
    
    prev_chunk_pixels = None 
    final_stitched_pixels_list = []
    pixel_init = torch.randn((1, 1, 1, 144, 256), device=device) * 0.02 + 0.5 
    pixel_init = torch.clamp(pixel_init, 0.0, 1.0)

    for i, (c_start, c_end) in enumerate(chunks):
        current_chunk_len = c_end - c_start
        model.reset()
        
        if i == 0:
            pixels_init = pixel_init.repeat(1, 1, current_chunk_len, 1, 1).clone().detach()
        else:
            overlap_len = min(chunk_overlap, prev_chunk_pixels.shape[2])
            overlap_section = prev_chunk_pixels[:, :, -overlap_len:, :, :].clone().detach()
            new_len = current_chunk_len - overlap_len
            if new_len > 0:
                new_section = pixel_init.repeat(1, 1, new_len, 1, 1).clone().detach()
                pixels_init = torch.cat([overlap_section, new_section], dim=2)
            else:
                pixels_init = overlap_section

        pixels = pixels_init.clone().detach().requires_grad_(True)
        
        optimizer = torch.optim.SGD([pixels], lr=lr, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations, eta_min=lr*0.1)
        
        pbar = tqdm(range(iterations), desc=f"Pixel Chunk {i}", leave=False)
        for it in pbar:
            optimizer.zero_grad()
            
            noisy_pixels = pixels + torch.randn_like(pixels) * 0.05
            pixel_input = noisy_pixels.squeeze(0).permute(1, 0, 2, 3)
            pixel_input = torch.clamp(pixel_input, 0.0, 1.0)
            
            dec_robust = apply_robust_transforms(pixel_input * 2.0 - 1.0, jitter_range=4)
            model_in = torch.clamp((dec_robust + 1.0) * 0.5, 0.0, 1.0)
            
            model.reset()
            N = model_in.shape[0]
            dummy_p = torch.zeros(N, 2, device=device)
            dummy_m = torch.zeros(N, 2, device=device)
            
            resps = []
            for t in range(N):
                resps.append(model(model_in[t:t+1], dummy_p[t:t+1], dummy_m[t:t+1]))
            resp = torch.cat(resps, dim=0)
            
            target_resp = resp[:, readout_index].mean()
            safe_resp = torch.relu(target_resp) + 1e-4
            loss_log_resp = -torch.log(safe_resp) * 3.0
            
            d_pixels = model_in[1:] - model_in[:-1]
            loss_temp_pixel = torch.mean(d_pixels ** 2) * 5000.0
            
            d_spat_y = pixels[:, :, :, 1:, :] - pixels[:, :, :, :-1, :]
            d_spat_x = pixels[:, :, :, :, 1:] - pixels[:, :, :, :, :-1]
            loss_spatial = (torch.mean(d_spat_y**2) + torch.mean(d_spat_x**2)) * 500.0
            
            total_loss = loss_log_resp + loss_temp_pixel + loss_spatial
            total_loss.backward()
            
            if chunk_overlap > 0 and i > 0:
                scales = torch.linspace(0.0, 1.0, steps=chunk_overlap, device=pixels.grad.device)
                scales = scales.view(1, 1, chunk_overlap, 1, 1)
                pixels.grad[:, :, 0:chunk_overlap, :, :] *= scales
                    
            pixels.grad[:, :, 0:1, :, :] = 0.0
            torch.nn.utils.clip_grad_norm_([pixels], 0.1) 
            optimizer.step()
            scheduler.step()
            pixels.data.clamp_(0.0, 1.0)
        
        prev_chunk_pixels = pixels.detach().clone()
        len_to_keep = current_chunk_len
        if i < len(chunks) - 1:
            len_to_keep -= chunk_overlap
            
        chunk_pixels_final = pixels[:, :, :len_to_keep, :, :].detach().cpu()
        final_stitched_pixels_list.append(chunk_pixels_final)

    final_pixels = torch.cat(final_stitched_pixels_list, dim=2)
    final_pixels = final_pixels.squeeze(0).squeeze(0)
    
    vid_numpy = final_pixels.numpy()
    vid_uint8 = (vid_numpy * 255).astype(np.uint8)
        
    return vid_uint8[:total_frames]

def generate_medi(
    vae,
    model,
    readout_index,
    total_frames=60,
    chunk_size=15,
    chunk_overlap=8,
    iterations=20,
    lr=10.0,
    seed=42,
    device='cuda'
):
    vae.enable_gradient_checkpointing()
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False

    step = chunk_size - chunk_overlap
    chunks = []
    start = 0
    while start < total_frames:
        end = start + chunk_size
        chunks.append((start, end))
        if end >= total_frames:
            break
        start += step
    
    prev_chunk_latents = None 
    final_stitched_latents_list = []
    latent_init = torch.randn((1, LATENT_CHANNELS, 1, LATENT_H, LATENT_W), device=device) * 0.02 + 0.5

    for i, (c_start, c_end) in enumerate(chunks):
        current_chunk_len = c_end - c_start
        model.reset()
        
        if i == 0:
            latents_init = latent_init.repeat(1, 1, current_chunk_len, 1, 1).clone().detach()

        else:
            overlap_len = min(chunk_overlap, prev_chunk_latents.shape[2])
            overlap_section = prev_chunk_latents[:, :, -overlap_len:, :, :].clone().detach()
            
            new_len = current_chunk_len - overlap_len
            if new_len > 0:
                new_section = latent_init.repeat(1, 1, new_len, 1, 1).clone().detach()
                latents_init = torch.cat([overlap_section, new_section], dim=2)
            else:
                latents_init = overlap_section

        latents = latents_init.clone().detach().requires_grad_(True)
        
        optimizer = torch.optim.SGD([latents], lr=lr, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations, eta_min=lr*0.1)
        
        pbar = tqdm(range(iterations), desc=f"Chunk {i}", leave=False)
        for it in pbar:
            optimizer.zero_grad()
            
            # 1. Latent Robustness
            noise = torch.randn_like(latents) * 0.05
            noisy_latents = latents + noise
            
            # 2. Decode
            lat_input = noisy_latents.squeeze(0).permute(1,0,2,3)
            scaled_latents = (1.0/VAE_SCALING_FACTOR) * lat_input
            lat_sc = torch.clamp(scaled_latents, -10, 10)
            dec_list = []
            vae.float()
            for chi in range(0, lat_sc.shape[0]):
                ch = lat_sc[chi:chi+1].float()
                d = torch.utils.checkpoint.checkpoint(vae.decode, ch, use_reentrant=False).sample
                dec_list.append(d)
            decoded = torch.cat(dec_list, dim=0)
            
            # 3. Preprocess
            dec_sz = F.interpolate(decoded, (144, 256), mode='bilinear', align_corners=False)
            dec_gray = (0.299*dec_sz[:,0:1] + 0.587*dec_sz[:,1:2] + 0.114*dec_sz[:,2:3])
            dec_clamped = torch.clamp(dec_gray, -1, 1)
            dec_robust = apply_robust_transforms(dec_clamped, jitter_range=4) 
            model_in = torch.clamp((dec_robust+1.0)*0.5, 0.0, 1.0)
            
            # 4. Forward Pass
            model.reset()
            
            N = model_in.shape[0]
            dummy_p = torch.zeros(N, 2, device=device)
            dummy_m = torch.zeros(N, 2, device=device)
            
            resps = []
            for t in range(N):
                resps.append(model(model_in[t:t+1], dummy_p[t:t+1], dummy_m[t:t+1]))
            resp = torch.cat(resps, dim=0)
            
            # 5. Losses
            target_resp = resp[:, readout_index].mean()
            
            safe_resp = torch.relu(target_resp) + 1e-4
            loss_log_resp = -torch.log(safe_resp) * 3.0
            
            d_latents = latents[:, :, 1:] - latents[:, :, :-1]
            loss_temp_latent = torch.mean(d_latents ** 2) * 2000.0
            
            d_pixels = model_in[1:] - model_in[:-1]
            loss_temp_pixel = torch.mean(d_pixels ** 2) * 5000.0
            
            d_lat_y = latents[:,:,:,1:,:] - latents[:,:,:,:-1,:]
            d_lat_x = latents[:,:,:,:,1:] - latents[:,:,:,:,:-1]
            loss_spatial = (torch.mean(d_lat_y**2) + torch.mean(d_lat_x**2)) * 500.0
            
            total_loss = loss_log_resp + loss_temp_latent + loss_temp_pixel + loss_spatial
            total_loss.backward()
            
            # 6. Gradient Masking & Clipping
            if chunk_overlap > 0 and i > 0:
                scales = torch.linspace(0.0, 1.0, steps=chunk_overlap, device=latents.grad.device)
                scales = scales.view(1, 1, chunk_overlap, 1, 1)
                latents.grad[:, :, 0:chunk_overlap, :, :] *= scales
                    
            latents.grad[:,:,0:1,:,:] = 0.0
            torch.nn.utils.clip_grad_norm_([latents], 0.1) 
            optimizer.step()
            scheduler.step()
            latents.data.clamp_(-5.0, 5.0)
        
        prev_chunk_latents = latents.detach().clone()
        len_to_keep = current_chunk_len
        if i < len(chunks) - 1:
            len_to_keep -= chunk_overlap
            
        chunk_latents_final = latents[:, :, :len_to_keep, :, :].detach().cpu()
        final_stitched_latents_list.append(chunk_latents_final)

    final_latents_tensor = torch.cat(final_stitched_latents_list, dim=2)
    final_latents_tensor = final_latents_tensor.squeeze(0).permute(1, 0, 2, 3)
    
    with torch.no_grad():
        final_latents_tensor = final_latents_tensor.to(device)
        d_list = []
        vae.float()
        for chi in range(0, final_latents_tensor.shape[0]):
            d_list.append(vae.decode(final_latents_tensor[chi:chi+1] / VAE_SCALING_FACTOR).sample)
        
        vid = torch.cat(d_list, dim=0)
        
        vid_144 = F.interpolate(vid, size=(144, 256), mode='bilinear', align_corners=False)
        vid_gray = (0.299*vid_144[:,0:1] + 0.587*vid_144[:,1:2] + 0.114*vid_144[:,2:3])
        vid_clamped = torch.clamp(vid_gray, -1, 1)
        vid_final = torch.clamp((vid_clamped + 1.0) * 0.5, 0, 1)
        vid_numpy = vid_final.permute(0, 2, 3, 1).cpu().numpy()
        if vid_numpy.shape[-1] == 1:
            vid_numpy = vid_numpy.squeeze(-1)
        vid_uint8 = (vid_numpy * 255).astype(np.uint8)
        
    return vid_uint8[:total_frames]

def get_mesi_prior(mesi_path, image_size=(144, 256)):
    try:
        img = imageio.imread(mesi_path)
        if len(img.shape) == 3: img = np.mean(img, axis=-1)
        data = img.astype(np.float32)
        data = (data - data.min()) / (data.max() - data.min() + 1e-8)
        
        H, W = image_size
        y_vec = np.linspace(-1, 1, H)
        x_vec = np.linspace(-1, 1, W)
        y, x = np.meshgrid(y_vec, x_vec, indexing='ij')
        
        y0_guess = y_vec[np.unravel_index(np.argmax(np.abs(data)), data.shape)[0]]
        x0_guess = x_vec[np.unravel_index(np.argmax(np.abs(data)), data.shape)[1]]
        
        p0 = [1.0, x0_guess, y0_guess, 0.15, 0.15, 0.0, 0.5, 0.0, float(np.mean(data))]
        
        def residual(p):
            amp, x0, y0, sig_x, sig_y, th, Lam, psi, off = p
            xt = (x - x0)*np.cos(th) + (y - y0)*np.sin(th)
            yt = -(x - x0)*np.sin(th) + (y - y0)*np.cos(th)
            gb = np.exp(-0.5*(xt**2/sig_x**2 + yt**2/sig_y**2)) * np.cos(2*np.pi/Lam * xt + psi)
            return (amp * gb + off - data).ravel()
            
        bounds = (
            [0, -1, -1, 0.05, 0.05, -np.pi, 0.1, -np.pi, -1],
            [5,  1,  1, 1.0,  1.0,  np.pi, 2.0,  np.pi,  1]
        )
        res = least_squares(residual, p0, bounds=bounds, max_nfev=150)
        
        amp, x0, y0, sig_x, sig_y, th, Lam, psi, off = res.x
        return {
            'theta': float(th),
            'sigma': float(sig_x),
            'Lambda': float(Lam),
            'psi': float(psi),
            'gamma': float(sig_x / sig_y) if sig_y > 1e-5 else 1.0,
            'center': [float(x0), float(y0)]
        }
    except Exception as e:
        print("Failed to fit MESI prior:", e)
        return None

def gen_gabor(theta, sigma, Lambda, psi, gamma, center, image_size, device='cuda'):
    sigma_x = sigma
    sigma_y = sigma / gamma
    ny, nx = image_size
    
    y = torch.linspace(-1, 1, ny, device=device)
    x = torch.linspace(-1, 1, nx, device=device)
    (y, x) = torch.meshgrid(y, x, indexing='ij')

    x_theta = (x - center[0]) * torch.cos(theta) + (y - center[1]) * torch.sin(theta)
    y_theta = -(x - center[0]) * torch.sin(theta) + (y - center[1]) * torch.cos(theta)

    gb = torch.exp(-0.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * torch.cos(2 * math.pi / Lambda * x_theta + psi)
    return gb

class GaborGenerator(torch.nn.Module):
    def __init__(self, image_size, device='cuda', init_params=None):
        super().__init__()
        
        t_theta = torch.rand(1) * 4 * math.pi - 2 * math.pi
        t_sigma = torch.rand(1) * 0.05 + 0.15
        t_Lambda = torch.rand(1) * 0.2 + 0.5
        t_psi = torch.rand(1) * math.pi / 2
        t_gamma = torch.ones(1)
        t_center = torch.zeros(2)
        
        if init_params is not None:
            t_theta = torch.tensor([init_params['theta']])
            t_sigma = torch.tensor([init_params['sigma']])
            t_Lambda = torch.tensor([init_params['Lambda']])
            t_psi = torch.tensor([init_params['psi']])
            t_gamma = torch.tensor([init_params['gamma']])
            t_center = torch.tensor(init_params['center'])

        self.theta = torch.nn.Parameter(t_theta.to(device))
        self.sigma = torch.nn.Parameter(t_sigma.to(device), requires_grad=False)
        self.Lambda = torch.nn.Parameter(t_Lambda.to(device))
        self.psi = torch.nn.Parameter(t_psi.to(device))
        self.gamma = torch.nn.Parameter(t_gamma.to(device), requires_grad=False)
        self.center = torch.nn.Parameter(t_center.to(device))
        self.image_size = image_size
        self.device = device
    
    def forward(self):
        self.theta.data.clamp_(-2 * math.pi, 2 * math.pi)
        self.sigma.data.clamp_(0.13, 1.0)
        self.Lambda.data.clamp_(0.2, 2.0)
        self.center.data.clamp_(-0.8, 0.8)
        
        gb = gen_gabor(self.theta, self.sigma, self.Lambda, self.psi, self.gamma, self.center, self.image_size, device=self.device)
        return gb.view(1, 1, *self.image_size)
    
    def apply_changes(self):
        self.sigma.requires_grad_(True)

def generate_gabor(
    model,
    readout_index,
    image_size=(144, 256),
    N_frames=20,
    iterations=100,
    lr=0.1,
    fixed_std=0.1,
    seed=42,
    device='cuda',
    mesi_prior_path=None
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False

    init_params = None
    if mesi_prior_path and os.path.exists(mesi_prior_path):
        init_params = get_mesi_prior(mesi_prior_path, image_size)
        if init_params:
            print(f"   [+] Initializing Gabor with MESI prior: center={init_params['center']}, lambda={init_params['Lambda']:.3f}, theta={init_params['theta']:.3f}")

    gabor_generator = GaborGenerator(image_size, device=device, init_params=init_params)
    gabor_generator.apply_changes()
    
    optimizer = torch.optim.Adam(gabor_generator.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations, eta_min=lr*0.1)

    saved_frames = []

    pbar = tqdm(range(iterations), desc="Gabor Optimization")
    for iter in pbar:

        optimizer.zero_grad()
        gabor = gabor_generator()
        
        if fixed_std is not None:
            gabor_std = gabor.std()
            gabor_constrained = fixed_std * gabor / (gabor_std + 1e-8)
        else:
            gabor_constrained = gabor
        
        gabor_clamped = torch.clamp(gabor_constrained, -1.0, 1.0)
        model_in_single = (gabor_clamped + 1.0) * 0.5
        
        model_in = model_in_single.repeat(N_frames, 1, 1, 1)
        
        model.reset()
        dummy_p = torch.zeros(N_frames, 2, device=device)
        dummy_m = torch.zeros(N_frames, 2, device=device)
        
        resps = []
        for t in range(N_frames):
            resps.append(model(model_in[t:t+1], dummy_p[t:t+1], dummy_m[t:t+1]))
        resp = torch.cat(resps, dim=0)
        
        target_resp = resp[:, readout_index].mean()
        
        loss = -target_resp
        loss.backward()
        optimizer.step()
        scheduler.step(-loss.item())
        
        frame_numpy = model_in_single.detach().squeeze().cpu().numpy()
        frame_uint8 = (frame_numpy * 255).astype(np.uint8)
        saved_frames.append(frame_uint8)

    gabor_generator.eval()
    with torch.no_grad():
        gabor = gabor_generator()
        if fixed_std is not None:
            gabor_std = gabor.std()
            gabor_constrained = fixed_std * gabor / (gabor_std + 1e-8)
        else:
            gabor_constrained = gabor
            
        gabor_clamped = torch.clamp(gabor_constrained, -1.0, 1.0)
        model_in_single = (gabor_clamped + 1.0) * 0.5
        pic_numpy = model_in_single.squeeze().cpu().numpy()
        pic_uint8 = (pic_numpy * 255).astype(np.uint8)
            
    return pic_uint8
