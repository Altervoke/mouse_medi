import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image, ImageSequence

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))

if project_root not in sys.path:
    sys.path.append(project_root)

from medi_pipeline.config import paths

def set_style():
    sns.set_style("white", {'axes.spines.top': False, 'axes.spines.right': False})
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'font.size': 14,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'axes.linewidth': 1.0,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'legend.fontsize': 12,
        'legend.title_fontsize': 12,
        'legend.frameon': False, 
        'lines.linewidth': 1.5,
        'lines.markersize': 4,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',  
        'savefig.pad_inches': 0.1
    })

def plot_frames(fig, master_gs):
    color_stii = {
        'Low': '#FFB55A',
        'Medium': '#E85D04',
        'High': '#9D0208'
    }
    
    medi_df = pd.read_csv(os.path.join(paths.DATA_DIR, 'medi_features.csv'))

    manual_selections = [
        ('Low', 9, 3, 4641, os.path.join(paths.RESULTS_DIR, 'MEDI', 'V1', 'L4', '9_3_r4641.gif')),
        ('Medium', 7, 3, 8304, os.path.join(paths.RESULTS_DIR, 'MEDI', 'V1', 'L5', '7_3_r8304.gif')),
        ('High', 7, 3, 6785, os.path.join(paths.RESULTS_DIR, 'MEDI', 'RL', 'L5', '7_3_r6785.gif')),
    ]
    
    frames_to_plot = list(range(22, 38, 2))
    n_frames = len(frames_to_plot)
    
    outer_gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=master_gs, hspace=0)
    
    for row_idx, (level, sess, scan, readout_id, medi_path) in enumerate(manual_selections):
        match = medi_df[(medi_df['session'] == sess) & (medi_df['scan_idx'] == scan) & (medi_df['readout_id'] == readout_id)]
        stii_val = match.iloc[0]['STII'] if not match.empty else np.nan
        medi_frames = []
        if os.path.exists(medi_path):
            with Image.open(medi_path) as img:
                frs = [np.array(frame.copy().convert('L')) for frame in ImageSequence.Iterator(img)]
                if len(frs) >= max(frames_to_plot):
                    for f_idx in frames_to_plot:
                        medi_frames.append(frs[min(f_idx, len(frs)-1)])
        
        if not medi_frames:
            medi_frames = [np.zeros((144, 256))] * n_frames

        medi_strip = np.concatenate(medi_frames, axis=1)
        
        ax_medi = fig.add_subplot(outer_gs[row_idx])
        ax_medi.imshow(medi_strip, cmap='gray', vmin=0, vmax=255)
        ax_medi.set_xticks([])
        ax_medi.set_yticks([])
        ax_medi.set_zorder(10)
        
        ax_medi.text(-0.02, 0.5, f"STII={stii_val:.2f}", transform=ax_medi.transAxes, ha='right', va='center', rotation=0, color='black', fontsize=20, clip_on=False)
        
        if row_idx == 0:
            ax_medi.text(80, -20, 'Fr.', ha='right', va='bottom', fontsize=20, color='black', clip_on=False)
            for i, f_val in enumerate(frames_to_plot):
                ax_medi.text(i * 256 + 128, -20, f"{f_val+1}", ha='center', va='bottom', fontsize=20, color='black', clip_on=False)
        
        for spine in ax_medi.spines.values():
            spine.set_visible(False)
            

def generate_figs4():
    set_style()
    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 1)
    
    plot_frames(fig, gs[0, 0])
    
    out_path = os.path.join(paths.FIGURES_DIR, 'figs4.pdf')
    plt.savefig(out_path, dpi=300)
    print(f"Appendix Figure S4 generated at {out_path}")

if __name__ == "__main__":
    generate_figs4()
