import os
import sys
import numpy as np
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
    color_method = {'MEDI': '#1f77b4', 'Pixel': "#ddd42a"}
    
    examples = [
        ('V1', 'L4', 9, 3, 6900, os.path.join(paths.RESULTS_DIR, 'MEDI', 'V1', 'L4', '9_3_r6900.gif'), os.path.join(paths.RESULTS_DIR, 'MEDI_pixel', 'V1', 'L4', '9_3_r6900.gif')),
        ('LM', 'L5', 5, 6, 6184, os.path.join(paths.RESULTS_DIR, 'MEDI', 'LM', 'L5', '5_6_r6184.gif'), os.path.join(paths.RESULTS_DIR, 'MEDI_pixel', 'LM', 'L5', '5_6_r6184.gif')),
        ('RL', 'L5', 6, 2, 6139, os.path.join(paths.RESULTS_DIR, 'MEDI', 'RL', 'L5', '6_2_r6139.gif'), os.path.join(paths.RESULTS_DIR, 'MEDI_pixel', 'RL', 'L5', '6_2_r6139.gif')),
        ('AL', 'L23', 6, 7, 4143, os.path.join(paths.RESULTS_DIR, 'MEDI', 'AL', 'L23', '6_7_r4143.gif'), os.path.join(paths.RESULTS_DIR, 'MEDI_pixel', 'AL', 'L23', '6_7_r4143.gif')),
    ]
    frames_to_plot = list(range(22, 38, 2))
    n_frames = len(frames_to_plot)
    
    outer_gs = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=master_gs, hspace=0.1)
    
    for row_idx, (area, layer, sess, scan, readout_id, medi_path, pxl_path) in enumerate(examples):
        medi_frames = []
        if os.path.exists(medi_path):
            with Image.open(medi_path) as img:
                frs = [np.array(frame.copy().convert('L')) for frame in ImageSequence.Iterator(img)]
                if len(frs) >= max(frames_to_plot):
                    for f_idx in frames_to_plot:
                        medi_frames.append(frs[min(f_idx, len(frs)-1)])
        if not medi_frames:
            medi_frames = [np.zeros((144, 256))] * n_frames

        pxl_frames = []
        if os.path.exists(pxl_path):
            with Image.open(pxl_path) as img:
                frs = [np.array(frame.copy().convert('L')) for frame in ImageSequence.Iterator(img)]
                if len(frs) >= max(frames_to_plot):
                    for f_idx in frames_to_plot:
                        pxl_frames.append(frs[min(f_idx, len(frs)-1)])
        if not pxl_frames:
            pxl_frames = [np.zeros((144, 256))] * n_frames

        medi_strip = np.concatenate(medi_frames, axis=1)
        pxl_strip = np.concatenate(pxl_frames, axis=1)
        
        inner_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[row_idx], hspace=0.05)
        
        ax_medi = fig.add_subplot(inner_gs[0])
        ax_medi.imshow(medi_strip, cmap='gray', vmin=0, vmax=255)
        ax_medi.set_xticks([])
        ax_medi.set_yticks([])
        ax_medi.set_zorder(10)
        
        ax_medi.text(-0.02, 0.0, area, transform=ax_medi.transAxes, ha='right', va='center', rotation=0, color='black', fontsize=20, clip_on=False)
        
        if row_idx == 0:
            ax_medi.text(80, -20, 'Fr.', ha='right', va='bottom', fontsize=20, color='black', clip_on=False)
            for i, f_val in enumerate(frames_to_plot):
                ax_medi.text(i * 256 + 128, -20, f"{f_val+1}", ha='center', va='bottom', fontsize=20, color='black', clip_on=False)
        
        for spine in ax_medi.spines.values():
            spine.set_visible(True)
            spine.set_color(color_method['MEDI'])
            spine.set_linewidth(3)
            
        ax_pxl = fig.add_subplot(inner_gs[1])
        ax_pxl.imshow(pxl_strip, cmap='gray', vmin=0, vmax=255)
        ax_pxl.set_xticks([])
        ax_pxl.set_yticks([])
        ax_pxl.set_zorder(1)
        
        for spine in ax_pxl.spines.values():
            spine.set_visible(True)
            spine.set_color(color_method['Pixel'])
            spine.set_linewidth(3)
            

def generate_figs1():
    set_style()
    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(1, 1)
    
    plot_frames(fig, gs[0, 0])
    fig.canvas.draw()
    
    axes = fig.axes
    for i in range(0, len(axes), 2):
        ax_medi = axes[i]
        ax_pxl = axes[i+1]
        
        pos_medi = ax_medi.get_position()
        pos_pxl = ax_pxl.get_position()
        
        height = pos_medi.height
        
        pos_pxl.y1 = pos_medi.y0 - 7 / (9.0 * 72.0)
        pos_pxl.y0 = pos_pxl.y1 - height
        
        ax_pxl.set_position(pos_pxl)
        
    plt.subplots_adjust(top=0.98, bottom=0.12, left=0.06, right=0.98)
        
    legend_ax = fig.add_axes([0.35, 0.06, 0.3, 0.05])
    legend_ax.axis('off')
    
    colors = ['#1f77b4', '#ddd42a']
    labels = ['MEDI', 'Pixel']
    n = len(labels)
    x_start = 0.2
    x_end = 0.8
    xs = np.linspace(x_start, x_end, n)
    
    for x, col, lab in zip(xs, colors, labels):
        outer_rect = plt.Rectangle(
            (x - 0.16, 0.28), 0.3, 0.5,
            facecolor='none', edgecolor=col, linewidth=3,
            transform=legend_ax.transAxes
        )
        legend_ax.add_patch(outer_rect)
        
        inner_rect = plt.Rectangle(
            (x - 0.155, 0.28), 0.29, 0.46,
            facecolor=col, alpha=0.3, edgecolor='none',
            transform=legend_ax.transAxes
        )
        legend_ax.add_patch(inner_rect)
        
        legend_ax.text(x + 0.17, 0.5, lab, ha='left', va='center', fontsize=18, transform=legend_ax.transAxes)
        
    out_path = os.path.join(paths.FIGURES_DIR, 'figs1.pdf')
    plt.savefig(out_path, dpi=300)
    print(f"Appendix Figure S1 generated at {out_path}")

if __name__ == "__main__":
    generate_figs1()
