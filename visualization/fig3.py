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
        'font.size': 9,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'axes.linewidth': 0.8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'legend.fontsize': 8,
        'legend.title_fontsize': 9,
        'legend.frameon': False,
        'lines.linewidth': 1.5,
        'lines.markersize': 4,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.01
    })

def plot_frames(fig, outer_gs, units_df, fnn_resp, fnn_stim):
    color_method = {'MEDI': '#1f77b4', 'Natural': '#ff7f0e', 'Grating': '#2ca02c'}

    examples = [
        ('V1', 'L4', 9, 3, 6900, os.path.join(paths.RESULTS_DIR, 'MEDI', 'V1', 'L4', '9_3_r6900.gif')),
        ('LM', 'L5', 5, 6, 6184, os.path.join(paths.RESULTS_DIR, 'MEDI', 'LM', 'L5', '5_6_r6184.gif')),
        ('RL', 'L5', 6, 2, 6139, os.path.join(paths.RESULTS_DIR, 'MEDI', 'RL', 'L5', '6_2_r6139.gif')),
        ('AL', 'L23', 6, 7, 4143, os.path.join(paths.RESULTS_DIR, 'MEDI', 'AL', 'L23', '6_7_r4143.gif')),
    ]
    frames_to_plot = list(range(22, 38, 2))
    n_frames = len(frames_to_plot)

    grating_df = pd.read_csv(os.path.join(paths.DATA_DIR, 'grating_features.csv'))
    baseline_df = pd.read_csv(os.path.join(paths.DATA_DIR, 'baseline_comparisons.csv'))
    nat_meta = pd.read_csv(os.path.join(paths.DATA_DIR, 'natural_max_features.csv'))

    stimulus = None

    outer_gs_inner = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=outer_gs, hspace=0.1)

    for row_idx, (area, layer, sess, scan, readout_id, gif_path) in enumerate(examples):
        baseline_match = baseline_df[(baseline_df['session'] == sess) & (baseline_df['scan_idx'] == scan) & (baseline_df['readout_id'] == readout_id)]

        if baseline_match.empty:
            medi_act, nat_act, grat_act = 0.0, 0.0, 0.0
        else:
            medi_act = baseline_match.iloc[0]['MEDI']
            nat_act = baseline_match.iloc[0]['Natural']
            grat_act = baseline_match.iloc[0]['Grating']

        medi_frames = []
        if os.path.exists(gif_path):
            with Image.open(gif_path) as img:
                frs = [np.array(frame.copy().convert('L')) for frame in ImageSequence.Iterator(img)]
                if len(frs) >= max(frames_to_plot):
                    for f_idx in frames_to_plot:
                        medi_frames.append(frs[min(f_idx, len(frs)-1)])
        if not medi_frames:
            medi_frames = [np.zeros((144, 256))] * n_frames

        if stimulus is None:
            stimulus = np.load(fnn_stim, mmap_mode='r')

        nat_frames = []
        n_match = nat_meta[(nat_meta['session'] == sess) & (nat_meta['scan_idx'] == scan) & (nat_meta['readout_id'] == readout_id)]
        if not n_match.empty:
            abs_start = int(n_match.iloc[0]['start_frame'])
            for f_idx in frames_to_plot:
                a_frm = min(abs_start + f_idx, stimulus.shape[0] - 1)
                nat_frames.append(stimulus[a_frm])
        else:
            nat_frames = [np.zeros((144, 256))] * n_frames

        g_match = grating_df[(grating_df['session'] == sess) & (grating_df['scan_idx'] == scan) & (grating_df['readout_id'] == readout_id)]
        grat_img = np.zeros((144, 256))
        pref_dir = 0.0
        if not g_match.empty:
            pref_dir = g_match.iloc[0]['pref_dir']
            pref_sf = g_match.iloc[0]['pref_sf']
            y_coords, x_coords = np.meshgrid(np.arange(144), np.arange(256), indexing='ij')
            theta_rad = -np.deg2rad(pref_dir)
            kx = pref_sf * np.cos(theta_rad)
            ky = pref_sf * np.sin(theta_rad)
            grat_img = 0.5 + 0.5 * np.cos(2 * np.pi * (kx * x_coords + ky * y_coords))

        medi_strip = np.concatenate(medi_frames, axis=1)
        natural_strip = np.concatenate(nat_frames, axis=1)

        inner_gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer_gs_inner[row_idx],
                                                    width_ratios=[8.2, 1.5], hspace=0.08, wspace=0.05)

        ax_medi = fig.add_subplot(inner_gs[0, 0])
        ax_medi.imshow(medi_strip, cmap='gray', vmin=0, vmax=255)
        ax_medi.set_xticks([])
        ax_medi.set_yticks([])
        ax_medi.set_zorder(10)
        ax_medi.text(-0.02, 0.0, area, transform=ax_medi.transAxes, ha='right', va='center',
                     rotation=0, color='black', fontsize=14, clip_on=False)

        if row_idx == 0:
            ax_medi.text(80, -12, 'Fr.', ha='right', va='bottom', fontsize=14, color='black', clip_on=False)
            for i, f_val in enumerate(frames_to_plot):
                ax_medi.text(i * 256 + 128, -12, f"{f_val+1}", ha='center', va='bottom',
                             fontsize=14, color='black', clip_on=False)

        for spine in ax_medi.spines.values():
            spine.set_visible(True)
            spine.set_color(color_method['MEDI'])
            spine.set_linewidth(2)
        ax_medi.text(1, 0, f"{medi_act:.2f}", transform=ax_medi.transAxes,
                     color='black', ha='right', va='bottom', fontsize=14,
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', linewidth=0.0, pad=1.5), zorder=100)

        ax_grat = fig.add_subplot(inner_gs[:, 1])
        ax_grat.imshow(grat_img, cmap='gray', vmin=0, vmax=1)
        ax_grat.set_xticks([])
        ax_grat.set_yticks([])

        for spine in ax_grat.spines.values():
            spine.set_visible(True)
            spine.set_color(color_method['Grating'])
            spine.set_linewidth(2)
        ax_grat.text(1, 0, f"{grat_act:.2f}", transform=ax_grat.transAxes,
                     color='black', ha='right', va='bottom', fontsize=14,
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', linewidth=0.0, pad=1.5), zorder=100)

        theta_rad = np.deg2rad(pref_dir)
        dx = 0.4 * np.cos(theta_rad)
        dy = 0.4 * np.sin(theta_rad)
        ax_grat.annotate('', xy=(0.5 + dx, 0.5 + dy), xytext=(0.5 - dx, 0.5 - dy),
                         xycoords='axes fraction', textcoords='axes fraction',
                         arrowprops=dict(arrowstyle="-|>", color='red', lw=2.5, mutation_scale=20),
                         ha='center', va='center')

        ax_nat = fig.add_subplot(inner_gs[1, 0])
        ax_nat.imshow(natural_strip, cmap='gray', vmin=0, vmax=255)
        ax_nat.set_xticks([])
        ax_nat.set_yticks([])
        ax_nat.set_zorder(1)

        for spine in ax_nat.spines.values():
            spine.set_visible(True)
            spine.set_color(color_method['Natural'])
            spine.set_linewidth(2)
        ax_nat.text(1, 0, f"{nat_act:.2f}", transform=ax_nat.transAxes,
                    color='black', ha='right', va='bottom', fontsize=14,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', linewidth=0.0, pad=1.5), zorder=100)

def plot_baseline_density_all(ax):
    csv_path = os.path.join(paths.DATA_DIR, 'baseline_comparisons.csv')
    df = pd.read_csv(csv_path)

    baselines = ['Gabor', 'MESI', 'Grating', 'Natural']
    display_names = {'Gabor': 'Gabor', 'MESI': 'MESI', 'Grating': 'Grating', 'Natural': 'Natural'}
    colors = {'Gabor': "#ea4444", 'MESI': "#7552b4", 'Grating': '#2ca02c', 'Natural': '#ff7f0e'}

    all_ratios = []
    for b in baselines:
        ratio = df['MEDI'] / (df[b].replace(0, np.nan) + 1e-9)
        ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
        if len(ratio) == 0:
            continue
        med = ratio.median()
        label_text = f"MEDI / {display_names[b]} (Median: {med:.1f})"
        sns.kdeplot(ratio, fill=True, log_scale=False, color=colors[b], label=label_text,
                    ax=ax, alpha=0.4, linewidth=2, gridsize=1000, bw_adjust=1.8)
        ax.axvline(med, color=colors[b], linestyle='--', linewidth=1.5, zorder=2)
        all_ratios.extend(ratio.values)
    ax.axvline(1.0, color='k', linestyle='-', linewidth=1.5, zorder=1)

    if len(all_ratios) > 0:
        upper_bound = np.percentile(all_ratios, 95)
        xmax = max(6.0, min(20.0, upper_bound * 1.3))
        ax.set_xlim(0, xmax)

    ax.legend(loc='upper right', fontsize=15, frameon=False)
    sns.despine(ax=ax, left=True)
    ax.set_yticks([])
    ax.set_xticks([0, 1, 4, 8, 12, 16, 20])
    ax.set_xlim(0, 20)
    ax.set_title('')
    ax.set_ylabel('')
    ax.set_xlabel('Response Gain Ratio (MEDI / Baseline)', fontsize=18, labelpad=10)
    ax.tick_params(axis='x', labelsize=14)
    import matplotlib.ticker as ticker
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x)}"))
    ax.tick_params(axis='both', which='major', direction='out', length=3, width=1, bottom=True, left=True)

def generate_fig3():
    set_style()
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'legend.title_fontsize': 12,
    })

    units_df = pd.read_csv(os.path.join(paths.DATA_ROOT, 'properties', 'responses', 'units.csv'))
    fnn_resp = os.path.join(paths.DATA_ROOT, 'properties', 'responses', 'responses.npy')
    fnn_stim = os.path.join(paths.DATA_ROOT, 'properties', 'responses', 'stimulus.npy')

    fig = plt.figure(figsize=(18, 6.125))
    plt.subplots_adjust(top=0.92, bottom=0.10, left=0.05, right=0.92)

    left_rect = [0.08, 0.08, 0.55, 0.82]
    right_rect = [0.68, 0.12, 0.28, 0.82]

    left_gs = gridspec.GridSpec(1, 1, figure=fig,
                                left=left_rect[0], bottom=left_rect[1],
                                right=left_rect[0]+left_rect[2],
                                top=left_rect[1]+left_rect[3])
    plot_frames(fig, left_gs[0], units_df, fnn_resp, fnn_stim)
    
    left_pos = left_gs[0].get_position(fig)
    legend_ax = fig.add_axes([left_pos.x0, left_pos.y0 - 0.05, left_pos.width, 0.04])
    legend_ax.axis('off')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    labels = ['MEDI', 'Natural', 'Grating']
    
    n = len(labels)
    x_start = 0.2
    x_end = 0.8
    xs = np.linspace(x_start, x_end, n)
    
    for x, col, lab in zip(xs, colors, labels):
        outer_rect = plt.Rectangle(
            (x - 0.05, 0.2), 0.1, 0.6,
            facecolor='none', edgecolor=col, linewidth=2,
            transform=legend_ax.transAxes
        )
        legend_ax.add_patch(outer_rect)
        
        inner_rect = plt.Rectangle(
            (x - 0.05, 0.205), 0.1, 0.59,
            facecolor=col, alpha=0.3, edgecolor='none',
            transform=legend_ax.transAxes
        )
        legend_ax.add_patch(inner_rect)
        
        legend_ax.text(x + 0.07, 0.5, lab, ha='left', va='center', fontsize=14, transform=legend_ax.transAxes)

    ax_dens = fig.add_axes(right_rect)
    plot_baseline_density_all(ax_dens)

    pos_left = left_gs[0].get_position(fig)
    fig.text(pos_left.x0 - 0.02, pos_left.y1 + 0.1, 'a',
             fontsize=26, fontweight='bold', va='top')
    pos_right = ax_dens.get_position()
    fig.text(pos_right.x0 - 0.02, pos_right.y1 + 0.055, 'b',
             fontsize=26, fontweight='bold', va='top')

    out_path = os.path.join(paths.FIGURES_DIR, 'fig3.pdf')
    plt.savefig(out_path, dpi=300)
    print(f"Figure 3 generated at {out_path}")

if __name__ == "__main__":
    generate_fig3()