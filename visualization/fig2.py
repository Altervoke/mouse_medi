import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))

if project_root not in sys.path:
    sys.path.append(project_root)

from mouse_medi.config import paths

def set_style():
    sns.set_style("white", {'axes.spines.top': False, 'axes.spines.right': False})
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'pdf.fonttype': 42,
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'axes.linewidth': 1.5,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })

def plot_r2_density(ax):
    df_r2 = pd.read_csv(os.path.join(paths.DATA_DIR, 'gabor_fit.csv'))
    sns.kdeplot(df_r2['r2'].dropna(), fill=True, color="#d6275b", ax=ax, alpha=0.4, linewidth=2, bw_adjust=1.0)
    ax.set_xlabel(r'Gabor Fit $R^2$', fontsize=20)
    sns.despine(ax=ax, left=True)
    ax.set_yticks([])
    ax.set_ylabel('')
    ax.tick_params(axis='x', labelsize=18)
    ax.set_xlim(0, 1)
    med = df_r2['r2'].median()
    ax.plot([med, med], [0, 3.2], color="#d6275b", linestyle='--', lw=2.0, zorder=0, label=f'Median: {med:.2f}')
    ax.legend(frameon=False, loc='upper left', fontsize=18)
    ax.tick_params(axis='both', which='major', direction='out', length=4, width=1.5, bottom=True, left=True)

def plot_mesi_gabor_examples(fig, outer_gs):
    df = pd.read_csv(os.path.join(paths.DATA_DIR, 'baseline_comparisons.csv'))
    df_r2 = pd.read_csv(os.path.join(paths.DATA_DIR, 'gabor_fit.csv'))

    examples = [
        ('L4', 7, 5, 6484),
        ('L5', 5, 6, 6309),
        ('L4', 6, 6, 3538),
        ('L23', 4, 7, 775),
    ]

    inner_gs = gridspec.GridSpecFromSubplotSpec(4, 2, subplot_spec=outer_gs, wspace=0.1, hspace=0.1)

    saved_axes = []
    first_ax = None
    for i, (layer, s, sc, rid) in enumerate(examples):
        area = 'V1'
        r2_match = df_r2[(df_r2['session'] == s) & (df_r2['scan_idx'] == sc) & (df_r2['readout_id'] == rid)]
        r2_val = r2_match.iloc[0]['r2'] if not r2_match.empty else np.nan
        
        m_path = os.path.join(paths.RESULTS_DIR, 'MESI', area, layer, f"{s}_{sc}_r{rid}.png")
        g_path = os.path.join(paths.RESULTS_DIR, 'Gabor', area, layer, f"{s}_{sc}_r{rid}.png")
        
        try:
            m_img = imageio.imread(m_path)
            g_img = imageio.imread(g_path)
            
            ax_m = fig.add_subplot(inner_gs[i, 0])
            ax_m.imshow(m_img, cmap='gray', vmin=0, vmax=255)
            ax_m.set_xticks([])
            ax_m.set_yticks([])
            for spine in ax_m.spines.values(): spine.set_visible(False)
            
            ax_m.text(-0.12, 0.5, f"$R^2={r2_val:.2f}$", transform=ax_m.transAxes, ha='right', va='center', fontsize=20)
            
            if i == 0:
                ax_m.set_title('MESI', fontsize=20, pad=10)
                first_ax = ax_m
            saved_axes.append(ax_m)
                
            ax_g = fig.add_subplot(inner_gs[i, 1])
            ax_g.imshow(g_img, cmap='gray', vmin=0, vmax=255)
            ax_g.set_xticks([])
            ax_g.set_yticks([])
            for spine in ax_g.spines.values(): spine.set_visible(False)
            
            if i == 0:
                ax_g.set_title('Fitted Gabor', fontsize=20, pad=10)
            saved_axes.append(ax_g)
        except Exception as e:
            print(f"Error loading image for {s}_{sc}_r{rid}: {e}")

    return saved_axes, first_ax

def plot_mesi_vs_gabor_scatter(ax):
    csv_path = os.path.join(paths.DATA_DIR, 'baseline_comparisons.csv')
    df = pd.read_csv(csv_path)
    df_v1 = df[df['brain_area'] == 'V1'].dropna(subset=['Gabor', 'MESI'])
    df_v1 = df_v1[(df_v1['Gabor'] <= 80) & (df_v1['MESI'] <= 80)]

    x = df_v1['MESI'].values
    y = df_v1['Gabor'].values
    m = np.sum(x * y) / np.sum(x * x) if np.sum(x * x) != 0 else 0

    ax.scatter(x, y, alpha=0.3, color="#d6275b", s=15)

    max_val = max(x.max(), y.max())
    min_val = min(x.min(), y.min())
    plot_max = max_val
    plot_min = min(0, min_val)

    line_x = np.array([plot_min, plot_max])
    ax.plot(line_x, m * line_x, color='black', linewidth=2.5)
    ax.plot(line_x, line_x, 'k--', linewidth=2.0)

    ax.set_xlim(plot_min, plot_max)
    ax.set_ylim(plot_min, plot_max)
    ax.tick_params(axis='both', which='major', labelsize=18)

    ax.set_xlabel('MESI Response', fontsize=20)
    ax.set_ylabel('Optimal Gabor Response', fontsize=20)

    ax.set_xticks([0, 20, 40, 60, 80])
    ax.set_yticks([0, 20, 40, 60, 80])

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis='both', which='major', direction='out', length=4, width=1.5, bottom=True, left=True)

def generate_fig2():
    set_style()

    fig = plt.figure(figsize=(22, 6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.2, 1, 1.2], wspace=0.35)

    ax_dens = fig.add_subplot(gs[0])
    plot_r2_density(ax_dens)
    
    all_axes_b, ax_b_img = plot_mesi_gabor_examples(fig, gs[1])

    ax_scatter = fig.add_subplot(gs[2])
    plot_mesi_vs_gabor_scatter(ax_scatter)
    ax_scatter.set_box_aspect(1)
    
    plt.tight_layout()
    fig.canvas.draw()
    
    delta_x = 0.018
    for ax in all_axes_b:
        pos = ax.get_position()
        new_x0 = pos.x0 + delta_x
        new_x1 = pos.x1 + delta_x
        if new_x1 < 0.95:
            ax.set_position([new_x0, pos.y0, pos.width, pos.height])
    
    lbl_font = {'fontsize': 28, 'fontweight': 'bold', 'va': 'bottom', 'ha': 'right'}
    
    pos_a = ax_dens.get_position()
    if ax_b_img:
        pos_b = ax_b_img.get_position()
    else:
        temp_ax = fig.add_subplot(gs[1]); pos_b = temp_ax.get_position(); temp_ax.remove()
        
    pos_c = ax_scatter.get_position()
    
    y_title = max(pos_a.y1, pos_b.y1, pos_c.y1) + 0.05
    
    fig.text(pos_a.x0 - 0.02, y_title, 'a', **lbl_font)
    fig.text(pos_b.x0 - 0.06, y_title, 'b', **lbl_font)
    fig.text(pos_c.x0 - 0.03, y_title, 'c', **lbl_font)

    out_path = os.path.join(paths.FIGURES_DIR, 'fig2.pdf')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Figure 2 generated at {out_path}")

if __name__ == "__main__":
    generate_fig2()
