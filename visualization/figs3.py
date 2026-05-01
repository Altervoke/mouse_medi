import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
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
        'ps.fonttype': 42,
        'font.size': 18,
        'axes.labelsize': 20,
        'axes.titlesize': 22,
        'axes.linewidth': 1.5,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',  
        'savefig.pad_inches': 0.1
    })

def generate_figs3():
    set_style()
    csv_path = os.path.join(paths.DATA_DIR, 'baseline_comparisons.csv')
    df_all = pd.read_csv(csv_path)

    areas = ['V1', 'LM', 'RL', 'AL']
    baselines = ['Gabor', 'MESI', 'Grating', 'Natural']
    display_names = {'Gabor': 'Gabor', 'MESI': 'MESI', 'Grating': 'Grating', 'Natural': 'Natural'}
    colors = {'Gabor': "#ea4444", 'MESI': "#7552b4", 'Grating': '#2ca02c', 'Natural': '#ff7f0e'}

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.5, wspace=0.25)
    
    label_font = {'fontsize': 24, 'fontweight': 'bold', 'va': 'bottom', 'ha': 'right'}
    letters = ['a', 'b', 'c', 'd']

    for i, area in enumerate(areas):
        row = i // 2
        col = i % 2
        ax = fig.add_subplot(gs[row, col])
        df = df_all[df_all['brain_area'] == area].copy()
        
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

        ax.set_title(area, fontsize=18)
        
        ax.legend(loc='upper right', fontsize=12, frameon=False)
        sns.despine(ax=ax, left=True)
        ax.set_yticks([])
        ax.set_xticks([0, 1, 4, 8, 12, 16, 20])
        ax.set_xlim(0, 20)
        
        ax.set_xlabel('Response Gain Ratio', fontsize=16, labelpad=10)
        ax.set_ylabel('')
        
        ax.text(-0.05, 1.05, letters[i], transform=ax.transAxes, **label_font)
        ax.tick_params(axis='both', which='major', direction='out', length=4, width=1.5, bottom=True, left=True)

    plt.subplots_adjust(bottom=0.1, top=0.92)

    out_path = os.path.join(paths.FIGURES_DIR, 'figs3.pdf')
    plt.savefig(out_path, dpi=300)
    print(f"Appendix Figure S3 generated at {out_path}")

if __name__ == "__main__":
    generate_figs3()