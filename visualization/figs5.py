import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))

if project_root not in sys.path:
    sys.path.append(project_root)

from medi_pipeline.config import paths

def get_star(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    return 'ns'

def set_style():
    sns.set_style("white", {'axes.spines.top': False, 'axes.spines.right': False})
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'font.size': 20,
        'axes.labelsize': 26,
        'axes.titlesize': 26,
        'axes.linewidth': 2.0,
        'xtick.labelsize': 26,
        'ytick.labelsize': 26,
        'xtick.major.width': 2.0,
        'ytick.major.width': 2.0,
        'legend.fontsize': 20,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',  
        'savefig.pad_inches': 0.1
    })

def generate_figs5():
    set_style()
    
    medi_df = pd.read_csv(os.path.join(paths.DATA_DIR, 'medi_features.csv'))
    grat_df = pd.read_csv(os.path.join(paths.DATA_DIR, 'grating_features.csv'))
    
    merged = pd.merge(medi_df, grat_df, on=['session', 'scan_idx', 'unit_id'], suffixes=('_medi', '_grat'))
    
    from scipy.stats import pearsonr
    
    metrics = ['gOSI', 'gDSI']
    
    plot_data = []
    for metric in metrics:
        col_medi = f"{metric}_medi"
        col_grat = f"{metric}_grat"
        
        valid = merged[[col_medi, col_grat]].dropna()
        for val in valid[col_medi]: plot_data.append({'Metric': metric, 'Value': val, 'Method': 'MEDI'})
        for val in valid[col_grat]: plot_data.append({'Metric': metric, 'Value': val, 'Method': 'Grating'})
    
    fig = plt.figure(figsize=(32, 8))
    gs = gridspec.GridSpec(1, 4, wspace=0.35)

    def plot_scatter(ax, metric_base, letter):
        col_medi, col_grat = f"{metric_base}_medi", f"{metric_base}_grat"
        clean = merged[[col_grat, col_medi]].dropna()
        sns.regplot(data=clean, x=col_grat, y=col_medi, ax=ax, scatter_kws={'alpha': 0.3, 's': 10, 'color': 'teal'}, line_kws={'color': 'black', 'linewidth': 3.5})
        r, _ = pearsonr(clean[col_grat], clean[col_medi])
        ax.text(0.05, 0.95, f'r = {r:.2f}', transform=ax.transAxes, va='top', fontsize=24, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.set_xlabel(f"Grating {metric_base}", labelpad=12, fontsize=28)
        ax.set_ylabel(f"MEDI {metric_base}", labelpad=12, fontsize=28)
        ax.tick_params(axis='both', pad=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_box_aspect(1)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.tick_params(axis='both', which='major', labelsize=24)
        ax.set_title(f"{metric_base}", pad=20, fontsize=28)
        sns.despine(ax=ax)
        ax.text(-0.18, 1.05, letter, transform=ax.transAxes, fontsize=38, fontweight='bold', va='bottom', ha='right')
        ax.tick_params(axis='both', which='major', direction='out', length=6, width=2, bottom=True, left=True)

    plot_scatter(fig.add_subplot(gs[0]), 'gOSI', 'a')
    plot_scatter(fig.add_subplot(gs[1]), 'gDSI', 'b')

    out_path = os.path.join(paths.FIGURES_DIR, 'figs5.pdf')
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.08, right=0.98)
    plt.savefig(out_path, dpi=300)
    print(f"Appendix Figure S5 generated at {out_path}")

if __name__ == "__main__":
    generate_figs5()