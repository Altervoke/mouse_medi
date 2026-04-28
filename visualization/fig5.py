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
from scipy.stats import wilcoxon

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

def generate_fig5():
    set_style()
    
    medi_df = pd.read_csv(os.path.join(paths.DATA_DIR, 'medi_features.csv'))
    grat_df = pd.read_csv(os.path.join(paths.DATA_DIR, 'grating_features.csv'))
    
    merged = pd.merge(medi_df, grat_df, on=['session', 'scan_idx', 'unit_id'], suffixes=('_medi', '_grat'))
    
    metrics = ['pref_tf', 'pref_sf', 'gOSI', 'gDSI']
    labels = {
        'pref_tf': 'Preferred TF (Hz)',
        'pref_sf': 'Preferred SF (cpp)',
        'gOSI': 'gOSI',
        'gDSI': 'gDSI'
    }
    labels2 = {
        'pref_tf': 'Preferred TF',
        'pref_sf': 'Preferred SF',
        'gOSI': 'gOSI',
        'gDSI': 'gDSI'
    }
    
    plot_data = []
    for metric in metrics:
        col_medi = f"{metric}_medi"
        col_grat = f"{metric}_grat"
        
        valid = merged[[col_medi, col_grat]].dropna()
        for val in valid[col_medi]: plot_data.append({'Metric': metric, 'Value': val, 'Method': 'MEDI'})
        for val in valid[col_grat]: plot_data.append({'Metric': metric, 'Value': val, 'Method': 'Grating'})
    df_plot = pd.DataFrame(plot_data)
    
    fig = plt.figure(figsize=(32, 8))
    gs = gridspec.GridSpec(1, 4, wspace=0.35)
    
    palette = {'MEDI': '#66B2FF', 'Grating': '#78DE9A'}
    
    raw_pvals = []
    for metric in metrics:
        sub_df = df_plot[df_plot['Metric'] == metric]
        g1 = sub_df[sub_df['Method'] == 'MEDI']['Value'].values
        g2 = sub_df[sub_df['Method'] == 'Grating']['Value'].values
        _, p_val = wilcoxon(g1, g2)
        raw_pvals.append(p_val)
    from statsmodels.stats.multitest import multipletests
    _, qvals, _, _ = multipletests(raw_pvals, method='fdr_bh')
    qval_dict = {metric: q for metric, q in zip(metrics, qvals)}
    
    def plot_violin(ax, metric, letter, x_offset=-0.25):
        sub_df = df_plot[df_plot['Metric'] == metric]
        sns.violinplot(data=sub_df, x='Method', y='Value', order=['MEDI', 'Grating'],
            palette=palette, inner='quartile', ax=ax, linewidth=2.0, hue='Method', legend=False, cut=0)
        
        m_medi = sub_df[sub_df['Method'] == 'MEDI']['Value'].mean()
        m_grat = sub_df[sub_df['Method'] == 'Grating']['Value'].mean()
        ax.plot([0, 1], [m_medi, m_grat], color='#333333', linestyle='', linewidth=3.0, marker='o', markersize=10, zorder=5, alpha=1)
        
        g1 = sub_df[sub_df['Method'] == 'MEDI']['Value'].values
        g2 = sub_df[sub_df['Method'] == 'Grating']['Value'].values
        q_val = qval_dict[metric]
        star = get_star(q_val)
        
        y_max = max(g1.max(), g2.max())
        y_min = min(g1.min(), g2.min())
        y_range = y_max - y_min
        y = y_max + 0.05 * y_range
        h = 0.02 * y_range
        x1, x2 = 0, 1
        ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=2.0, c='black')
        ax.text((x1+x2)*.5, y + h + 0.01 * y_range, star, ha='center', va='bottom', color='black', fontsize=28, fontweight='bold')
        
        if metric in ['gOSI', 'gDSI']: ax.set_ylim(0, 1.1)
        else: ax.set_ylim(bottom=ax.get_ylim()[0], top=y + h + 0.15 * y_range)
        ax.set_ylabel(labels[metric], labelpad=15, fontsize=32)
        ax.set_xlabel('')
        if metric == 'pref_tf': ax.set_yticks([0, 2, 4, 6, 8, 10])
        if metric == 'pref_sf': ax.set_yticks([0.0, 0.1, 0.2, 0.3])
        ax.tick_params(axis='both', which='major', labelsize=28)
        ax.set_title(f"{labels2[metric]}", pad=20, fontsize=32)
        sns.despine(ax=ax)
        ax.text(x_offset, 1.05, letter, transform=ax.transAxes, fontsize=50, fontweight='bold', va='bottom', ha='right')
        ax.tick_params(axis='both', which='major', direction='out', length=6, width=2, bottom=True, left=True)

    plot_violin(fig.add_subplot(gs[0]), 'pref_tf', 'a', x_offset=-0.15)
    plot_violin(fig.add_subplot(gs[1]), 'pref_sf', 'b', x_offset=-0.15)
    plot_violin(fig.add_subplot(gs[2]), 'gOSI', 'c', x_offset=-0.18)
    plot_violin(fig.add_subplot(gs[3]), 'gDSI', 'd', x_offset=-0.18)

    out_path = os.path.join(paths.FIGURES_DIR, 'fig5.pdf')
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.08, right=0.98)
    plt.savefig(out_path, dpi=300)
    print(f"Figure 5 generated at {out_path}")

if __name__ == "__main__":
    generate_fig5()