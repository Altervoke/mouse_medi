import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import ttest_ind, gaussian_kde

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))

if project_root not in sys.path:
    sys.path.append(project_root)

from medi_pipeline.config import paths

import warnings
warnings.filterwarnings('ignore')

def get_star(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    return 'ns'

def generate_fig6():
    medi_features = pd.read_csv(os.path.join(paths.DATA_DIR, 'medi_features.csv'))
    grating_features = pd.read_csv(os.path.join(paths.DATA_DIR, 'grating_features.csv'))
    neuron_info = pd.read_csv(os.path.join(paths.DATA_DIR, 'neuron_info.csv'))
    
    df = pd.merge(medi_features, grating_features, on=['session', 'scan_idx', 'unit_id'], how='inner', suffixes=('_medi', '_grating'))
    df = pd.merge(df, neuron_info, on=['session', 'scan_idx', 'unit_id'], how='inner')
    
    stii_col = 'STII' if 'STII' in df.columns else 'stii_snr_mask'
    df_stii = df.dropna(subset=[stii_col]).copy()
    
    if 'layer' in df_stii.columns:
        df_stii['layer'] = df_stii['layer'].replace({'L23': 'L2/3'})

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 38,
        'axes.labelsize': 38,
        'axes.titlesize': 38,
        'xtick.labelsize': 32,
        'ytick.labelsize': 32,
        'legend.fontsize': 36
    })
    
    fig = plt.figure(figsize=(32, 8.5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.4, 1.1, 1.1], wspace=0.45)
    
    axA = fig.add_subplot(gs[0])
    
    c_medi = "#a81fb4"
    sns.kdeplot(data=df_stii, x=stii_col, fill=True, color=c_medi, lw=2, ax=axA, alpha=0.4, bw_adjust=1.8)
    
    stii_mean = df_stii[stii_col].mean()
    stii_data = df_stii[stii_col].values
    kde = gaussian_kde(stii_data, bw_method=0.35)
    y_mean = kde.evaluate(stii_mean)[0]
    y_max = kde.evaluate(stii_data).max()
    
    axA.plot([stii_mean, stii_mean], [0, y_max + 0.3], color=c_medi, linestyle='--', lw=4.0, zorder=0, label=f'Mean: {stii_mean:.3f}')
    axA.legend(frameon=False, loc='upper left', prop={'size':32})
    
    current_ymax = axA.get_ylim()[1]
    axA.set_ylim(bottom=0, top=max(y_mean, current_ymax)*1.4)
    
    axA.set_xlabel('STII', labelpad=15)
    axA.set_ylabel('', labelpad=10)
    axA.set_xlim(0, 1.0)
    axA.set_yticks([])
    sns.despine(ax=axA, left=True, right=True, top=True)

    area_colors = sns.color_palette("plasma", 6)[1:5]
    color_area = {'V1': area_colors[0], 'LM': area_colors[1], 'RL': area_colors[2], 'AL': area_colors[3]}
    layer_colors = sns.color_palette("Blues", 5)[2:5]
    color_layer = {'L2/3': layer_colors[0], 'L4': layer_colors[1], 'L5': layer_colors[2]}
    
    axB = fig.add_subplot(gs[1])
    area_order = ['V1', 'LM', 'RL', 'AL']
    filtered_area = df_stii[df_stii['brain_area'].isin(area_order)]
    
    sns.violinplot(
        data=filtered_area, x='brain_area', y=stii_col, order=area_order,
        palette=[color_area.get(a, '#333333') for a in area_order], inner='quartile', alpha=0.8, ax=axB, linewidth=2, hue='brain_area', legend=False
    )
    
    means_b = [filtered_area[filtered_area['brain_area'] == a][stii_col].mean() for a in area_order]
    x_pos_b = np.arange(len(area_order))
    axB.plot(x_pos_b, means_b, color='black', linestyle='', linewidth=2, marker='o', markersize=8, zorder=5, alpha=1)
    
    pvals_area = []
    groups = []
    for i in range(len(area_order) - 1):
        g1 = filtered_area[filtered_area['brain_area'] == area_order[i]][stii_col].dropna()
        g2 = filtered_area[filtered_area['brain_area'] == area_order[i+1]][stii_col].dropna()
        if len(g1) > 0 and len(g2) > 0:
            _, p_val = ttest_ind(g1, g2, equal_var=False)
            pvals_area.append(p_val)
            groups.append((area_order[i], area_order[i+1]))
        else:
            pvals_area.append(np.nan)
            groups.append((area_order[i], area_order[i+1]))

    from statsmodels.stats.multitest import multipletests
    reject, p_corrected, _, _ = multipletests(pvals_area, method='fdr_bh')

    maxs_c = [filtered_area[filtered_area['brain_area'] == a][stii_col].max() for a in area_order]
    for idx, (a1, a2) in enumerate(groups):
        if np.isnan(p_corrected[idx]):
            continue
        star = get_star(p_corrected[idx])
        x1, x2 = idx, idx + 1
        y = max(maxs_c[x1], maxs_c[x2]) + 0.05 + idx * 0.04
        h = 0.02
        axB.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=2.0, c='black')
        axB.text((x1+x2)*.5, y + h + 0.01, star, ha='center', va='bottom', color='black', fontsize=38)
    
    axB.set_ylabel('STII', labelpad=15)
    axB.set_xlabel('Visual Area', labelpad=15)
    axB.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    sns.despine(ax=axB)
    
    axC = fig.add_subplot(gs[2], sharey=axB)
    layer_order = ['L2/3', 'L4', 'L5']
    filtered_layer = df_stii[df_stii['layer'].isin(layer_order)]
    
    sns.violinplot(
        data=filtered_layer, x='layer', y=stii_col, order=layer_order,
        palette=[color_layer.get(l, '#333333') for l in layer_order], inner='quartile', alpha=0.8, ax=axC, linewidth=2, hue='layer', legend=False
    )
    
    means_c = [filtered_layer[filtered_layer['layer'] == l][stii_col].mean() for l in layer_order]
    x_pos_d = np.arange(len(layer_order))
    axC.plot(x_pos_d, means_c, color='black', linestyle='', linewidth=2, marker='o', markersize=8, zorder=5, alpha=1)
    
    maxs_c = [filtered_layer[filtered_layer['layer'] == l][stii_col].max() for l in layer_order]
    
    pvals_layer = []
    groups_layer = []
    
    for i in range(len(layer_order) - 1):
        g1 = filtered_layer[filtered_layer['layer'] == layer_order[i]][stii_col].dropna()
        g2 = filtered_layer[filtered_layer['layer'] == layer_order[i+1]][stii_col].dropna()
        if len(g1) > 0 and len(g2) > 0:
            _, p_val = ttest_ind(g1, g2, equal_var=False)
            pvals_layer.append(p_val)
            groups_layer.append((i, i+1))
        else:
            pvals_layer.append(np.nan)
            groups_layer.append((i, i+1))
            
    if pvals_layer:
        valid_indices = [i for i, p in enumerate(pvals_layer) if not np.isnan(p)]
        valid_pvals = [pvals_layer[i] for i in valid_indices]
        if valid_pvals:
            _, p_corrected_layer, _, _ = multipletests(valid_pvals, method='fdr_bh')
            for idx, corr_pval in zip(valid_indices, p_corrected_layer):
                pvals_layer[idx] = corr_pval

    for idx, (i, next_i) in enumerate(groups_layer):
        if not np.isnan(pvals_layer[idx]):
            star = get_star(pvals_layer[idx])
            x1, x2 = i, i + 1
            y, h = max(maxs_c[i], maxs_c[i+1]) + 0.05 + i*0.04, 0.02
            axC.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=2.0, c='black')
            axC.text((x1+x2)*.5, y + h + 0.01, star, ha='center', va='bottom', color='black', fontsize=38)
    
    axC.set_ylabel('STII', labelpad=15)
    axC.tick_params(labelleft=True)
    axC.set_xlabel('Cortical Layer', labelpad=15)
    axC.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    sns.despine(ax=axC)
    
    plt.tight_layout(pad=1.0)
    plt.subplots_adjust(bottom=0.22, top=0.85, left=0.055, right=0.98)
    
    fig.canvas.draw()
    
    posA = axA.get_position()
    posB = axB.get_position()
    posC = axC.get_position()
    
    posB.y0 -= 0.03
    posB.y1 -= 0.03
    axB.set_position(posB)
    
    posC.y0 -= 0.02
    posC.y1 -= 0.02
    axC.set_position(posC)
    
    posA = axA.get_position()
    posB = axB.get_position()
    posC = axC.get_position()
    
    y_title = max(posA.y1, posB.y1, posC.y1) + 0.04
    fig.text(posA.x0 - 0.035, y_title, 'a', fontsize=50, fontweight='bold', va='bottom', ha='right')
    fig.text(posB.x0 - 0.035, y_title, 'b', fontsize=50, fontweight='bold', va='bottom', ha='right')
    fig.text(posC.x0 - 0.035, y_title, 'c', fontsize=50, fontweight='bold', va='bottom', ha='right')
    
    out_path = os.path.join(paths.FIGURES_DIR, 'fig6.pdf')
    plt.savefig(out_path, dpi=300)
    print(f"Figure 6 generated at {out_path}")

if __name__ == "__main__":
    generate_fig6()
