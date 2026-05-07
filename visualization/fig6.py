import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import ttest_ind, gaussian_kde, mannwhitneyu
from statsmodels.stats.multitest import multipletests

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))

if project_root not in sys.path:
    sys.path.append(project_root)

from mouse_medi.config import paths

import warnings
warnings.filterwarnings('ignore')

def get_star(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    return 'ns'

def compute_significance_from_data(df_medi, df_grat, df_info):
    merge_keys = ['session', 'scan_idx', 'readout_id']
    if 'readout_id' not in df_medi.columns and 'unit_id' in df_medi.columns:
        merge_keys = ['session', 'scan_idx', 'unit_id']
        
    d_medi = pd.merge(df_medi, df_info[merge_keys + ['brain_area']], on=merge_keys)
    d_grat = pd.merge(df_grat, d_medi[merge_keys + ['brain_area']], on=merge_keys)
    
    areas = ['V1', 'LM', 'RL', 'AL']
    metrics = ['TF', 'SF']
    
    data = {'TF': {'MEDI': {}, 'Grating': {}}, 'SF': {'MEDI': {}, 'Grating': {}}}
    
    p_values = []
    keys = []
    
    for metric in metrics:
        col_name = f'pref_{metric.lower()}'
        for a1 in areas:
            for a2 in areas:
                if a1 == a2:
                    continue
                    
                d1 = d_medi[d_medi['brain_area'] == a1][col_name].dropna()
                d2 = d_medi[d_medi['brain_area'] == a2][col_name].dropna()
                if len(d1) > 1 and len(d2) > 1:
                    _, p_val = mannwhitneyu(d1, d2)
                    p_values.append(p_val)
                    keys.append((metric, 'MEDI', a1, a2))
                else:
                    data[metric]['MEDI'][(a1, a2)] = 'N/A'
                    
                g1 = d_grat[d_grat['brain_area'] == a1][col_name].dropna()
                g2 = d_grat[d_grat['brain_area'] == a2][col_name].dropna()
                if len(g1) > 1 and len(g2) > 1:
                    _, p_val = mannwhitneyu(g1, g2)
                    p_values.append(p_val)
                    keys.append((metric, 'Grating', a1, a2))
                else:
                    data[metric]['Grating'][(a1, a2)] = 'N/A'
                    
    if p_values:
        _, p_vals_fdr, _, _ = multipletests(p_values, method='fdr_bh')
        for (metric, method, a1, a2), p_fdr in zip(keys, p_vals_fdr):
            data[metric][method][(a1, a2)] = get_star(p_fdr)
            
    return data

def generate_fig6():
    medi_features = pd.read_csv(os.path.join(paths.DATA_DIR, 'medi_features.csv'))
    grating_features = pd.read_csv(os.path.join(paths.DATA_DIR, 'grating_features.csv'))
    neuron_info = pd.read_csv(os.path.join(paths.DATA_DIR, 'neuron_info.csv'))
    
    merge_keys = ['session', 'scan_idx', 'unit_id']
    if 'readout_id' in medi_features.columns:
        merge_keys = ['session', 'scan_idx', 'readout_id']
        
    df = pd.merge(medi_features, grating_features, on=merge_keys, how='inner', suffixes=('_medi', '_grating'))
    df = pd.merge(df, neuron_info, on=merge_keys, how='inner')
    
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
    
    fig = plt.figure(figsize=(36, 16))
    
    gs = gridspec.GridSpec(2, 3, width_ratios=[1.4, 1.1, 2.5], height_ratios=[1.0, 1.0], wspace=0.45, hspace=0.45)
    
    axA = fig.add_subplot(gs[0, 0])
    
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
    
    axB = fig.add_subplot(gs[0, 1])
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
    
    axC = fig.add_subplot(gs[1, 0])
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
    layer_pairs = [(0,1), (1,2), (0,2)]
    
    for i, j in layer_pairs:
        g1 = filtered_layer[filtered_layer['layer'] == layer_order[i]][stii_col].dropna()
        g2 = filtered_layer[filtered_layer['layer'] == layer_order[j]][stii_col].dropna()
        if len(g1) > 0 and len(g2) > 0:
            _, p_val = ttest_ind(g1, g2, equal_var=False)
            pvals_layer.append(p_val)
            groups_layer.append((i, j))
        else:
            pvals_layer.append(np.nan)
            groups_layer.append((i, j))
            
    if pvals_layer:
        valid_indices = [i for i, p in enumerate(pvals_layer) if not np.isnan(p)]
        valid_pvals = [pvals_layer[i] for i in valid_indices]
        if valid_pvals:
            _, p_corrected_layer, _, _ = multipletests(valid_pvals, method='fdr_bh')
            for idx, corr_pval in zip(valid_indices, p_corrected_layer):
                pvals_layer[idx] = corr_pval

    for idx, (i, j) in enumerate(groups_layer):
        if not np.isnan(pvals_layer[idx]):
            star = get_star(pvals_layer[idx])
            x1, x2 = i, j
            y = max(maxs_c[i], maxs_c[j]) + 0.05 + idx * 0.04
            if i == 0 and j == 2:
                y += 0.12
            h = 0.02
            axC.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=2.0, c='black')
            axC.text((x1+x2)*.5, y + h + 0.01, star, ha='center', va='bottom', color='black', fontsize=38)
    
    axC.set_ylabel('STII', labelpad=15)
    axC.tick_params(labelleft=True)
    axC.set_xlabel('Cortical Layer', labelpad=15)
    axC.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    sns.despine(ax=axC)
    
    axD = fig.add_subplot(gs[:, 2])
    
    areas = ['V1', 'LM', 'RL', 'AL']
    medi_tf_mean, medi_tf_sem, grat_tf_mean, grat_tf_sem = [], [], [], []
    medi_sf_mean, medi_sf_sem, grat_sf_mean, grat_sf_sem = [], [], [], []
    
    df_medi_clean = pd.merge(medi_features, neuron_info[merge_keys + ['brain_area']], on=merge_keys)
    df_grat_clean = pd.merge(grating_features, df_medi_clean[merge_keys + ['brain_area']], on=merge_keys)
    
    for area in areas:
        m_a = df_medi_clean[df_medi_clean['brain_area'] == area]
        medi_tf_mean.append(m_a['pref_tf'].mean()); medi_tf_sem.append(m_a['pref_tf'].sem())
        medi_sf_mean.append(m_a['pref_sf'].mean()); medi_sf_sem.append(m_a['pref_sf'].sem())
        
        g_a = df_grat_clean[df_grat_clean['brain_area'] == area]
        grat_tf_mean.append(g_a['pref_tf'].mean()); grat_tf_sem.append(g_a['pref_tf'].sem())
        grat_sf_mean.append(g_a['pref_sf'].mean()); grat_sf_sem.append(g_a['pref_sf'].sem())
    
    x_positions = np.arange(len(areas))
    c_m, c_g = '#1f77b4', '#2ca02c'
    axD.errorbar(x_positions, medi_tf_mean, yerr=medi_tf_sem, color=c_m, marker='o', linestyle='-', linewidth=4, label='MEDI TF', capsize=12, capthick=4, elinewidth=4, markersize=12)
    axD.errorbar(x_positions, grat_tf_mean, yerr=grat_tf_sem, color=c_g, marker='o', linestyle='-', linewidth=4, label='Grating TF', capsize=12, capthick=4, elinewidth=4, markersize=12)
    axD.set_ylabel('Mean Preferred TF (Hz)', color='black', fontsize=38, labelpad=20)
    axD.set_xticks(x_positions); axD.set_xticklabels(areas, fontsize=32)
    axD.margins(y=0.45); axD.spines['top'].set_visible(False)
    
    axD_sf = axD.twinx()
    axD_sf.errorbar(x_positions, medi_sf_mean, yerr=medi_sf_sem, color=c_m, marker='s', linestyle='--', linewidth=4, label='MEDI SF', capsize=12, capthick=4, elinewidth=4, markersize=12)
    axD_sf.errorbar(x_positions, grat_sf_mean, yerr=grat_sf_sem, color=c_g, marker='s', linestyle='--', linewidth=4, label='Grating SF', capsize=12, capthick=4, elinewidth=4, markersize=12)
    axD_sf.set_ylabel('Mean Preferred SF (cpp)', color='black', fontsize=38, labelpad=20)
    axD_sf.margins(y=0.45); axD_sf.spines['top'].set_visible(False)
    
    h1, l1 = axD.get_legend_handles_labels()
    h2, l2 = axD_sf.get_legend_handles_labels()
    axD.legend(h1+h2, l1+l2, bbox_to_anchor=(0.1, 1.0), loc='upper left', frameon=False, ncol=2, fontsize=32)
    
    axE = fig.add_subplot(gs[1, 1])
    sig_data = compute_significance_from_data(medi_features, grating_features, neuron_info)
    axE.set_xlim(0, len(areas)); axE.set_ylim(0, len(areas)); axE.invert_yaxis()
    axE.set_xticks(np.arange(len(areas)) + 0.5); axE.set_yticks(np.arange(len(areas)) + 0.5)
    axE.set_xticklabels(areas); axE.set_yticklabels(areas)
    axE.xaxis.tick_top()
    axE.tick_params(axis='both', which='both', length=0, pad=12)
    for i in range(len(areas) + 1):
        axE.axhline(i, color='black', linewidth=0.5); axE.axvline(i, color='black', linewidth=0.5)
        
    for i, a1 in enumerate(areas):
        for j, a2 in enumerate(areas):
            if i == j: axE.fill_between([j, j+1], [i, i], [i+1, i+1], color='#F0F0F0'); continue
            
            metric = 'TF' if i > j else 'SF'
            sym_m = sig_data.get(metric, {}).get('MEDI', {}).get((a1, a2), 'ns')
            sym_g = sig_data.get(metric, {}).get('Grating', {}).get((a1, a2), 'ns')
            
            axE.plot([j+1, j], [i, i+1], color='black', linewidth=0.5)
            axE.text(j + 0.3, i + 0.3, sym_m, ha='center', va='center', color=c_m, fontsize=28)
            axE.text(j + 0.7, i + 0.8, sym_g, ha='center', va='center', color=c_g, fontsize=28)
                
    axE.text(0.5, -0.05, "TF Significance", transform=axE.transAxes, ha='center', va='top', fontsize=38, color='black')
    axE.text(1.05, 0.5, "SF Significance", transform=axE.transAxes, ha='left', va='center', rotation=-90, fontsize=38, color='black')
    axE.set_box_aspect(0.9)
    for spine in axE.spines.values(): spine.set_visible(False)
    
    plt.tight_layout(pad=3.0)
    fig.canvas.draw()
    
    lbl_font = {'fontsize': 50, 'fontweight': 'bold', 'va': 'bottom', 'ha': 'right'}
    
    _posA = axA.get_position()
    _posB = axB.get_position()
    _posC = axC.get_position()
    _posD = axD.get_position()
    _posE = axE.get_position()
    
    y_top = max(_posA.y1, _posB.y1, _posD.y1) + 0.04
    fig.text(_posA.x0 - 0.032, y_top, 'a', **lbl_font)
    fig.text(_posB.x0 - 0.032, y_top, 'b', **lbl_font)
    fig.text(_posD.x0 - 0.032, y_top, 'd', **lbl_font)
    
    y_bot = max(_posC.y1, _posE.y1) + 0.01
    fig.text(_posC.x0 - 0.032, y_bot, 'c', **lbl_font)
    fig.text(_posE.x0 - 0.032, y_bot, 'e', **lbl_font)
    
    out_path = os.path.join(paths.FIGURES_DIR, 'fig6.pdf')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Figure 6 generated at {out_path}")

if __name__ == "__main__":
    generate_fig6()
