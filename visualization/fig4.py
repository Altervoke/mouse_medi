import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import imageio
import cv2
from scipy.interpolate import CubicSpline

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))

if project_root not in sys.path:
    sys.path.append(project_root)

from mouse_medi.config import paths
from mouse_medi.features.medi_analysis import estimate_direction_3d_fft

import warnings
warnings.filterwarnings('ignore')

def set_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif']
    })

def load_medi_video(video_path):
    try:
        reader = imageio.get_reader(video_path)
        frames = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) if f.ndim==3 else f, (256, 144)) for f in reader]
        return np.stack(frames).astype(np.float32) / 255.0
    except Exception as e:
        return None

def generate_fig4():
    set_style()
    medi_csv = os.path.join(paths.DATA_DIR, 'medi_features.csv')
    grating_csv = os.path.join(paths.DATA_DIR, 'grating_features.csv')
    info_csv = os.path.join(paths.DATA_DIR, 'neuron_info.csv')
    
    df_MEDI_raw = pd.read_csv(medi_csv)
    df_grating_raw = pd.read_csv(grating_csv)
    df_info = pd.read_csv(info_csv)
    
    merge_keys = ['session', 'scan_idx', 'readout_id']
    df = pd.merge(df_MEDI_raw, df_grating_raw, on=merge_keys, suffixes=('_MEDI', '_grating'))
    
    eps = 1e-9
    df['diff_tf_log2'] = np.abs(np.log2(df['pref_tf_MEDI'] + eps) - np.log2(df['pref_tf_grating'] + eps))
    df['diff_sf_log2'] = np.abs(np.log2(df['pref_sf_MEDI'] + eps) - np.log2(df['pref_sf_grating'] + eps))
    
    def get_circular_diff(a, b, period=360):
        diff = (a - b) % period
        diff[diff > period/2] -= period
        return np.abs(diff)
        
    df['diff_ori'] = get_circular_diff(df['pref_ori_MEDI'], df['pref_ori_grating'], 180)
    df['diff_dir'] = get_circular_diff(df['pref_dir_MEDI'], df['pref_dir_grating'], 360)
    
    thresholds = np.arange(0, 1.0, 0.01)
    results = []
    
    for thr in thresholds:
        mask = (df['gOSI_grating'] > thr) & (df['gDSI_grating'] > thr)
        sub = df[mask]
        if len(sub) < 100:
            break
        results.append({
            'threshold': thr,
            'corr': sub['curve_corr'].values,
            'tf_log2': sub['diff_tf_log2'].values,
            'sf_log2': sub['diff_sf_log2'].values,
            'ori': sub['diff_ori'].values,
            'dir': sub['diff_dir'].values
        })
    
    def bootstrap_median_ci(data, n_bootstrap=1000, alpha=0.05):
        if len(data) == 0:
            return (np.nan, np.nan)
        medians = []
        rng = np.random.default_rng()
        for _ in range(n_bootstrap):
            sample = rng.choice(data, size=len(data), replace=True)
            medians.append(np.median(sample))
        lower = np.percentile(medians, 100 * alpha / 2)
        upper = np.percentile(medians, 100 * (1 - alpha / 2))
        return lower, upper
    
    def mean_ci(data, alpha=0.05):
        n = len(data)
        if n == 0:
            return (np.nan, np.nan)
        mean = np.mean(data)
        se = np.std(data, ddof=1) / np.sqrt(n)
        z = 1.96
        return mean - z*se, mean + z*se
    
    thr_vals = [r['threshold'] for r in results]
    corr_medians = [np.median(r['corr']) for r in results]
    corr_lower, corr_upper = [], []
    for r in results:
        l, u = bootstrap_median_ci(r['corr'])
        corr_lower.append(l)
        corr_upper.append(u)
    
    tf_means = [np.mean(r['tf_log2']) for r in results]
    tf_lower, tf_upper = [], []
    sf_means = [np.mean(r['sf_log2']) for r in results]
    sf_lower, sf_upper = [], []
    for r in results:
        l_tf, u_tf = mean_ci(r['tf_log2'])
        tf_lower.append(l_tf); tf_upper.append(u_tf)
        l_sf, u_sf = mean_ci(r['sf_log2'])
        sf_lower.append(l_sf); sf_upper.append(u_sf)
    
    ori_medians = [np.median(r['ori']) for r in results]
    ori_lower, ori_upper = [], []
    dir_medians = [np.median(r['dir']) for r in results]
    dir_lower, dir_upper = [], []
    for r in results:
        l_o, u_o = bootstrap_median_ci(r['ori'])
        ori_lower.append(l_o); ori_upper.append(u_o)
        l_d, u_d = bootstrap_median_ci(r['dir'])
        dir_lower.append(l_d); dir_upper.append(u_d)
    
    video_dir = os.path.join(paths.RESULTS_DIR, 'MEDI')
    gif_files = glob.glob(os.path.join(video_dir, '**', '*.gif'), recursive=True)
    
    def plot_neuron_example(ax_p, ax_t, t_sess, t_scan, t_readout):
        chosen_gif = None
        for g in gif_files:
            fname = os.path.basename(g).split('.')[0]
            parts = fname.split('_')
            try:
                if int(parts[0]) == t_sess and int(parts[1]) == t_scan and int(parts[2][1:]) == t_readout:
                    chosen_gif = g
                    break
            except Exception:
                continue
        bu_df = df[(df['session'] == t_sess) & (df['scan_idx'] == t_scan) & (df['readout_id'] == t_readout)]
        best_u = bu_df.iloc[0] if not bu_df.empty else df.sort_values('curve_corr', ascending=False).iloc[0]
        
        m_tun = None
        if chosen_gif is not None:
            vid = load_medi_video(chosen_gif)
            if vid is not None:
                m_tun = estimate_direction_3d_fft(vid)
        if m_tun is None: m_tun = np.zeros(360)
            
        c_m, c_g = '#1f77b4', '#2ca02c'
        
        ax_p.set_anchor('W')
        theta_medi_rad = np.linspace(0, 2*np.pi, 361)[:-1]
        m_tun_norm = m_tun / m_tun.max() if m_tun.max() > 0 else np.zeros(360)
        ax_p.plot(theta_medi_rad, m_tun_norm, color=c_m, linewidth=4, label='MEDI')
        
        grating_cols = [f'resp_{i}' for i in range(16)]
        grat_resps = best_u[grating_cols].values.astype(float)
        grat_resps = grat_resps / grat_resps.max() if grat_resps.max() > 0 else grat_resps
        grat_resps = np.append(grat_resps, grat_resps[0])
        
        theta_grat_orig = np.linspace(0, 2*np.pi, 17)
        cs = CubicSpline(theta_grat_orig, grat_resps, bc_type='periodic')
        theta_grat_smooth = np.linspace(0, 2*np.pi, 360)
        grat_resps_smooth = np.clip(cs(theta_grat_smooth), 0, None)
        
        ax_p.plot(theta_grat_smooth, grat_resps_smooth, color=c_g, linewidth=4, label='Grating')
        ax_p.set_rmax(1.05)
        ax_p.set_rticks([0.5, 1.0])
        
        m_pref_dir = np.deg2rad(best_u['pref_dir_MEDI'])
        ax_p.annotate("", xy=(m_pref_dir, 0.5), xycoords='data', xytext=(0, 0), textcoords='data', arrowprops=dict(arrowstyle="->", color=c_m, lw=4, mutation_scale=25))
        g_pref_dir = np.deg2rad(best_u['pref_dir_grating'])
        ax_p.annotate("", xy=(g_pref_dir, 0.6), xycoords='data', xytext=(0, 0), textcoords='data', arrowprops=dict(arrowstyle="->", color=c_g, lw=4, mutation_scale=25))
        
        ax_p.set_yticklabels([])
        ax_p.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
        ax_p.tick_params(axis='x', pad=22, labelsize=22)
        ax_p.tick_params(axis='y', labelsize=20)
        if 'curve_corr' in best_u:
            corr_val = best_u['curve_corr']
            ax_p.text(0.87, -0.09, f"r = {corr_val:.2f}", color='black', transform=ax_p.transAxes, fontsize=22, bbox=dict(boxstyle="round,pad=0.3", facecolor="#F8F8F8", edgecolor="#DDDDDD"))
        ax_p.legend(loc='lower center', bbox_to_anchor=(0.5, -0.35), ncol=2, frameon=False, fontsize=22)

        ax_t.axis('off')
        cell_text = [['Feature', 'MEDI', 'Grating'],
            ['Pref TF (Hz)', f"{best_u['pref_tf_MEDI']:.1f}", f"{best_u['pref_tf_grating']:.1f}"],
            ['Pref SF (cpp)', f"{best_u['pref_sf_MEDI']:.2f}", f"{best_u['pref_sf_grating']:.2f}"],
            ['Pref Ori (deg)', f"{best_u['pref_ori_MEDI']:.1f}", f"{best_u['pref_ori_grating']:.1f}"],
            ['Pref Dir (deg)', f"{best_u['pref_dir_MEDI']:.1f}", f"{best_u['pref_dir_grating']:.1f}"],
            ['gOSI', f"{best_u['gOSI_MEDI']:.2f}", f"{best_u['gOSI_grating']:.2f}"],
            ['gDSI', f"{best_u['gDSI_MEDI']:.2f}", f"{best_u['gDSI_grating']:.2f}"]]
        table_t = ax_t.table(cellText=cell_text, colWidths=[0.55, 0.23, 0.29], loc='center', cellLoc='center', bbox=[-0.18, 0.05, 1.15, 0.9])
        table_t.auto_set_font_size(False)
        table_t.set_fontsize(18)
        for (row, col), cell in table_t.get_celld().items():
            cell.set_edgecolor('#E0E0E0')
            cell.set_linewidth(1.0)
            if row == 0: cell.set_text_props(color='black'); cell.set_facecolor('#EBEBEB')
            else:
                if col == 0: cell.set_text_props(color='black', ha='left'); cell.PAD = 0.05
                else:
                    if col == 1: cell.set_text_props(color=c_m)
                    elif col == 2: cell.set_text_props(color=c_g)
                cell.set_facecolor('#FAFAFA' if row % 2 == 1 else '#FFFFFF')
    
    plt.rcParams.update({'font.size': 22, 'axes.labelsize': 24, 'axes.titlesize': 26})
    fig = plt.figure(figsize=(26, 12))
    
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.2, 1.0], hspace=0.65)
    
    gs_row0 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[0], width_ratios=[1.0, 0.8, 1.0, 0.8], wspace=0.35)
    
    ax_p1 = fig.add_subplot(gs_row0[0], projection='polar')
    ax_t1 = fig.add_subplot(gs_row0[1])
    plot_neuron_example(ax_p1, ax_t1, 5, 7, 3838)
    
    ax_p2 = fig.add_subplot(gs_row0[2], projection='polar')
    ax_t2 = fig.add_subplot(gs_row0[3])
    plot_neuron_example(ax_p2, ax_t2, 6, 7, 4143)
    
    gs_row1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1], wspace=0.35)
    
    ax_c = fig.add_subplot(gs_row1[0])
    ax_c.fill_between(thr_vals, corr_lower, corr_upper, color='#CCB974', alpha=0.2)
    ax_c.plot(thr_vals, corr_medians, color='#CCB974', linewidth=5, label='Median Corr')
    ax_c.set_xlabel("Threshold (gOSI & gDSI)")
    ax_c.set_ylabel("Median Correlation")
    ax_c.set_xticks([0.0, 0.2, 0.4, 0.6]); ax_c.set_yticks([0.5, 0.6, 0.7, 0.8])
    
    ax_d = fig.add_subplot(gs_row1[1])
    ax_d.fill_between(thr_vals, tf_lower, tf_upper, color='#C44E52', alpha=0.2)
    ax_d.plot(thr_vals, tf_means, color='#C44E52', linewidth=5, label=r'Pref TF')
    ax_d.fill_between(thr_vals, sf_lower, sf_upper, color='#8172B3', alpha=0.2)
    ax_d.plot(thr_vals, sf_means, color='#8172B3', linewidth=5, label=r'Pref SF')
    ax_d.set_xlabel("Threshold (gOSI & gDSI)")
    ax_d.set_ylabel(r"Mean Diff. ($\log_2$)")
    ax_d.margins(y=0.4)
    ax_d.set_xticks([0.0, 0.2, 0.4, 0.6]); ax_d.set_yticks([0, 0.4, 0.8, 1.2, 1.6])
    ax_d.legend(bbox_to_anchor=(1.0, 1.15), loc='upper right', frameon=False)
    
    ax_e = fig.add_subplot(gs_row1[2])
    ax_e.fill_between(thr_vals, ori_lower, ori_upper, color='#FF8C00', alpha=0.2)
    ax_e.plot(thr_vals, ori_medians, color='#FF8C00', linewidth=5, label=r'Pref Ori')
    ax_e.fill_between(thr_vals, dir_lower, dir_upper, color='#00CED1', alpha=0.2)
    ax_e.plot(thr_vals, dir_medians, color='#00CED1', linewidth=5, label=r'Pref Dir')
    ax_e.set_xlabel("Threshold (gOSI & gDSI)")
    ax_e.set_ylabel(r"Median Diff. (deg)")
    ax_e.set_xticks([0.0, 0.2, 0.4, 0.6]); ax_e.set_yticks([0, 10, 20, 30, 40, 50])
    ax_e.legend(bbox_to_anchor=(1.0, 1.15), loc='upper right', frameon=False)
        
    for ax in [ax_d, ax_c, ax_e]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout(pad=3.0, rect=[0.02, 0, 1, 0.95])
    fig.canvas.draw()
    
    lbl_font = {'fontsize': 38, 'fontweight': 'bold', 'va': 'bottom', 'ha': 'right'}
    
    y_row0 = ax_p1.get_position().y1 + 0.04
    fig.text(ax_p1.get_position().x0 - 0.02, y_row0, 'a', **lbl_font)
    fig.text(ax_p2.get_position().x0 - 0.02, y_row0, 'b', **lbl_font)
    
    y_row1 = ax_c.get_position().y1 + 0.04
    fig.text(ax_c.get_position().x0 - 0.04, y_row1, 'c', **lbl_font)
    fig.text(ax_d.get_position().x0 - 0.04, y_row1, 'd', **lbl_font)
    fig.text(ax_e.get_position().x0 - 0.04, y_row1, 'e', **lbl_font)
    
    out_path = os.path.join(paths.FIGURES_DIR, 'fig4.pdf')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Figure 4 generated at {out_path}")

if __name__ == "__main__":
    generate_fig4()