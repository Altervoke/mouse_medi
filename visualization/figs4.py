import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
        'font.size': 20,
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'axes.linewidth': 1.0,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'xtick.major.width': 2.0,
        'ytick.major.width': 2.0,
        'legend.fontsize': 20,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',  
        'savefig.pad_inches': 0.1
    })

def generate_figs4():
    set_style()
    medi_csv = os.path.join(paths.DATA_DIR, 'medi_features.csv')
    grating_csv = os.path.join(paths.DATA_DIR, 'grating_features.csv')
    
    df_MEDI_raw = pd.read_csv(medi_csv)
    df_grating_raw = pd.read_csv(grating_csv)
    
    merge_keys = ['session', 'scan_idx', 'readout_id']
    df = pd.merge(df_MEDI_raw, df_grating_raw, on=merge_keys, suffixes=('_MEDI', '_grating'))
    
    thresholds = np.arange(0, 1.0, 0.01)
    results = []
    
    for thr in thresholds:
        mask = (df['gOSI_grating'] > thr) & (df['gDSI_grating'] > thr)
        sub = df[mask]
        if len(sub) < 100:
            break
        results.append({
            'threshold': thr,
            'n': len(sub),
        })
    
    thr_vals = [r['threshold'] for r in results]
    counts = [r['n'] for r in results]
    
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    
    ax.plot(thr_vals, counts, color='#3498db', linewidth=2)
    ax.set_yscale('log')
    ax.set_ylim(bottom=100)
    ax.set_xlabel("Threshold (gOSI & gDSI)")
    ax.set_ylabel(r"Neuron Count")
    ax.set_xticks([0.0, 0.2, 0.4, 0.6])
    ax.tick_params(axis='both', which='major', direction='out', length=3, width=1, bottom=True, left=True)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    sns.despine(ax=ax)
    
    plt.tight_layout()
    
    out_path = os.path.join(paths.FIGURES_DIR, 'figs4.pdf')
    plt.savefig(out_path)
    print(f"Appendix Figure S4 generated at {out_path}")

if __name__ == "__main__":
    generate_figs4()
