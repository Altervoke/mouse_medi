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
        'font.size': 20,
        'axes.labelsize': 26,
        'axes.titlesize': 26,
        'axes.linewidth': 1.0,
        'xtick.labelsize': 26,
        'ytick.labelsize': 26,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'legend.fontsize': 20,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',  
        'savefig.pad_inches': 0.1
    })

def generate_figs2():
    set_style()
    
    KEY = ["session", "scan_idx", "readout_id"]
    
    CONDITION_FILES = {
        "MEDI (default)":                     "medi_features.csv",
        r"$\lambda_{\text{sp}}$=100":         "medi_features-ls_100.csv",
        r"$\lambda_{\text{sp}}$=1000":        "medi_features-ls_1000.csv",
        r"$\lambda_{\text{t1}}$=1000":        "medi_features-pt_1000.csv",
        r"$\lambda_{\text{t1}}$=10000":       "medi_features-pt_10000.csv",
        r"$\lambda_{\text{t2}}$=700":         "medi_features-lt_700.csv",
        r"$\lambda_{\text{t2}}$=7000":        "medi_features-lt_7000.csv",
    }
    
    COLORS = {
        "MEDI (default)":                     "#000000",
        r"$\lambda_{\text{sp}}$=100":         "#0072B2",
        r"$\lambda_{\text{sp}}$=1000":        "#72B8E8",
        r"$\lambda_{\text{t1}}$=1000":        "#CC79A7",
        r"$\lambda_{\text{t1}}$=10000":       "#EAC0D8",
        r"$\lambda_{\text{t2}}$=700":         "#009E73",
        r"$\lambda_{\text{t2}}$=7000":        "#70DEBB",
    }
    
    raw = {}
    for label, fname in CONDITION_FILES.items():
        fp = os.path.join(paths.DATA_DIR, fname)
        if os.path.exists(fp):
            df = pd.read_csv(fp)
            if 'resp' in df.columns:
                pass
            raw[label] = df
            
    def to_keys(df):
        return set(map(tuple, df[KEY].values))

    import functools
    common_keys = functools.reduce(lambda a, b: a & b, [to_keys(df) for df in raw.values()])
    
    conditions = {
        label: df[df.apply(lambda r: tuple(r[KEY]) in common_keys, axis=1)].copy()
        for label, df in raw.items()
    }
    
    metrics = [
        ("resp", 'Response', 'a', (0, 30)),
        ("STII", 'STII', 'b', (0, 1)),
        ("gOSI", 'gOSI', 'c', (0, 1)),
        ("gDSI", 'gDSI', 'd', (0, 1))
    ]
    
    fig = plt.figure(figsize=(8, 10))
    gs = gridspec.GridSpec(2, 2, wspace=0.3, hspace=0.4)
    
    def plot_scatter_panel(ax, col, title, letter, xlim=None):
        df_ref = conditions["MEDI (default)"].set_index(KEY)
        if col not in df_ref.columns:
            return
            
        ref_vals = df_ref[col].dropna()
        r_values = {}
        
        for label, df in conditions.items():
            if label == "MEDI (default)":
                continue
            if col not in df.columns:
                continue

            df_cond = df.set_index(KEY)
            cond_vals = df_cond[col].dropna()
            
            idx_common = ref_vals.index.intersection(cond_vals.index)
            if len(idx_common) == 0:
                continue
                
            x = ref_vals.loc[idx_common]
            y = cond_vals.loc[idx_common]
            
            if col == 'resp':
                q_x = x.quantile(0.99)
                q_y = y.quantile(0.99)
                mask = (x <= q_x) & (y <= q_y)
                x, y = x[mask], y[mask]

            if len(x) > 1:
                mu_x, mu_y = x.mean(), y.mean()
                cov_xy = np.cov(x.values, y.values)[0, 1]
                ccc = 2 * cov_xy / (x.var() + y.var() + (mu_x - mu_y) ** 2)
                if not np.isnan(ccc):
                    r_values[label] = ccc

            ax.scatter(x, y, label=label, color=COLORS[label], alpha=0.5, s=25, edgecolors='none')

        if len(r_values) > 0:
            if col == 'resp':
                ax.set_xticks([0, 20, 40, 60, 80])
                ax.set_yticks([0, 20, 40, 60, 80])
                ax.set_xlim(0, 80)
                ax.set_ylim(0, 80)
                ax.plot((0, 80), (0, 80), color='black', linestyle='--', linewidth=1.5, alpha=0.5, zorder=0)
            else:
                ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.plot((0, 1), (0, 1), color='black', linestyle='--', linewidth=1.5, alpha=0.5, zorder=0)

            y_pos = 0.95
            for lbl, r_val in r_values.items():
                ax.text(0.05, y_pos, rf"$\rho_c$ = {r_val:.3f}", transform=ax.transAxes, color=COLORS[lbl], fontsize=12, va='top', ha='left', fontweight=500)
                y_pos -= 0.08

        ax.set_title(title, fontsize=14, pad=12)
        ax.tick_params(labelsize=12)

        ax.set_xlabel(f"Default {title}", labelpad=12, fontsize=14)
        ax.set_ylabel(f"Variants {title}", labelpad=12, fontsize=14)

        ax.set_box_aspect(1)
        sns.despine(ax=ax)
        
        ax.text(-0.18, 1.05, letter, transform=ax.transAxes, fontsize=20, fontweight='bold', va='bottom', ha='right')
        
        ax.tick_params(axis='both', which='major', direction='out', length=3, width=1, bottom=True, left=True)

    for idx, (col, title, letter, _) in enumerate(metrics):
        row = idx // 2
        col_idx = idx % 2
        ax = fig.add_subplot(gs[row, col_idx])
        plot_scatter_panel(ax, col, title, letter)

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=COLORS[l], alpha=0.75, label=l) for l in conditions if l != "MEDI (default)"]
    fig.legend(handles, [l for l in conditions if l != "MEDI (default)"],
               loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.02),
               fontsize=12, frameon=False)

    out_path = os.path.join(paths.FIGURES_DIR, 'figs2.pdf')
    plt.subplots_adjust(top=0.92, bottom=0.18, left=0.08, right=0.98)
    plt.savefig(out_path, dpi=300)
    print(f"Appendix Figure S2 generated at {out_path}")

if __name__ == "__main__":
    generate_figs2()
