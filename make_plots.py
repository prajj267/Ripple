"""
Generate the RIPPLE figure set from results.json, in the style of the TOFU
reference paper (scatter/line comparisons, grouped bars, trial traces,
distribution histograms, a feature-comparison table).
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11,
    'legend.fontsize': 9.5, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'figure.dpi': 130, 'savefig.dpi': 200, 'axes.grid': True,
    'grid.alpha': 0.30, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
    'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans',
    'lines.linewidth': 2.0, 'lines.markersize': 6,
})

# Output directory for figures. Override with the FIGDIR environment
# variable, or edit this line.
OUTDIR = os.environ.get('FIGDIR',
                        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     'figures'))
os.makedirs(OUTDIR, exist_ok=True)

# consistent per-scheme identity
STY = {
    'RIPPLE': dict(color='#1b3a6b', marker='o', ls='-'),
    'IND-GR': dict(color='#c0392b', marker='s', ls='--'),
    'CFR':    dict(color='#1e8449', marker='^', ls='-.'),
    'TAS':    dict(color='#d68910', marker='D', ls=':'),
}
ORDER = ['RIPPLE', 'IND-GR', 'CFR', 'TAS']
_HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(_HERE, 'results.json')))
saved = []


def finish(fig, name):
    fig.tight_layout()
    p = os.path.join(OUTDIR, name)
    fig.savefig(p, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    saved.append(p)
    print('saved', name)


def sweep_xy(sweep, scheme, key):
    xs = D[sweep]['values']
    ys = [row[key] for row in D[sweep][scheme]]
    es = [row.get(key + '_std', 0.0) for row in D[sweep][scheme]]
    return np.array(xs), np.array(ys), np.array(es)


# ======================================================================
# GROUP A -- performance vs swarm size N
# ======================================================================
def fig_welfare_N():
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for s in ORDER:
        x, y, e = sweep_xy('sweep_N', s, 'welfare_norm')
        ax.errorbar(x, y, yerr=e, label=s, capsize=3, **STY[s])
    ax.set_xlabel('Number of UAVs, $N$')
    ax.set_ylabel('Normalised system welfare  $\\overline{W}$')
    ax.set_title('System welfare vs swarm size')
    ax.legend()
    finish(fig, '01_welfare_vs_N.png')


def fig_cascade_N():
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for s in ORDER:
        x, y, e = sweep_xy('sweep_N', s, 'cascades')
        ax.plot(x, y, label=s, **STY[s])
    ax.set_xlabel('Number of UAVs, $N$')
    ax.set_ylabel('Cascade events per trial')
    ax.set_title('Cascade events vs swarm size')
    ax.legend()
    finish(fig, '02_cascades_vs_N.png')


def fig_mig_N():
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for s in ORDER:
        x, y, e = sweep_xy('sweep_N', s, 'mig_cost')
        ax.plot(x, y, label=s, **STY[s])
    ax.set_xlabel('Number of UAVs, $N$')
    ax.set_ylabel('Total migration cost')
    ax.set_title('Migration cost vs swarm size')
    ax.set_yscale('log')
    ax.legend()
    finish(fig, '03_migration_vs_N.png')


def fig_latency_N():
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for s in ORDER:
        x, y, e = sweep_xy('sweep_N', s, 'latency')
        ax.errorbar(x, y, yerr=e, label=s, capsize=3, **STY[s])
    ax.set_xlabel('Number of UAVs, $N$')
    ax.set_ylabel('Recovery latency (rounds)')
    ax.set_title('Recovery latency vs swarm size')
    ax.legend()
    finish(fig, '04_latency_vs_N.png')


def fig_restored_N():
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for s in ORDER:
        x, y, e = sweep_xy('sweep_N', s, 'restored_frac')
        ax.errorbar(x, y, yerr=e, label=s, capsize=3, **STY[s])
    ax.set_xlabel('Number of UAVs, $N$')
    ax.set_ylabel('Fraction of degraded coalitions restored')
    ax.set_title('Restoration reliability vs swarm size')
    ax.legend()
    finish(fig, '05_restored_vs_N.png')


def fig_comptime_N():
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for s in ORDER:
        x, y, e = sweep_xy('sweep_N', s, 'comp_time')
        ax.plot(x, y, label=s, **STY[s])
    # O(N max(M,N)^2) reference with M = N/6 (<= N), so ~ O(N^3)
    x = np.array(D['sweep_N']['values'], dtype=float)
    ref = x ** 3
    yr = np.array([row['comp_time'] for row in D['sweep_N']['RIPPLE']])
    ref = ref / ref[-1] * yr[-1]
    ax.plot(x, ref, color='gray', ls=(0, (1, 1)), lw=1.5,
            label=r'$\mathcal{O}(N\max(M,N)^2)$ ref.')
    ax.set_xlabel('Number of UAVs, $N$')
    ax.set_ylabel('Per-round computation time (ms)')
    ax.set_title('Computation time vs swarm size')
    ax.legend()
    finish(fig, '06_comptime_vs_N.png')


# ======================================================================
# GROUP B -- performance vs number of coalitions M
# ======================================================================
def fig_welfare_M():
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for s in ORDER:
        x, y, e = sweep_xy('sweep_M', s, 'welfare_norm')
        ax.errorbar(x, y, yerr=e, label=s, capsize=3, **STY[s])
    ax.set_xlabel('Number of coalitions, $M$')
    ax.set_ylabel('Normalised system welfare  $\\overline{W}$')
    ax.set_title('System welfare vs number of coalitions')
    ax.legend()
    finish(fig, '07_welfare_vs_M.png')


def fig_cascade_M():
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for s in ORDER:
        x, y, e = sweep_xy('sweep_M', s, 'cascades')
        ax.plot(x, y, label=s, **STY[s])
    ax.set_xlabel('Number of coalitions, $M$')
    ax.set_ylabel('Cascade events per trial')
    ax.set_title('Cascade events vs number of coalitions')
    ax.legend()
    finish(fig, '08_cascades_vs_M.png')


def fig_mig_M():
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for s in ORDER:
        x, y, e = sweep_xy('sweep_M', s, 'mig_cost')
        ax.plot(x, y, label=s, **STY[s])
    ax.set_xlabel('Number of coalitions, $M$')
    ax.set_ylabel('Total migration cost')
    ax.set_title('Migration cost vs number of coalitions')
    ax.set_yscale('log')
    ax.legend()
    finish(fig, '09_migration_vs_M.png')


def fig_restored_M():
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for s in ORDER:
        x, y, e = sweep_xy('sweep_M', s, 'restored_frac')
        ax.errorbar(x, y, yerr=e, label=s, capsize=3, **STY[s])
    ax.set_xlabel('Number of coalitions, $M$')
    ax.set_ylabel('Fraction of degraded coalitions restored')
    ax.set_title('Restoration reliability vs number of coalitions')
    ax.legend()
    finish(fig, '10_restored_vs_M.png')


# ======================================================================
# GROUP C -- headline grouped bars at the default operating point
# ======================================================================
def _bars(key, ylabel, title, name, logy=False, annotate=True):
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    vals = [D['main'][s][key] for s in ORDER]
    errs = [D['main'][s].get(key + '_std', 0) for s in ORDER]
    cols = [STY[s]['color'] for s in ORDER]
    b = ax.bar(ORDER, vals, yerr=errs, capsize=4, color=cols,
               edgecolor='black', linewidth=0.6, width=0.62)
    if logy:
        ax.set_yscale('log')
    if annotate:
        for rect, v in zip(b, vals):
            ax.annotate(f'{v:.3g}', (rect.get_x() + rect.get_width() / 2,
                        rect.get_height()),
                        ha='center', va='bottom', fontsize=9,
                        xytext=(0, 2), textcoords='offset points')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    finish(fig, name)


def fig_bars_welfare():
    _bars('welfare_norm', 'Normalised system welfare',
          'System welfare (default: $N{=}150$, $M{=}26$)', '11_bar_welfare.png')


def fig_bars_cascade():
    _bars('cascades', 'Cascade events per trial',
          'Cascade events by scheme', '12_bar_cascades.png')


def fig_bars_mig():
    _bars('mig_cost', 'Total migration cost',
          'Migration cost by scheme', '13_bar_migration.png', logy=True)


def fig_bars_transfers():
    _bars('transfers', 'Transfers executed per trial',
          'Transfer activity by scheme', '14_bar_transfers.png')


# ======================================================================
# GROUP D -- temporal behaviour over a trial
# ======================================================================
def fig_welfare_trace():
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    for s in ORDER:
        w = np.array(D['traces'][s]['welfare'])
        lo = np.array(D['traces'][s]['welfare_lo'])
        hi = np.array(D['traces'][s]['welfare_hi'])
        t = np.arange(len(w))
        ax.plot(t, w, label=s, **STY[s])
        ax.fill_between(t, lo, hi, color=STY[s]['color'], alpha=0.12)
    ax.set_xlabel('Monitoring round $t$')
    ax.set_ylabel('Normalised system welfare  $\\overline{W}(t)$')
    ax.set_title('Welfare trajectory over a mission')
    ax.legend()
    finish(fig, '15_welfare_trace.png')


def fig_degraded_trace():
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    for s in ORDER:
        dv = np.array(D['traces'][s]['degraded'])
        ax.plot(np.arange(len(dv)), dv, label=s, **STY[s])
    ax.set_xlabel('Monitoring round $t$')
    ax.set_ylabel(r'Degraded coalitions $|\mathcal{D}(t)|$')
    ax.set_title('Degraded-coalition count over a mission')
    ax.legend()
    finish(fig, '16_degraded_trace.png')


# ======================================================================
# GROUP E -- RIPPLE parameter sensitivity
# ======================================================================
def fig_gamma():
    g = D['gamma']
    fig, ax1 = plt.subplots(figsize=(5.6, 4.0))
    ax2 = ax1.twinx()
    l1 = ax1.plot(g['values'], g['welfare'], color='#1b3a6b', marker='o',
                  label='Welfare')
    l2 = ax2.plot(g['values'], g['cascades'], color='#c0392b', marker='s',
                  ls='--', label='Cascades')
    ax1.axvline(2.0, color='gray', ls=':', lw=1.2)
    ax1.set_xlabel('Cascade penalty  $\\gamma$')
    ax1.set_ylabel('Normalised system welfare', color='#1b3a6b')
    ax2.set_ylabel('Cascade events per trial', color='#c0392b')
    ax2.grid(False)
    ax1.set_title('Sensitivity to cascade penalty $\\gamma$')
    ax1.legend(l1 + l2, [x.get_label() for x in l1 + l2], loc='center right')
    finish(fig, '17_sensitivity_gamma.png')


def fig_kappa():
    k = D['kappa']
    fig, ax1 = plt.subplots(figsize=(5.6, 4.0))
    ax2 = ax1.twinx()
    l1 = ax1.plot(k['values'], k['welfare'], color='#1b3a6b', marker='o',
                  label='Welfare')
    l2 = ax1.plot(k['values'], k['restored'], color='#1e8449', marker='^',
                  ls='-.', label='Restored frac.')
    l3 = ax2.plot(k['values'], k['cascades'], color='#c0392b', marker='s',
                  ls='--', label='Cascades')
    ax1.set_xlabel('Donor withdrawal cap  $\\kappa_d$')
    ax1.set_ylabel('Welfare / restored fraction')
    ax2.set_ylabel('Cascade events per trial', color='#c0392b')
    ax2.grid(False)
    ax1.set_xticks(k['values'])
    ax1.set_title('Sensitivity to donor withdrawal cap $\\kappa_d$')
    ax1.legend(l1 + l2 + l3, [x.get_label() for x in l1 + l2 + l3])
    finish(fig, '18_sensitivity_kappa.png')


def fig_epsw():
    e = D['epsw']
    fig, ax1 = plt.subplots(figsize=(5.6, 4.0))
    ax2 = ax1.twinx()
    l1 = ax1.plot(e['values'], e['welfare'], color='#1b3a6b', marker='o',
                  label='Welfare')
    l2 = ax2.plot(e['values'], e['transfers'], color='#d68910', marker='D',
                  ls=':', label='Transfers')
    ax1.set_xscale('log')
    ax1.set_xlabel('Welfare threshold  $\\epsilon_w$')
    ax1.set_ylabel('Normalised system welfare', color='#1b3a6b')
    ax2.set_ylabel('Transfers per trial', color='#d68910')
    ax2.grid(False)
    ax1.set_title('Sensitivity to welfare threshold $\\epsilon_w$')
    ax1.legend(l1 + l2, [x.get_label() for x in l1 + l2])
    finish(fig, '19_sensitivity_epsw.png')


# ======================================================================
# GROUP F -- theory validation / mechanism characterisation
# ======================================================================
def fig_repair_ratio():
    rr = np.array(D['diag']['repair_ratio'])
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.hist(rr, bins=40, color='#1b3a6b', alpha=0.85, edgecolor='white')
    ax.axvline(0.5, color='#c0392b', ls='--', lw=1.8,
               label=r'Lemma 6 bound $=\frac{1}{2}$')
    ax.axvline(rr.mean(), color='black', ls=':', lw=1.5,
               label=f'mean $={rr.mean():.3f}$')
    ax.set_xlabel(r"Repair ratio  $\max(\mathrm{ALG}',v^*)/\mathrm{OPT}_d$")
    ax.set_ylabel('Count')
    ax.set_title('Density-greedy repair quality (Lemma 6)')
    ax.legend()
    finish(fig, '20_repair_ratio_hist.png')


def fig_delta():
    de = np.array(D['diag']['delta'])
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.hist(de, bins=40, color='#1e8449', alpha=0.85, edgecolor='white')
    ax.axvline(de.mean(), color='black', ls=':', lw=1.5,
               label=f'mean $={de.mean():.3f}$')
    ax.set_xlabel(r'Candidate dominance  $\delta = \max_i\Omega_i/\mathrm{OPT}_d$')
    ax.set_ylabel('Count')
    ax.set_title('Empirical distribution of $\\delta$ (Theorem 4)')
    ax.legend()
    finish(fig, '21_delta_hist.png')


def fig_rho():
    rho = np.array(D['diag']['rho'])
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    vals, counts = np.unique(rho, return_counts=True)
    frac = counts / counts.sum() * 100
    b = ax.bar(vals, frac, color='#6b3a8b', edgecolor='black',
               width=0.6, linewidth=0.6)
    for rect, f in zip(b, frac):
        ax.annotate(f'{f:.0f}%', (rect.get_x() + rect.get_width() / 2,
                    rect.get_height()), ha='center', va='bottom',
                    fontsize=9, xytext=(0, 2), textcoords='offset points')
    ax.set_xlabel(r'Binding dimensions  $\varrho$ at first violation')
    ax.set_ylabel('Percentage of repair events')
    ax.set_xticks(vals)
    ax.set_title(r'Distribution of $\varrho$ (Eq. 29)')
    finish(fig, '22_rho_dist.png')


def fig_c7gap():
    t = np.array(D['diag']['c7_true'])
    b = np.array(D['diag']['c7_bound'])
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    ax.hist(b, bins=30, color='#c0392b', alpha=0.6,
            edgecolor='white', label='Lemma 2$-$5 certificate')
    ax.hist(t, bins=30, color='#1b3a6b', alpha=0.6,
            edgecolor='white', label='Exact $H_d^-$')
    ax.axvline(0.55, color='black', ls='--', lw=1.6, label='$H_{\\min}=0.55$')
    ax.set_xlabel('Post-withdrawal donor health')
    ax.set_ylabel('Count')
    ax.set_title('Constraint-7 certificate conservatism')
    ax.legend()
    finish(fig, '23_c7_conservatism.png')


def fig_finalhealth():
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    data = [D['diag']['final_health'][s] for s in ORDER]
    parts = ax.violinplot(data, showmeans=True, showextrema=False)
    for pc, s in zip(parts['bodies'], ORDER):
        pc.set_facecolor(STY[s]['color'])
        pc.set_alpha(0.55)
    parts['cmeans'].set_color('black')
    ax.axhline(0.55, color='gray', ls='--', lw=1.4, label='$H_{\\min}=0.55$')
    ax.set_xticks(range(1, len(ORDER) + 1))
    ax.set_xticklabels(ORDER)
    ax.set_ylabel('Final coalition health $H_k$')
    ax.set_title('Distribution of final coalition health')
    ax.legend()
    finish(fig, '24_final_health_violin.png')


def fig_welfare_improvement():
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    wr = D['main']['RIPPLE']['welfare_norm']
    others = ['IND-GR', 'CFR', 'TAS']
    imp = [100 * (wr - D['main'][s]['welfare_norm']) /
           D['main'][s]['welfare_norm'] for s in others]
    cols = [STY[s]['color'] for s in others]
    b = ax.bar(others, imp, color=cols, edgecolor='black',
               width=0.6, linewidth=0.6)
    for rect, v in zip(b, imp):
        ax.annotate(f'{v:+.1f}%', (rect.get_x() + rect.get_width() / 2,
                    rect.get_height()), ha='center',
                    va='bottom' if v >= 0 else 'top', fontsize=10,
                    xytext=(0, 2 if v >= 0 else -12),
                    textcoords='offset points')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_ylabel('RIPPLE welfare improvement (%)')
    ax.set_title('RIPPLE welfare gain over each benchmark')
    finish(fig, '25_welfare_improvement.png')


def fig_masked():
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    vals = [D['main'][s]['masked'] for s in ORDER]
    cols = [STY[s]['color'] for s in ORDER]
    b = ax.bar(ORDER, vals, color=cols, edgecolor='black',
               width=0.62, linewidth=0.6)
    for rect, v in zip(b, vals):
        ax.annotate(f'{v:.1f}', (rect.get_x() + rect.get_width() / 2,
                    rect.get_height()), ha='center', va='bottom',
                    fontsize=9, xytext=(0, 2), textcoords='offset points')
    ax.set_ylabel('Masked-failure events per trial')
    ax.set_title('Coalitions an arithmetic mean would misjudge healthy')
    finish(fig, '26_masked_geometric.png')


# ======================================================================
# GROUP G -- migration cost per recovery + feature table
# ======================================================================
def fig_mig_per_rec():
    _bars('mig_cost_per_rec', 'Migration cost per restored coalition',
          'Migration efficiency by scheme', '27_bar_mig_per_recovery.png',
          logy=True)


def fig_feature_table():
    feats = [
        'Coalition-level health model',
        'Multi-dimensional (geometric) score',
        'Joint (coordinated) recovery',
        'Donor viability guaranteed',
        'Aggregate donor budget (Constraint 7)',
        'Cascade-risk penalty',
        'Provable donor safety',
        'Bounded-round convergence',
        'Polynomial-time complexity',
    ]
    mat = {
        'RIPPLE': [1, 1, 1, 1, 1, 1, 1, 1, 1],
        'IND-GR': [1, 0, 0, 0, 0, 0, 0, 0, 1],
        'CFR':    [1, 0, 0, 0, 0, 0, 0, 0, 0],
        'TAS':    [1, 0, 0, 0, 0, 0, 0, 0, 1],
    }
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.axis('off')
    ax.grid(False)
    ncol = len(ORDER) + 1
    nrow = len(feats) + 1
    ax.set_xlim(0, ncol)
    ax.set_ylim(0, nrow)
    # header
    ax.text(0.05, nrow - 0.5, 'Feature', va='center', ha='left',
            fontweight='bold', fontsize=11)
    for j, s in enumerate(ORDER):
        ax.add_patch(plt.Rectangle((1 + j, nrow - 1), 1, 1,
                     facecolor=STY[s]['color'], alpha=0.85))
        ax.text(1 + j + 0.5, nrow - 0.5, s, va='center', ha='center',
                color='white', fontweight='bold', fontsize=10.5)
    for i, feat in enumerate(feats):
        row_y = nrow - 2 - i
        if i % 2 == 0:
            ax.add_patch(plt.Rectangle((0, row_y), ncol, 1,
                         facecolor='#f2f2f2', edgecolor='none'))
        ax.text(0.05, row_y + 0.5, feat, va='center', ha='left', fontsize=10)
        for j, s in enumerate(ORDER):
            ok = mat[s][i]
            ax.text(1 + j + 0.5, row_y + 0.5, '\u2713' if ok else '\u2717',
                    va='center', ha='center', fontsize=13,
                    color='#1e8449' if ok else '#c0392b',
                    fontweight='bold')
    for j in range(ncol + 1):
        ax.plot([j, j], [0, nrow], color='#cccccc', lw=0.6)
    for i in range(nrow + 1):
        ax.plot([0, ncol], [i, i], color='#cccccc', lw=0.6)
    ax.set_title('Theoretical comparison of RIPPLE with benchmark schemes',
                 fontsize=12, pad=10)
    finish(fig, '28_feature_table.png')



def fig_radar():
    """Normalised multi-metric profile. Each axis is scaled so 1.0 = best
    scheme on that metric; lower-is-better metrics are inverted. This is a
    multi-criteria summary, not a single-number claim -- TAS's welfare parity
    is visible on the welfare axis."""
    metrics = ['Welfare', 'Cascade\navoidance', 'Migration\nefficiency',
               'Recovery\nspeed', 'Restoration']
    raw = {}
    for s in ORDER:
        m = D['main'][s]
        raw[s] = [m['welfare_norm'],
                  1.0 / (1.0 + m['cascades']),
                  1.0 / (1.0 + m['mig_cost'] / 100.0),
                  1.0 / (1.0 + m['latency']),
                  m['restored_frac']]
    arr = np.array([raw[s] for s in ORDER])
    norm = arr / arr.max(axis=0)
    ang = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
    ang = np.concatenate([ang, ang[:1]])
    fig, ax = plt.subplots(figsize=(5.8, 5.6), subplot_kw=dict(polar=True))
    for j, s in enumerate(ORDER):
        v = np.concatenate([norm[j], norm[j:j + 1, 0][0:1]])
        ax.plot(ang, v, label=s, color=STY[s]['color'],
                marker=STY[s]['marker'], lw=2)
        ax.fill(ang, v, color=STY[s]['color'], alpha=0.08)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(metrics, fontsize=9.5)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['', '0.5', '', '1.0'], fontsize=8)
    ax.set_title('Normalised multi-metric profile\n(1.0 = best scheme per axis)',
                 fontsize=12, pad=18)
    ax.legend(loc='upper right', bbox_to_anchor=(1.28, 1.12))
    ax.grid(True, alpha=0.4)
    finish(fig, '29_radar_profile.png')


def fig_welfare_cost_pareto():
    """Joint welfare / migration-cost view. The desirable corner is high
    welfare at low cost (upper-left)."""
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    for s in ORDER:
        m = D['main'][s]
        ax.scatter(m['mig_cost'], m['welfare_norm'], s=180,
                   color=STY[s]['color'], edgecolor='black', linewidth=0.8,
                   marker=STY[s]['marker'], zorder=3, label=s)
        ax.annotate(s, (m['mig_cost'], m['welfare_norm']),
                    xytext=(8, 6), textcoords='offset points',
                    fontsize=10, fontweight='bold', color=STY[s]['color'])
    ax.set_xlabel('Total migration cost  (lower is better \u2190)')
    ax.set_ylabel('Normalised system welfare  (higher is better \u2191)')
    ax.set_title('Welfare vs migration cost')
    ax.annotate('preferred', (0.06, 0.93), xycoords='axes fraction',
                fontsize=10, style='italic', color='gray')
    ax.annotate('', xy=(0.02, 0.99), xytext=(0.14, 0.90),
                xycoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))
    finish(fig, '30_welfare_cost_pareto.png')


if __name__ == '__main__':
    # The curated set of 20 figures. Every figure referenced by Section V-C of
    # the paper is included; redundant sweeps (migration/restored vs M, latency
    # vs N) and uninformative sweeps (kappa_d, which is flat) are omitted.
    FIGURES = [
        # -- headline summary ------------------------------------------------
        fig_radar,                 # 29 multi-metric profile
        fig_welfare_cost_pareto,   # 30 welfare vs migration cost
        # -- operating-point bars --------------------------------------------
        fig_bars_welfare,          # 11
        fig_bars_cascade,          # 12
        fig_bars_mig,              # 13
        fig_mig_per_rec,           # 27
        # -- temporal behaviour ----------------------------------------------
        fig_welfare_trace,         # 15
        fig_degraded_trace,        # 16
        # -- scaling in N ----------------------------------------------------
        fig_welfare_N,             # 01
        fig_cascade_N,             # 02
        fig_restored_N,            # 05
        fig_comptime_N,            # 06
        # -- scaling in M ----------------------------------------------------
        fig_welfare_M,             # 07
        # -- parameter sensitivity -------------------------------------------
        fig_gamma,                 # 17
        fig_epsw,                  # 19
        # -- theory / mechanism validation -----------------------------------
        fig_repair_ratio,          # 20 Lemma 6
        fig_delta,                 # 21 Theorem 4
        fig_rho,                   # 22 Eq. (29)
        fig_c7gap,                 # 23 Constraint-7 conservatism
        fig_masked,                # 26 geometric vs arithmetic
    ]
    for fn in FIGURES:
        fn()
    print(f'\nTOTAL: {len(saved)} figures in {OUTDIR}')
