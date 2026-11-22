"""
Composite figure generator for the RIPPLE paper.

Renders the same 20 panels as `make_plots.py`, but grouped into six
multi-panel figures sized for IEEE two-column layout, in the style of the
TOFU reference paper (Figs. 2, 5 and 8 there are (a)/(b)/(c) panel groups).

Output:
  fig1_summary.png       1x2  radar profile, welfare-cost plane
  fig2_comparison.png    2x2  welfare, cascades, migration cost, cost/recovery
  fig3_temporal.png      1x2  welfare trace, degraded-count trace
  fig4_scaling.png       2x3  welfare/cascades/restored/time vs N, vs M
  fig5_sensitivity.png   1x2  gamma, epsilon_w
  fig6_theory.png        2x3  Lemma 6, delta, rho, Constraint-7, masked, health

Run:
  python3 make_figures_composite.py
  FIGDIR=/path/to/figures python3 make_figures_composite.py
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D

# Panels are small, so the type is scaled down relative to the single-plot set.
rcParams.update({
    'font.size': 8.5, 'axes.titlesize': 9, 'axes.labelsize': 8.5,
    'legend.fontsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'figure.dpi': 150, 'savefig.dpi': 400, 'axes.grid': True,
    'grid.alpha': 0.30, 'grid.linewidth': 0.5, 'axes.axisbelow': True,
    'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans',
    'lines.linewidth': 1.5, 'lines.markersize': 4,
})

_HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.environ.get('FIGDIR', os.path.join(_HERE, 'figures_composite'))
os.makedirs(OUTDIR, exist_ok=True)
D = json.load(open(os.path.join(_HERE, 'results.json')))

STY = {
    'RIPPLE': dict(color='#1b3a6b', marker='o', ls='-'),
    'IND-GR': dict(color='#c0392b', marker='s', ls='--'),
    'CFR':    dict(color='#1e8449', marker='^', ls='-.'),
    'TAS':    dict(color='#d68910', marker='D', ls=':'),
}
ORDER = ['RIPPLE', 'IND-GR', 'CFR', 'TAS']
IEEE_W = 7.16          # full two-column text width, inches
saved = []


# ----------------------------------------------------------------------
# (a)/(b)/(c) tagging -- placed from the *measured* bottom of each axes
# (title/ticks/xlabel included) instead of a guessed axes-fraction offset,
# so it never collides with tick labels, rotated ticks, or twin-axis text.
# ----------------------------------------------------------------------
def add_tags(fig, ax_letter_pairs, pad=0.012, fontsize=9):
    """ax_letter_pairs: list of (ax, 'a'), (ax, 'b'), ... """
    fig.canvas.draw()  # force a layout pass so bounding boxes are accurate
    renderer = fig.canvas.get_renderer()
    for ax, letter in ax_letter_pairs:
        bbox = ax.get_tightbbox(renderer)
        bbox_fig = bbox.transformed(fig.transFigure.inverted())
        xc = 0.5 * (bbox_fig.x0 + bbox_fig.x1)
        y0 = bbox_fig.y0 - pad
        fig.text(xc, y0, f'({letter})', ha='center', va='top',
                  fontsize=fontsize)


def scheme_legend(fig, ncol=4, y=1.005):
    handles = [Line2D([0], [0], color=STY[s]['color'], marker=STY[s]['marker'],
                      ls=STY[s]['ls'], lw=1.6, ms=5, label=s) for s in ORDER]
    fig.legend(handles=handles, loc='upper center', ncol=ncol,
               bbox_to_anchor=(0.5, y), frameon=False, fontsize=8.5)


def finish(fig, name, rect=None, hspace=None, tag_pairs=None):
    """Lay out, reserve room for tags, add tags from real geometry, save."""
    fig.tight_layout(rect=rect or (0, 0, 1, 1))
    # tight_layout's own hspace is sized for xlabels only, not for a tag
    # sitting below them too -- widen it before measuring/placing tags.
    if hspace is not None:
        fig.subplots_adjust(hspace=hspace)
    if tag_pairs:
        add_tags(fig, tag_pairs)
    p = os.path.join(OUTDIR, name)
    fig.savefig(p, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    saved.append(p)
    print('saved', name)


# ======================================================================
# panel primitives
# ======================================================================

def min_span(ax, span=0.05):
    """Enforce a minimum y-range so that a very small effect is not visually
    magnified by autoscaling into a dramatic slope."""
    lo, hi = ax.get_ylim()
    if hi - lo < span:
        mid = 0.5 * (lo + hi)
        ax.set_ylim(mid - span / 2, mid + span / 2)


def p_bar(ax, key, ylabel, logy=False, fmt='{:.3g}'):
    vals = [D['main'][s][key] for s in ORDER]
    errs = [D['main'][s].get(key + '_std', 0) for s in ORDER]
    cols = [STY[s]['color'] for s in ORDER]
    b = ax.bar(ORDER, vals, yerr=errs, capsize=2.5, color=cols,
               edgecolor='black', linewidth=0.5, width=0.62)
    if logy:
        ax.set_yscale('log')
    for rect, v in zip(b, vals):
        ax.annotate(fmt.format(v),
                    (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    ha='center', va='bottom', fontsize=7,
                    xytext=(0, 1.5), textcoords='offset points')
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='x', labelrotation=18)
    if not logy:
        ax.set_ylim(0, max(v + e for v, e in zip(vals, errs)) * 1.22)


def p_sweep(ax, sweep, key, xlabel, ylabel, logy=False, err=False):
    xs = D[sweep]['values']
    for s in ORDER:
        ys = [r[key] for r in D[sweep][s]]
        if err:
            es = [r.get(key + '_std', 0) for r in D[sweep][s]]
            ax.errorbar(xs, ys, yerr=es, capsize=2, **STY[s])
        else:
            ax.plot(xs, ys, **STY[s])
    if logy:
        ax.set_yscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def p_welfare_trace(ax):
    for s in ORDER:
        w = np.array(D['traces'][s]['welfare'])
        lo = np.array(D['traces'][s]['welfare_lo'])
        hi = np.array(D['traces'][s]['welfare_hi'])
        t = np.arange(len(w))
        ax.plot(t, w, **STY[s])
        ax.fill_between(t, lo, hi, color=STY[s]['color'], alpha=0.10)
    ax.set_xlabel('Monitoring round $t$')
    ax.set_ylabel('Normalised welfare $\\overline{W}(t)$')


def p_degraded_trace(ax):
    for s in ORDER:
        dv = np.array(D['traces'][s]['degraded'])
        ax.plot(np.arange(len(dv)), dv, **STY[s])
    ax.set_xlabel('Monitoring round $t$')
    ax.set_ylabel(r'Degraded coalitions $|\mathcal{D}(t)|$')


def p_comptime(ax):
    xs = np.array(D['sweep_N']['values'], dtype=float)
    for s in ORDER:
        ax.plot(xs, [r['comp_time'] for r in D['sweep_N'][s]], **STY[s])
    ref = xs ** 3
    yr = np.array([r['comp_time'] for r in D['sweep_N']['RIPPLE']])
    ref = ref / ref[-1] * yr[-1]
    ax.plot(xs, ref, color='gray', ls=(0, (1, 1)), lw=1.2)
    ax.set_xlabel('Number of UAVs, $N$')
    ax.set_ylabel('Per-round time (ms)')
    ax.annotate(r'$\mathcal{O}(N\max(M,N)^2)$', (0.05, 0.86),
                xycoords='axes fraction', fontsize=7, color='gray')


def p_gamma(ax):
    g = D['gamma']
    ax2 = ax.twinx()
    ax.plot(g['values'], g['welfare'], color='#1b3a6b', marker='o',
            label='Welfare')
    ax2.plot(g['values'], g['cascades'], color='#c0392b', marker='s',
             ls='--', label='Cascades')
    ax.axvline(2.0, color='gray', ls=':', lw=1.0)
    ax.set_xlabel('Cascade penalty  $\\gamma$')
    ax.set_ylabel('Normalised welfare', color='#1b3a6b')
    ax2.set_ylabel('Cascades / trial', color='#c0392b')
    ax2.set_ylim(-0.05, 1.0)
    ax2.grid(False)
    min_span(ax, 0.05)
    ax.legend(loc='center right', fontsize=7.5)
    ax2.legend(loc='upper right', fontsize=7.5)


def p_epsw(ax):
    e = D['epsw']
    ax2 = ax.twinx()
    ax.plot(e['values'], e['welfare'], color='#1b3a6b', marker='o',
            label='Welfare')
    ax2.plot(e['values'], e['transfers'], color='#d68910', marker='D',
             ls=':', label='Transfers')
    ax.set_xscale('log')
    ax.set_xlabel('Welfare threshold  $\\epsilon_w$')
    ax.set_ylabel('Normalised welfare', color='#1b3a6b')
    ax2.set_ylabel('Transfers / trial', color='#d68910')
    ax2.grid(False)
    min_span(ax, 0.05)
    ax.legend(loc='lower left', fontsize=7.5)
    ax2.legend(loc='upper right', fontsize=7.5)


def p_repair(ax):
    rr = np.array(D['diag']['repair_ratio'])
    ax.hist(rr, bins=36, color='#1b3a6b', alpha=0.85, edgecolor='white',
            linewidth=0.3)
    ax.axvline(0.5, color='#c0392b', ls='--', lw=1.4)
    ax.axvline(rr.mean(), color='black', ls=':', lw=1.2)
    ax.set_xlabel(r"$\max(\mathrm{ALG}',v^*)/\mathrm{OPT}_d$")
    ax.set_ylabel('Count')
    ax.annotate('Lemma 6\nbound $=1/2$', (0.505, 0.72), xycoords='axes fraction',
                fontsize=7, color='#c0392b')
    ax.annotate(f'mean $={rr.mean():.3f}$', (0.05, 0.90),
                xycoords='axes fraction', fontsize=7)


def p_delta(ax):
    de = np.array(D['diag']['delta'])
    ax.hist(de, bins=36, color='#1e8449', alpha=0.85, edgecolor='white',
            linewidth=0.3)
    ax.axvline(de.mean(), color='black', ls=':', lw=1.2)
    ax.set_xlabel(r'$\delta = \max_i\Omega_i/\mathrm{OPT}_d$')
    ax.set_ylabel('Count')
    ax.annotate(f'mean $={de.mean():.3f}$', (0.05, 0.90),
                xycoords='axes fraction', fontsize=7)


def p_rho(ax):
    rho = np.array(D['diag']['rho'])
    vals, counts = np.unique(rho, return_counts=True)
    frac = counts / counts.sum() * 100
    b = ax.bar(vals, frac, color='#6b3a8b', edgecolor='black', width=0.55,
               linewidth=0.5)
    for rect, f in zip(b, frac):
        ax.annotate(f'{f:.0f}%',
                    (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    ha='center', va='bottom', fontsize=7.5,
                    xytext=(0, 1.5), textcoords='offset points')
    ax.set_xlabel(r'Binding dimensions  $\varrho$')
    ax.set_ylabel('Repair events (%)')
    ax.set_xticks(vals)
    ax.set_ylim(0, 108)


def p_c7(ax):
    t = np.array(D['diag']['c7_true'])
    b = np.array(D['diag']['c7_bound'])
    ax.hist(b, bins=24, color='#c0392b', alpha=0.6, edgecolor='white',
            linewidth=0.3, label='Lemma 2$-$5 cert.')
    ax.hist(t, bins=24, color='#1b3a6b', alpha=0.6, edgecolor='white',
            linewidth=0.3, label='Exact $H_d^-$')
    ax.axvline(0.55, color='black', ls='--', lw=1.3)
    ax.set_xlabel('Post-withdrawal donor health')
    ax.set_ylabel('Count')
    ax.annotate('$H_{\\min}$', (0.56, 0.86), xycoords='axes fraction',
                fontsize=7.5)
    ax.legend(fontsize=7, loc='upper left')


def p_masked(ax):
    vals = [D['main'][s]['masked'] for s in ORDER]
    cols = [STY[s]['color'] for s in ORDER]
    b = ax.bar(ORDER, vals, color=cols, edgecolor='black', width=0.62,
               linewidth=0.5)
    for rect, v in zip(b, vals):
        ax.annotate(f'{v:.1f}',
                    (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    ha='center', va='bottom', fontsize=7,
                    xytext=(0, 1.5), textcoords='offset points')
    ax.set_ylabel('Masked failures / trial')
    ax.tick_params(axis='x', labelrotation=18)
    ax.set_ylim(0, max(vals) * 1.22)


def p_finalhealth(ax):
    data = [D['diag']['final_health'][s] for s in ORDER]
    parts = ax.violinplot(data, showmeans=True, showextrema=False)
    for pc, s in zip(parts['bodies'], ORDER):
        pc.set_facecolor(STY[s]['color'])
        pc.set_alpha(0.55)
    parts['cmeans'].set_color('black')
    ax.axhline(0.55, color='gray', ls='--', lw=1.2)
    ax.set_xticks(range(1, len(ORDER) + 1))
    ax.set_xticklabels(ORDER, rotation=18)
    ax.set_ylabel('Final health $H_k$')
    ax.annotate('$H_{\\min}$', (0.02, 0.56), xycoords='axes fraction',
                fontsize=7.5, color='gray')


def p_radar(ax):
    metrics = ['Welfare', 'Cascade\navoid.', 'Migration\neffic.',
               'Recovery\nspeed', 'Restor-\nation']
    raw = {}
    for s in ORDER:
        m = D['main'][s]
        raw[s] = [m['welfare_norm'], 1.0 / (1.0 + m['cascades']),
                  1.0 / (1.0 + m['mig_cost'] / 100.0),
                  1.0 / (1.0 + m['latency']), m['restored_frac']]
    arr = np.array([raw[s] for s in ORDER])
    norm = arr / arr.max(axis=0)
    ang = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
    ang = np.concatenate([ang, ang[:1]])
    for j, s in enumerate(ORDER):
        v = np.concatenate([norm[j], norm[j][:1]])
        ax.plot(ang, v, color=STY[s]['color'], marker=STY[s]['marker'],
                lw=1.5, ms=3.5)
        ax.fill(ang, v, color=STY[s]['color'], alpha=0.07)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(metrics, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.5, 1.0])
    ax.set_yticklabels(['0.5', '1.0'], fontsize=6.5)
    ax.grid(True, alpha=0.4)


def p_pareto(ax):
    for s in ORDER:
        m = D['main'][s]
        ax.scatter(m['mig_cost'], m['welfare_norm'], s=90,
                   color=STY[s]['color'], edgecolor='black', linewidth=0.6,
                   marker=STY[s]['marker'], zorder=3)
        ax.annotate(s, (m['mig_cost'], m['welfare_norm']),
                    xytext=(6, 4), textcoords='offset points',
                    fontsize=7.5, fontweight='bold', color=STY[s]['color'])
    ax.set_xlabel('Total migration cost  (lower better $\\leftarrow$)')
    ax.set_ylabel('Normalised welfare  (higher better $\\uparrow$)')
    ax.annotate('preferred', (0.05, 0.90), xycoords='axes fraction',
                fontsize=7.5, style='italic', color='gray')
    ax.annotate('', xy=(0.02, 0.98), xytext=(0.13, 0.88),
                xycoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.0))
    ax.set_xlim(0, 440)


# ======================================================================
# composite figures -- four dense panel groups, sized for IEEE two-column
# ======================================================================
def fig1_performance():
    """2x3: operating-point bars + temporal behaviour."""
    fig, axes = plt.subplots(2, 3, figsize=(IEEE_W, 3.9))
    p_bar(axes[0, 0], 'welfare_norm', 'Normalised welfare')
    p_bar(axes[0, 1], 'cascades', 'Cascades / trial', fmt='{:.2f}')
    p_bar(axes[0, 2], 'mig_cost', 'Migration cost', logy=True, fmt='{:.0f}')
    p_welfare_trace(axes[1, 0])
    p_degraded_trace(axes[1, 1])
    p_pareto(axes[1, 2])
    axes[1, 2].set_xlabel('Migration cost')
    axes[1, 2].set_ylabel('Normalised welfare')
    tag_pairs = list(zip(axes.ravel(), 'abcdef'))
    scheme_legend(fig, y=1.055)
    finish(fig, 'fig1_performance.png', rect=(0, 0.02, 1, 0.94),
           hspace=0.75, tag_pairs=tag_pairs)


def fig2_scaling():
    """2x3: scaling in N and M."""
    fig, axes = plt.subplots(2, 3, figsize=(IEEE_W, 3.9))
    p_sweep(axes[0, 0], 'sweep_N', 'welfare_norm', '$N$',
            'Normalised welfare', err=True)
    p_sweep(axes[0, 1], 'sweep_N', 'cascades', '$N$', 'Cascades / trial')
    p_sweep(axes[0, 2], 'sweep_N', 'restored_frac', '$N$',
            'Restored fraction')
    p_comptime(axes[1, 0])
    axes[1, 0].set_xlabel('$N$')
    p_sweep(axes[1, 1], 'sweep_M', 'welfare_norm', '$M$',
            'Normalised welfare', err=True)
    p_sweep(axes[1, 2], 'sweep_M', 'cascades', '$M$', 'Cascades / trial')
    tag_pairs = list(zip(axes.ravel(), 'abcdef'))
    scheme_legend(fig, y=1.055)
    finish(fig, 'fig2_scaling.png', rect=(0, 0.02, 1, 0.94),
           hspace=0.75, tag_pairs=tag_pairs)


def fig3_theory():
    """2x3: repair guarantees, Constraint-7 certificate, health model."""
    fig, axes = plt.subplots(2, 3, figsize=(IEEE_W, 3.9))
    p_repair(axes[0, 0])
    p_delta(axes[0, 1])
    p_rho(axes[0, 2])
    p_c7(axes[1, 0])
    p_masked(axes[1, 1])
    p_finalhealth(axes[1, 2])
    tag_pairs = list(zip(axes.ravel(), 'abcdef'))
    finish(fig, 'fig3_theory.png', hspace=0.75, tag_pairs=tag_pairs)


def fig4_design():
    """1x3: multi-metric profile, cost efficiency, epsilon_w sensitivity.
    The gamma-sensitivity panel is dropped here -- it duplicates
    fig3_theory's (f) panel (see p_gamma there), so keeping both wastes a
    panel and a paragraph of prose describing the same sweep twice."""
    fig = plt.figure(figsize=(IEEE_W, 2.3))
    ax1 = fig.add_subplot(1, 3, 1, projection='polar')
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    p_radar(ax1)
    p_bar(ax2, 'mig_cost_per_rec', 'Cost / restored coalition', logy=True,
          fmt='{:.1f}')
    p_epsw(ax3)
    tag_pairs = [(ax1, 'a'), (ax2, 'b'), (ax3, 'c')]
    scheme_legend(fig, y=1.10)
    finish(fig, 'fig4_design.png', rect=(0, 0.02, 1, 0.88),
           hspace=0.55, tag_pairs=tag_pairs)


if __name__ == '__main__':
    fig1_performance()
    fig2_scaling()
    fig3_theory()
    fig4_design()
    print(f'\nTOTAL: {len(saved)} composite figures ({OUTDIR})')