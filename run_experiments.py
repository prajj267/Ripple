"""
Run every experiment needed for the RIPPLE figures and cache the results to
results.json.  Written to save after each block so it can be run in stages or
resumed, and to keep each block within a single-core time budget.
"""
import json
import os
import time
import numpy as np
from ripple_sim import DEFAULTS, run_trial

SCHEMES = ['RIPPLE', 'IND-GR', 'CFR', 'TAS']
OUT = 'results.json'

# High-contention regime: donors are scarce and faults frequent, so multiple
# recipients draw from the same donor in one round.  This exercises the
# Constraint-7 repair machinery (rho, delta, Lemma-6 repair ratio, the
# certificate gap) and the cascade penalty, which almost never activate in the
# balanced performance regime.  Used ONLY for the mechanism-characterisation
# figures, not the performance comparison.
CONTENTION = dict(DEFAULTS, stress_frac=0.85, idle_frac=0.015,
                  member_fault_rate=0.14)


def load():
    if os.path.exists(OUT):
        with open(OUT) as f:
            return json.load(f)
    return {}


def save(d):
    with open(OUT, 'w') as f:
        json.dump(d, f)


def agg(rs, keys):
    out = {}
    for k in keys:
        v = [r[k] for r in rs]
        out[k] = float(np.mean(v))
        out[k + '_std'] = float(np.std(v))
    return out


MET = ['welfare_norm', 'welfare_final', 'cascades', 'cascade_rate', 'latency',
       'restored_frac', 'transfers', 'mig_cost', 'mig_cost_per_rec',
       'comp_time', 'comp_time_tot', 'pool', 'relaxations', 'masked']


def block_main(D, ntr=25):
    if 'main' in D:
        return
    t0 = time.time()
    out = {}
    for s in SCHEMES:
        rs = [run_trial(s, dict(DEFAULTS), sd) for sd in range(ntr)]
        out[s] = agg(rs, MET)
    D['main'] = out
    save(D)
    print(f'[main] {time.time()-t0:.0f}s')


def block_sweep(D, name, param, values, ntr=6, couple_M=False):
    if D.get(name, {}).get('_done'):
        return
    t0 = time.time()
    out = D.get(name, {'values': list(values)})
    for s in SCHEMES:
        if s in out:                       # already computed this scheme
            continue
        rows = []
        for v in values:
            prm = dict(DEFAULTS)
            prm[param] = v
            if couple_M:
                prm['M'] = int(min(30, max(5, round(v / 6))))
            rs = [run_trial(s, prm, sd) for sd in range(ntr)]
            rows.append(agg(rs, MET))
        out[s] = rows
        D[name] = out
        save(D)                            # checkpoint after each scheme
        print(f'[{name}:{s}] {time.time()-t0:.0f}s')
    out['_done'] = True
    D[name] = out
    save(D)
    print(f'[{name}] done {time.time()-t0:.0f}s')


def block_traces(D, ntr=12):
    if 'traces' in D:
        return
    t0 = time.time()
    out = {}
    for s in SCHEMES:
        wt, dt = [], []
        for sd in range(ntr):
            r = run_trial(s, dict(DEFAULTS), sd, trace=True)
            wt.append(r['welfare_trace'])
            dt.append(r['degraded_trace'])
        M = DEFAULTS['M']
        out[s] = dict(
            welfare=(np.array(wt).mean(0) / M).tolist(),
            welfare_lo=(np.percentile(np.array(wt) / M, 25, axis=0)).tolist(),
            welfare_hi=(np.percentile(np.array(wt) / M, 75, axis=0)).tolist(),
            degraded=np.array(dt).mean(0).tolist(),
        )
    D['traces'] = out
    save(D)
    print(f'[traces] {time.time()-t0:.0f}s')


def block_diagnostics(D, ntr=20):
    """Collect RIPPLE internal distributions: rho, delta, repair ratio,
    Constraint-7 conservatism gap, and per-scheme final health."""
    if 'diag' in D:
        return
    t0 = time.time()
    rho, delta, rr, c7t, c7b = [], [], [], [], []
    for sd in range(ntr):
        r = run_trial('RIPPLE', dict(CONTENTION), sd)
        rho += r['rho']
        delta += r['delta']
        rr += r['repair_ratio']
        for tr, bd in r['c7gap']:
            c7t.append(tr)
            c7b.append(bd)
    finalH = {}
    for s in SCHEMES:
        hs = []
        for sd in range(ntr):
            hs += run_trial(s, dict(DEFAULTS), sd)['final_health']
        finalH[s] = hs
    D['diag'] = dict(rho=rho, delta=delta, repair_ratio=rr,
                     c7_true=c7t, c7_bound=c7b, final_health=finalH)
    save(D)
    print(f'[diag] {time.time()-t0:.0f}s')


def block_gamma(D, ntr=10):
    if 'gamma' in D:
        return
    t0 = time.time()
    vals = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    out = {'values': vals, 'welfare': [], 'cascades': [], 'restored': []}
    for g in vals:
        rs = [run_trial('RIPPLE', dict(CONTENTION, gamma=g), sd)
              for sd in range(ntr)]
        out['welfare'].append(float(np.mean([r['welfare_norm'] for r in rs])))
        out['cascades'].append(float(np.mean([r['cascades'] for r in rs])))
        out['restored'].append(
            float(np.mean([r['restored_frac'] for r in rs])))
    D['gamma'] = out
    save(D)
    print(f'[gamma] {time.time()-t0:.0f}s')


def block_kappa(D, ntr=10):
    if 'kappa' in D:
        return
    t0 = time.time()
    vals = [1, 2, 3, 4]
    out = {'values': vals, 'welfare': [], 'cascades': [], 'restored': []}
    for kap in vals:
        rs = [run_trial('RIPPLE', dict(DEFAULTS, kappa_d=kap), sd)
              for sd in range(ntr)]
        out['welfare'].append(float(np.mean([r['welfare_norm'] for r in rs])))
        out['cascades'].append(float(np.mean([r['cascades'] for r in rs])))
        out['restored'].append(
            float(np.mean([r['restored_frac'] for r in rs])))
    D['kappa'] = out
    save(D)
    print(f'[kappa] {time.time()-t0:.0f}s')


def block_epsw(D, ntr=10):
    if 'epsw' in D:
        return
    t0 = time.time()
    vals = [0.005, 0.01, 0.02, 0.05, 0.1]
    out = {'values': vals, 'welfare': [], 'transfers': [], 'Tmax': [],
           'active': []}
    for e in vals:
        rs = [run_trial('RIPPLE', dict(DEFAULTS, eps_w=e), sd)
              for sd in range(ntr)]
        out['welfare'].append(float(np.mean([r['welfare_norm'] for r in rs])))
        out['transfers'].append(float(np.mean([r['transfers'] for r in rs])))
        out['Tmax'].append(float(np.mean([r['Tmax_bound'] for r in rs])))
        out['active'].append(float(np.mean([r['active_rounds'] for r in rs])))
    D['epsw'] = out
    save(D)
    print(f'[epsw] {time.time()-t0:.0f}s')


if __name__ == '__main__':
    import sys
    D = load()
    stage = sys.argv[1] if len(sys.argv) > 1 else 'all'

    if stage in ('all', 'main'):
        block_main(D)
    if stage in ('all', 'N'):
        block_sweep(D, 'sweep_N', 'N', [50, 100, 150, 200, 250, 300], couple_M=True)
    if stage in ('all', 'M'):
        block_sweep(D, 'sweep_M', 'M', [6, 12, 18, 24, 30])
    if stage in ('all', 'traces'):
        block_traces(D)
    if stage in ('all', 'diag'):
        block_diagnostics(D)
    if stage in ('all', 'gamma'):
        block_gamma(D)
    if stage in ('all', 'kappa'):
        block_kappa(D)
    if stage in ('all', 'epsw'):
        block_epsw(D)
    print('done', stage)
