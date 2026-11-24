#!/usr/bin/env python3
"""
cavity_angle.py -- in the cavity test (the gel alone, a rigid sphere growing in the organoid's hole; gel
model of experiments/ecm_connectivity/cavity_expansion.py: the engine's stretch, bend and contact terms),
which fiber angles carry the gel's push-back, and does cutting fibers of one angle matter more than cutting
the same number of another?
psi = angle between a fiber segment and the sphere's normal at a contact point: |cos psi| = 1 end-on,
0 lying flat on the sphere.

static    quasi-static expansion R = 3.0 .. 6.6 (step 0.6, 2 sub-steps; clamped boundary R_far = 20, k = 500:
          the earlier test's settings with a coarser R step). At every R the contact force is split into 5 bins
          of |cos psi|, with the contact length and the length of fiber within 1 of the sphere, and the mean
          strain of segments 0-2 outside the sphere by their angle to the radial direction.
cut       the same expansion after cutting K segments near the hole (closest approach to the centre < 7):
          K end-on ones (|cos| > 0.7 at their closest point) or K flat ones (|cos| < 0.3), K = half the smaller
          group, picked at random (seed 1). A cut segment goes with the bends that use it.
dynamic   the engine's dynamics on the periodic gel, as cavity_expansion.run_dynamic (all nodes free, sphere
          growing at 0.033 from R = 2.6 to 4.9 then held 5 time units, contact k = 50), split by angle every
          time unit.
Rows go to results/cavity_angle.csv after every case.
    python3 experiments/fiber_angle/cavity_angle.py                  # all cases, ~30 min
    python3 experiments/fiber_angle/cavity_angle.py --summary-only   # reprint the summary from the csv
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "experiments", "ecm_connectivity"))
import cavity_expansion as ce  # noqa: E402

INP = os.path.join(ROOT, "experiments", "ecm_connectivity", "inputs")
CSV = os.path.join(HERE, "results", "cavity_angle.csv")
EDGES = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0 + 1e-9])


def split(gel, Y, R, case, t=np.nan):
    """Contact force on the sphere split by |cos psi| of the touching segment (5 rows, one per bin)."""
    gel.set_samples(np.arange(len(gel.bonds)))
    si, sj = gel.bonds[gel.s_seg, 0], gel.bonds[gel.s_seg, 1]
    d = gel.wrap(Y[sj] - Y[si])
    x = Y[si] + gel.s_w[:, None] * d
    r = np.linalg.norm(x, axis=1)
    c = np.abs(np.einsum("ij,ij->i", d, x)) / (np.linalg.norm(d, axis=1) * r + 1e-30)
    b = np.clip(np.digitize(c, EDGES) - 1, 0, 4)
    pen = R + ce.RANGE - r
    on = pen > 0
    f = np.where(on, gel.kw * gel.s_ds * pen, 0.0)
    near = r < R + 1.0
    F_tot = float(f.sum())
    i, j = gel.bonds[:, 0], gel.bonds[:, 1]
    dd = gel.wrap(Y[j] - Y[i])
    m = Y[i] + 0.5 * dd
    rm, l = np.linalg.norm(m, axis=1), np.linalg.norm(dd, axis=1)
    bm = np.clip(np.digitize(np.abs(np.einsum("ij,ij->i", dd, m)) / (l * rm + 1e-30), EDGES) - 1, 0, 4)
    shell = (rm > R) & (rm < R + 2.0)
    strain = (l - ce.L0) / ce.L0
    rows = []
    for k in range(5):
        sel, ssel = b == k, shell & (bm == k)
        n_on = int((on & sel).sum())
        rows.append(dict(case=case, t=t, R=round(R, 4), cav_strain=(R - ce.R_HOLE) / ce.R_HOLE, bin=k,
                         P_total=F_tot / (4 * np.pi * R ** 2), F=float(f[sel].sum()),
                         F_share=float(f[sel].sum()) / F_tot if F_tot > 0 else np.nan,
                         contact_share=n_on / max(1, int(on.sum())), n_contact=n_on,
                         near_share=int((near & sel).sum()) / max(1, int(near.sum())),
                         F_per_len=float(f[sel].sum()) / (n_on * gel.s_ds) if n_on else np.nan,
                         shell_strain=float(strain[ssel].mean()) if ssel.any() else np.nan,
                         shell_n=int(ssel.sum())))
    return rows


def cut(gel, which, frac=0.5, seed=1, r_cut=7.0):
    """Remove K near-hole segments of one angle class (and the bends that use them)."""
    Y = gel.X0
    i, j = gel.bonds[:, 0], gel.bonds[:, 1]
    d = gel.wrap(Y[j] - Y[i])
    s = np.clip(-np.einsum("ij,ij->i", Y[i], d) / np.einsum("ij,ij->i", d, d), 0.0, 1.0)
    P = Y[i] + s[:, None] * d
    dist = np.linalg.norm(P, axis=1)
    c = np.abs(np.einsum("ij,ij->i", d, P)) / (np.linalg.norm(d, axis=1) * dist + 1e-30)
    endon = np.flatnonzero((dist < r_cut) & (c > 0.7))
    flat = np.flatnonzero((dist < r_cut) & (c < 0.3))
    K = int(frac * min(len(endon), len(flat)))
    gone = np.random.default_rng(seed).choice(endon if which == "endon" else flat, K, replace=False)
    bad = {(min(a, b), max(a, b)) for a, b in gel.bonds[gone].tolist()}
    keep = np.ones(len(gel.bonds), bool)
    keep[gone] = False
    keep_t = np.array([(min(a, b), max(a, b)) not in bad and (min(b, e), max(b, e)) not in bad
                       for a, b, e in gel.bends.tolist()], bool)
    gel.bonds, gel.bends = gel.bonds[keep], gel.bends[keep_t]
    gel.set_samples(np.arange(len(gel.bonds)))
    return dict(K=K, n_endon=len(endon), n_flat=len(flat), bends_cut=int((~keep_t).sum()))


def run_static(gel, case, radii=np.round(np.arange(3.0, 6.6 + 1e-9, 0.6), 4), substeps=2, maxiter=3000,
               maxiter_sub=600, gtol=1e-9):
    x = gel.X0[gel.fidx].ravel().copy()
    rows, R_prev, tic = [], None, time.time()
    for R in radii:
        steps = [R] if R_prev is None else list(np.linspace(R_prev, R, substeps + 1)[1:])
        for k, Rs in enumerate(steps):
            mi = maxiter if k == len(steps) - 1 else maxiter_sub
            gel.R = Rs
            gel.select_samples(gel.full(x), Rs)
            res = minimize(gel.fun, x, jac=True, method="L-BFGS-B",
                           options=dict(maxiter=mi, maxfun=2 * mi, gtol=gtol, ftol=1e-16, maxcor=30, maxls=50))
            x = res.x
        R_prev = R
        rr = split(gel, gel.full(x), R, case)
        rows += rr
        print(f"[{case}] R={R:.2f} P={rr[0]['P_total']:.3e} F_share by |cos| bin: "
              + " ".join(f"{q['F_share']:.2f}" for q in rr) + f"  {time.time() - tic:.0f}s", flush=True)
    return rows


def run_dyn(path, kb, case, v=0.033, R0=2.6, R1=4.9, hold=5.0, dt=0.005):
    gel = ce.Gel(path, kb, periodic=True, kw=50.0, h=0.5)
    x = gel.X0[gel.fidx].ravel().copy()
    T_grow = (R1 - R0) / v
    n_steps, every = int(round((T_grow + hold) / dt)), int(round(1.0 / dt))
    rows, tic = [], time.time()
    for k in range(n_steps + 1):
        t = k * dt
        R = R0 + v * min(t, T_grow)
        if k % 20 == 0:
            gel.select_samples(gel.full(x), R)
        gel.R = R
        _, g = gel.fun(x)
        if k % every == 0 and t >= 10.0:
            rows += split(gel, gel.full(x), R, case, t=round(t, 3))
            gel.select_samples(gel.full(x), R)          # split() used all segments; restore the pre-selection
            if int(round(t)) % 10 == 0:
                print(f"[{case}] t={t:.0f} R={R:.2f} P={rows[-1]['P_total']:.3e} F_share: "
                      + " ".join(f"{q['F_share']:.2f}" for q in rows[-5:]) + f"  {time.time() - tic:.0f}s",
                      flush=True)
        x = x - dt * g
    return rows


def save(rows):
    d = pd.DataFrame(rows)
    if os.path.exists(CSV):
        old = pd.read_csv(CSV)
        d = pd.concat([old[~old["case"].isin(d["case"].unique())], d])
    d.to_csv(CSV, index=False)


def summary():
    d = pd.read_csv(CSV)
    print("\nshare of push-back (F) vs share of contact length (C) vs fiber length within 1 of the sphere (N),\n"
          "by |cos psi| bins 0-.2 .2-.4 .4-.6 .6-.8 .8-1 (left = lying flat, right = end-on)")
    for case, g in d.groupby("case", sort=False):
        for R, gr in g.groupby("R" if g["t"].isna().all() else "t"):
            gr = gr.sort_values("bin")
            if gr["P_total"].iloc[0] <= 0:
                continue
            print(f"  {case:28s} {'R' if g['t'].isna().all() else 't'}={R:<5g} P={gr['P_total'].iloc[0]:.2e} "
                  f"F {' '.join(f'{q:.2f}' for q in gr['F_share'])} | C {' '.join(f'{q:.2f}' for q in gr['contact_share'])}"
                  f" | N {' '.join(f'{q:.2f}' for q in gr['near_share'])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--only", nargs="*", help="run only these case labels")
    a = ap.parse_args()
    if not a.summary_only:
        os.makedirs(os.path.dirname(CSV), exist_ok=True)
        cases = []
        for bend, kb in [("original", 0.001), ("wlc", 0.5)]:
            for arm in ["intact", "cut_endon", "cut_flat"]:
                cases.append(("static", "p1.0", bend, kb, arm))
        cases.append(("static", "p0.7", "original", 0.001, "intact"))
        cases += [("dynamic", "p1.0", "original", 0.001, "intact"), ("dynamic", "p0.7", "original", 0.001, "intact")]
        for mode, p, bend, kb, arm in cases:
            case = f"{mode}:{p}:{bend}:kb{kb:g}:{arm}"
            if a.only and case not in a.only:
                continue
            ce.BEND = bend                      # read by ce.Gel.parts at call time
            path = os.path.join(INP, f"{p}_seed7", "ECM.topo")
            if mode == "static":
                gel = ce.Gel(path, kb, R_far=20.0, kw=500.0, h=0.4)
                if arm != "intact":
                    info = cut(gel, arm.split("_")[1])
                    print(f"[{case}] cut {info}", flush=True)
                rows = run_static(gel, case)
            else:
                rows = run_dyn(path, kb, case)
            save(rows)
    summary()
