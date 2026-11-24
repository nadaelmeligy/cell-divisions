#!/usr/bin/env python3
"""
contact_sweep/sweep.py -- does the flattening of fibers around a growing organoid depend on the
shape of the contact force?  (Nada, 24 Nov 2026: "sweep k and delta, the fast way".)

Fast test = the gel alone, no cells: a rigid sphere grows in the organoid's hole (R = 3.0 .. 6.6,
quasi-static, gel clamped at R_far = 20), exactly as experiments/fiber_angle/cavity_angle.py "static",
with the engine's fiber energies (stretch ks = 10, original bend kb = 0.001). Only the contact term
changes: E_contact = 1/2 k ds max(0, R + delta - |x|)^2 at test points along every segment, the form
of Energy/CellFiberContact.cpp, with
    k     in {10, 50, 250}  per unit fiber length (engine: 50)
    delta in {0.1, 0.2, 0.4}                      (engine: 0.2)
At every R we record:
    S_shell   alignment of the segments whose midpoints are in the shell R .. R+2 around the sphere,
              S = <(3 cos^2 psi - 1)/2>, psi = angle to the radial direction (as Result 1;
              S = -1/2 all flat, 0 random, 1 all pointing at the sphere)
    flat_push share of the push-back carried by fibers lying flat (|cos psi| < 0.2, as Result 2)
    P_total   push-back pressure = total radial contact force / sphere area
    pen_max   deepest fiber point inside the sphere
Usage:  python3 experiments/contact_sweep/sweep.py K DELTA      (one case; writes results/k{K}_d{DELTA}.csv)
"""
import csv
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "experiments", "ecm_connectivity"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "fiber_angle"))
import cavity_expansion as ce  # noqa: E402
import cavity_angle as ca      # noqa: E402

k, delta = float(sys.argv[1]), float(sys.argv[2])
ce.RANGE = delta            # read by ce.Gel.parts and ca.split at call time
ce.BEND = "original"
gel = ce.Gel(os.path.join(ROOT, "experiments", "ecm_connectivity", "inputs", "p1.0_seed7", "ECM.topo"),
             0.001, R_far=20.0, kw=k, h=0.4)


def shell_S(Y, R):
    i, j = gel.bonds[:, 0], gel.bonds[:, 1]
    d = gel.wrap(Y[j] - Y[i])
    m = Y[i] + 0.5 * d
    rm = np.linalg.norm(m, axis=1)
    c = np.abs(np.einsum("ij,ij->i", d, m)) / (np.linalg.norm(d, axis=1) * rm + 1e-30)
    sel = (rm > R) & (rm < R + 2.0)
    return float(np.mean(1.5 * c[sel] ** 2 - 0.5)), int(sel.sum())


os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
out = os.path.join(HERE, "results", f"k{k:g}_d{delta:g}.csv")
x = gel.X0[gel.fidx].ravel().copy()
rows, R_prev, tic = [], None, time.time()
S0, n0 = shell_S(gel.full(x), 3.0)
for R in np.round(np.arange(3.0, 6.6 + 1e-9, 0.6), 4):           # same schedule as cavity_angle.run_static
    steps = [R] if R_prev is None else list(np.linspace(R_prev, R, 3)[1:])
    for s_i, Rs in enumerate(steps):
        mi = 3000 if s_i == len(steps) - 1 else 600
        gel.R = Rs
        gel.select_samples(gel.full(x), Rs)
        x = minimize(gel.fun, x, jac=True, method="L-BFGS-B",
                     options=dict(maxiter=mi, maxfun=2 * mi, gtol=1e-9, ftol=1e-16, maxcor=30, maxls=50)).x
    R_prev = R
    Y = gel.full(x)
    split = ca.split(gel, Y, R, f"k{k:g}_d{delta:g}")
    S, n = shell_S(Y, R)
    _, _, _, _, info = gel.parts(Y, R)
    rows.append(dict(k=k, delta=delta, R=R, cav_strain=round((R - ce.R_HOLE) / ce.R_HOLE, 3), S_shell=S, n_shell=n,
                     flat_push=split[0]["F_share"], P_total=split[0]["P_total"], pen_max=info.get("pen_max", np.nan)))
    print(f"[k={k:g} d={delta:g}] R={R:.1f} S_shell={S:+.3f} (start {S0:+.3f}) flat_push={split[0]['F_share']:.2f} "
          f"P={split[0]['P_total']:.3e} {time.time() - tic:.0f}s", flush=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
print("done", out)
