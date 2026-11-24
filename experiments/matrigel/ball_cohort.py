#!/usr/bin/env python3
"""
matrigel/ball_cohort.py -- the growing-ball test on the two gels of the FULL organoid runs, measured the
way Result 1 and the full runs are measured: S of the same strands, followed as the ball grows.
(Nada, 24 Nov 2026: the Matrigel slide's chart was unclear; ball_test.py's shell measure jumps between
lattice layers on a mesh whose strands are 2.83 long.)

Gels (same lattice sites, same hole, box 64):
  collagen   experiments/ecm_connectivity/inputs/p1.0_seed7/ECM.topo  long fibers, ks = 10, kb = 0.001 (original bend)
  mesh10     experiments/matrigel/inputs/mesh_p1.0_seed7/ECM.topo     every bond its own strand, ks = 10, no bending
  mesh4.5    the same mesh with ks = 4.5 (the first softened full run; chosen from ball_test.py's separately built
             lattice, where the mesh pushed 2.2x harder; on THIS mesh it still pushes about 2x harder)
  mesh2      ks = 2.0: matches the fibers' push when the ball reaches R = 6.0, about the organoid's size at t = 80
Test as experiments/contact_sweep/sweep.py: a rigid ball (centre 0) grown quasi-statically R = 3.0 .. 6.6 in steps
of 0.6, the gel relaxed to its energy minimum after each step (L-BFGS), held fixed beyond R_far = 20, contact
k = 50, delta = 0.2.
Measured at each R:
  S_same   S = <(3 cos^2 psi - 1)/2> of the segments/strands whose midpoints START between 3.15 and 5.15 from the
           centre (as Result 1 and full_runs.py); psi = angle to the radial direction; -1/2 all flat, 0 random
  S_shell  the same S over the segments with midpoints in the shell R .. R+2 now (as sweep.py / ball_test.py)
  P_total  the gel's push-back on the ball: total contact force / ball area
Writes results/ball_cohort.csv.   Usage: python3 experiments/matrigel/ball_cohort.py
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

ce.RANGE = 0.2
ce.BEND = "original"
COLLAGEN = os.path.join(ROOT, "experiments", "ecm_connectivity", "inputs", "p1.0_seed7", "ECM.topo")
MESH = os.path.join(HERE, "inputs", "mesh_p1.0_seed7", "ECM.topo")


class MeshGel(ce.Gel):
    """ce.Gel for single-bond strands with no bending (copied from ball_test.py: ce.Gel's constructor cannot
    index an empty bend array). Energies, contact and clamping are ce.Gel's."""
    def __init__(self, path, R_far, clamp_width, kw, h):
        nodes, _, fibers = ce.gv.read_ecm(path)
        ids = sorted(nodes)
        idx = {i: k for k, i in enumerate(ids)}
        X = np.array([nodes[i] for i in ids])
        bonds = np.array([(idx[p], idx[q]) for f in fibers for p, q in zip(f[:-1], f[1:])])
        free = np.linalg.norm(X, axis=1) < R_far - clamp_width
        keep_b = free[bonds].any(axis=1)
        used = np.unique(np.concatenate([bonds[keep_b].ravel(), np.flatnonzero(free)]))
        loc = -np.ones(len(X), int)
        loc[used] = np.arange(len(used))
        self.X0, self.free = X[used].copy(), free[used]
        self.fidx = np.flatnonzero(self.free)
        self.bonds, self.bends = loc[bonds[keep_b]], np.zeros((0, 3), int)
        self.kb, self.kw, self.h, self.periodic = 0.0, kw, h, False
        self.n, self.R = len(used), 0.0
        self.set_samples(np.arange(len(self.bonds)))


def S_of(gel, Y, sel=None, R=None):
    i, j = gel.bonds[:, 0], gel.bonds[:, 1]
    d = gel.wrap(Y[j] - Y[i])
    m = Y[i] + 0.5 * d
    rm = np.linalg.norm(m, axis=1)
    c = np.abs(np.einsum("ij,ij->i", d, m)) / (np.linalg.norm(d, axis=1) * rm + 1e-30)
    if sel is None:
        sel = (rm > R) & (rm < R + 2.0)
    return float(np.mean(1.5 * c[sel] ** 2 - 0.5)), int(np.sum(sel)), rm


rows = []
for name, ks in (("collagen", 10.0), ("mesh10", 10.0), ("mesh4.5", 4.5), ("mesh2", 2.0)):
    ce.KS = ks
    gel = ce.Gel(COLLAGEN, 0.001, R_far=20.0, kw=50.0, h=0.4) if name == "collagen" else MeshGel(MESH, 20.0, 2.83, 50.0, 0.4)
    x = gel.X0[gel.fidx].ravel().copy()
    _, _, rm0 = S_of(gel, gel.full(x), R=3.0)
    cohort = (rm0 >= 3.15) & (rm0 < 5.15)
    R_prev, tic = None, time.time()
    for R in np.round(np.arange(3.0, 6.6 + 1e-9, 0.6), 4):           # same schedule as sweep.py
        steps = [R] if R_prev is None else list(np.linspace(R_prev, R, 3)[1:])
        for s_i, Rs in enumerate(steps):
            mi = 3000 if s_i == len(steps) - 1 else 600
            gel.R = Rs
            gel.select_samples(gel.full(x), Rs)
            x = minimize(gel.fun, x, jac=True, method="L-BFGS-B",
                         options=dict(maxiter=mi, maxfun=2 * mi, gtol=1e-9, ftol=1e-16, maxcor=30, maxls=50)).x
        R_prev = R
        Y = gel.full(x)
        S_same, n_same, _ = S_of(gel, Y, sel=cohort)
        S_shell, n_shell, _ = S_of(gel, Y, R=R)
        P = ca.split(gel, Y, R, name)[0]["P_total"]
        rows.append(dict(gel=name, ks=ks, R=R, S_same=S_same, n_same=n_same, S_shell=S_shell, n_shell=n_shell, P_total=P))
        print(f"[{name}] R={R:.1f} S_same={S_same:+.3f} (n={n_same}) S_shell={S_shell:+.3f} P={P:.3f} {time.time() - tic:.0f}s", flush=True)
    with open(os.path.join(HERE, "results", "ball_cohort.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
print("done")
