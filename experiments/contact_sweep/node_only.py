#!/usr/bin/env python3
"""
contact_sweep/node_only.py -- does the flattening come from how the contact force is spread along a segment?
(Nada, 24 Nov 2026: "because of the way we segment the line, is every fiber doomed to end up aligned around the
organoid? Are we fabricating new biology?")

The engine's contact (Energy/CellFiberContact.cpp) pushes test points in the middle of n pieces of each segment and
shares each push between the two nodes like a lever, (1 - w) and w. A segment touching the ball with one end is
therefore pushed harder at that end, which turns it flat. Test: remove that torque completely. Contact acts only
on the NODES (test points at w = 0 and w = 1 only, each standing for half the segment), so every push goes straight
to one node and no contact force can rotate a segment. Everything else as experiments/matrigel/ball_cohort.py:
rigid ball R = 3.0 .. 6.6, gel relaxed to its energy minimum after each step, clamped beyond 20, k = 50, delta = 0.2.

Prediction written before the run: the same strands still turn flat (S near -0.4 to -0.5 at R = 6.6) in both gels,
because an expanding ball squeezes the gel along the radius and stretches it around the ball (hoop stretch), and
that deformation alone rotates strands toward flat; the lever torque is not needed.

Measured: S of the same strands (midpoints starting 3.15 .. 5.15 from the centre), as Result 1, and the push-back.
Writes results/node_only.csv.   Usage: python3 experiments/contact_sweep/node_only.py
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

ce.RANGE, ce.BEND = 0.2, "original"
COLLAGEN = os.path.join(ROOT, "experiments", "ecm_connectivity", "inputs", "p1.0_seed7", "ECM.topo")
MESH = os.path.join(ROOT, "experiments", "matrigel", "inputs", "mesh_p1.0_seed7", "ECM.topo")


def node_only_samples(self, segs):
    """Contact test points only at the two nodes of each segment (w = 0 and 1), each for half its length."""
    self.s_seg = np.repeat(segs, 2)
    self.s_w = np.tile(np.array([0.0, 1.0]), len(segs))
    self.s_ds = ce.L0 / 2
    self.ns = 2


class MeshGel(ce.Gel):
    """ce.Gel for single-bond strands with no bending (as experiments/matrigel/ball_cohort.py)."""
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
    return float(np.mean(1.5 * c[sel] ** 2 - 0.5)), rm


rows = []
ce.Gel.set_samples = node_only_samples       # the engine-like segment contact is in experiments/matrigel/results/ball_cohort.csv
contact = "nodes only"
if True:
    for name in ("collagen", "mesh10"):
        ce.KS = 10.0
        gel = ce.Gel(COLLAGEN, 0.001, R_far=20.0, kw=50.0, h=0.4) if name == "collagen" else MeshGel(MESH, 20.0, 2.83, 50.0, 0.4)
        x = gel.X0[gel.fidx].ravel().copy()
        _, rm0 = S_of(gel, gel.full(x), R=3.0)
        cohort = (rm0 >= 3.15) & (rm0 < 5.15)
        R_prev, tic = None, time.time()
        for R in np.round(np.arange(3.0, 6.6 + 1e-9, 0.6), 4):
            steps = [R] if R_prev is None else list(np.linspace(R_prev, R, 3)[1:])
            for s_i, Rs in enumerate(steps):
                mi = 3000 if s_i == len(steps) - 1 else 600
                gel.R = Rs
                gel.select_samples(gel.full(x), Rs)
                x = minimize(gel.fun, x, jac=True, method="L-BFGS-B",
                             options=dict(maxiter=mi, maxfun=2 * mi, gtol=1e-9, ftol=1e-16, maxcor=30, maxls=50)).x
            R_prev = R
            Y = gel.full(x)
            S_same, _ = S_of(gel, Y, sel=cohort)
            P = ca.split(gel, Y, R, name)[0]["P_total"]
            rows.append(dict(contact=contact, gel=name, R=R, S_same=S_same, P_total=P))
            print(f"[{contact} | {name}] R={R:.1f} S_same={S_same:+.3f} P={P:.3f} {time.time() - tic:.0f}s", flush=True)
        with open(os.path.join(HERE, "results", "node_only.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
print("done")
