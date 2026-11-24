#!/usr/bin/env python3
"""
matrigel/ball_test.py -- the gel-alone growing-ball test (as experiments/contact_sweep/sweep.py) on a
Matrigel-like gel instead of our collagen-like fiber gel.  (Nada, 24 Nov 2026: "the same experiment with
the Matrigel".)

Why: the experiments use Matrigel, a basement-membrane gel (mainly laminin, collagen IV, entactin and a
heparan sulfate proteoglycan; Emonard 1987) whose structure is "interconnected thin sheets" (Kleinman 1986),
and it is soft (a few hundred Pa; Soofi 2009). Our gel so far is collagen-like: long straight fibers of
2.83-long segments on an FCC lattice, held mainly by stretching along each fiber, with almost no bending
stiffness.

Matrigel-like proxy used here: an FCC lattice with nearest-neighbour spacing a, with EVERY lattice bond kept
as its own short strand (12 per node; no long fibers, so no bending term): a dense central-force network,
which behaves like an elastic solid. Two mesh sizes: a = 1.0 (finer than a cell, which is 1.2 across) and
a = 1.4. The hole has the same radius as in the collagen gel (no node closer than 3.2 to the centre), the gel
is clamped beyond R_far = 20, stretching ks = 10 per strand, contact as the engine (k = 50, delta = 0.2).
Only the alignment and the push pattern are compared: in a quasi-static test with an imposed sphere the
stiffness scale (ks) changes the push-back's size, not the geometry.

Measured at every radius, exactly as the sweep: S of strands with midpoints in the shell R..R+2
(S = <(3 cos^2 psi - 1)/2>, psi = angle to the radial direction; -1/2 = all flat), the share of the push
carried by flat strands (|cos psi| < 0.2), the push-back pressure and the deepest strand point inside.

Usage:  python3 experiments/matrigel/ball_test.py A [KS]
        (writes inputs/fcc_a{A}.topo once, then results/a{A}.csv, or results/a{A}_ks{KS}.csv when a strand
        stiffness KS other than 10 is given; the log goes to results/log_*.txt via the caller)
        KS added 24 Nov 2026: the full runs also use a softened mesh, ks = 4.5 (README, "Full organoid runs").
"""
import csv
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "experiments", "ecm_connectivity"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "fiber_angle"))
import cavity_expansion as ce  # noqa: E402
import cavity_angle as ca      # noqa: E402

a = float(sys.argv[1])
KS = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
R_FAR, HOLE = 20.0, 3.2
topo = os.path.join(HERE, "inputs", f"fcc_a{a:g}.topo")

if not os.path.exists(topo):            # build the mesh once
    c = a * np.sqrt(2.0)                 # FCC cubic cell for nearest-neighbour distance a
    n = int(np.ceil((R_FAR + 1.5) / c)) + 1
    g = np.arange(-n, n + 1) * c
    base = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]]) * c
    P = (np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 1, 3) + base).reshape(-1, 3)
    r = np.linalg.norm(P, axis=1)
    P = P[(r >= HOLE) & (r <= R_FAR + 1.5)]
    pairs = sorted(cKDTree(P).query_pairs(a * 1.01))
    os.makedirs(os.path.dirname(topo), exist_ok=True)
    with open(topo, "w") as f:
        f.write(f"nodes {len(P)}\n")
        for i, (x, y, z) in enumerate(P):
            f.write(f"{i:6d} {x:.12e} {y:.12e} {z:.12e} 0\n")
        f.write(f"fibers {len(pairs)}\n")
        for k, (i, j) in enumerate(pairs):
            f.write(f"{k} {i} {j}\n")
    print(f"mesh a = {a:g}: {len(P)} nodes, {len(pairs)} strands ({2 * len(pairs) / len(P):.1f} per node)", flush=True)

ce.L0 = a                    # rest length of every strand (read by ce.Gel at call time)
ce.KS = KS                   # strand stiffness (read by ce.Gel.parts at call time; ce's default is 10)
ce.RANGE = 0.2
ce.BEND = "original"
class MeshGel(ce.Gel):
    """ce.Gel for a mesh of single-bond strands: the same set-up, but with an empty bend list (ce.Gel's own
    constructor cannot index an empty bend array). Everything else (energies, contact, clamping) is ce.Gel's."""
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


gel = MeshGel(topo, R_FAR, a, 50.0, 0.4)


def shell_S(Y, R):
    i, j = gel.bonds[:, 0], gel.bonds[:, 1]
    d = gel.wrap(Y[j] - Y[i])
    m = Y[i] + 0.5 * d
    rm = np.linalg.norm(m, axis=1)
    cs = np.abs(np.einsum("ij,ij->i", d, m)) / (np.linalg.norm(d, axis=1) * rm + 1e-30)
    sel = (rm > R) & (rm < R + 2.0)
    return float(np.mean(1.5 * cs[sel] ** 2 - 0.5)), int(sel.sum())


os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
out = os.path.join(HERE, "results", f"a{a:g}.csv" if KS == 10.0 else f"a{a:g}_ks{KS:g}.csv")
x = gel.X0[gel.fidx].ravel().copy()
rows, R_prev, tic = [], None, time.time()
for R in np.round(np.arange(3.0, 6.6 + 1e-9, 0.6), 4):           # same schedule as the sweep
    steps = [R] if R_prev is None else list(np.linspace(R_prev, R, 3)[1:])
    for s_i, Rs in enumerate(steps):
        mi = 3000 if s_i == len(steps) - 1 else 600
        gel.R = Rs
        gel.select_samples(gel.full(x), Rs)
        x = minimize(gel.fun, x, jac=True, method="L-BFGS-B",
                     options=dict(maxiter=mi, maxfun=2 * mi, gtol=1e-9, ftol=1e-16, maxcor=30, maxls=50)).x
    R_prev = R
    Y = gel.full(x)
    split = ca.split(gel, Y, R, f"matrigel_a{a:g}")
    S, n_s = shell_S(Y, R)
    _, _, _, _, info = gel.parts(Y, R)
    rows.append(dict(a=a, R=R, cav_strain=round((R - ce.R_HOLE) / ce.R_HOLE, 3), S_shell=S, n_shell=n_s,
                     flat_push=split[0]["F_share"], P_total=split[0]["P_total"], pen_max=info.get("pen_max", np.nan)))
    print(f"[a={a:g}] R={R:.1f} S_shell={S:+.3f} flat_push={split[0]['F_share']:.2f} P={split[0]['P_total']:.3e} "
          f"{time.time() - tic:.0f}s", flush=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
print("done", out)
