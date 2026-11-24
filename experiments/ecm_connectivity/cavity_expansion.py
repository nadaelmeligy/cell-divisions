#!/usr/bin/env python3
"""
cavity_expansion.py -- how hard can our ECM gels push back on a growing organoid?

Quasi-static cavity expansion of the fiber network alone (no cells, no linkers). A rigid sphere of
radius R sits at the centre of the gel's hole (0, 0, 0). R is grown from 3.0 to 6.6 in steps of 0.3
(cavity strain e = (R - 3.3) / 3.3, 3.3 = nearest node at the start; the first segments are touched at
R = 2.8) and at every R the network is relaxed towards its energy minimum (L-BFGS-B, warm start from
the previous R, two sub-steps of 600 iterations in between, then 3000 iterations at R). The floppy
network never reaches a tiny gradient, so the run stops at the iteration limit; the column P_half (the
pressure half-way through those 3000 iterations) shows how well P_force has settled.
The pressure the gel puts on the sphere,
    P_force = (sum of the radial contact forces on the sphere) / (4 pi R^2),
is the pressure the gel would put on an organoid of surface radius R. Compare it with the cells'
pressure scale: growth stops at P_h = 6, a 10% slow-down needs P = 0.6 (run_growth.py).

Energy (model units; the same terms as the engine):
    stretch  1/2 ks (l - l0)^2           every fiber segment                  (Energy/FiberElasticity.cpp)
    bend     1/2 kb (cos(theta) + 1)^2   every two consecutive segments of a fiber (crosslinks rotate freely)
    contact  1/2 kw ds max(0, R + 0.2 - |x|)^2 at test points x = (1-w) x_i + w x_j, 8 per segment
             (0.35 apart, still < 0.5 at 40% stretch), force shared by the two nodes with weights (1-w)
             and w (the form of Energy/CellFiberContact.cpp, range 0.2; kw = 500 per unit length instead
             of the engine's 50, so the sphere is nearly rigid: the deepest penetration is printed).
             Test points are fixed in the fiber (w and ds = l0/8 from the start).
Note: all fibers of these gels start straight (cos(theta) = -1), and 1/2 kb (cos(theta) + 1)^2 is
quartic in the kink angle about a straight fiber, so bending adds no linear stiffness for any kb.
Boundary (default): nodes with |r| > R_far - 2.83 (R_far = 20) are clamped where they start, the rest
are free. Every energy term with at least one free node is kept (so a bend whose far end lies just past
R_far is not lost); terms of clamped nodes only are constant and dropped.
--periodic instead frees every node of the 64-box with periodic (minimum-image) bonds, as in the engine.
--dynamic V does not minimize: it runs the engine's own dynamics, overdamped dr/dt = mu F (mu = 1, explicit
Euler, dt = 0.005, as Run.cpp), on the periodic gel around a sphere whose radius grows at speed V (0.033 = the
organoid's surface speed between t = 80 and 130 in runs/long_ecm_p1.0_s7) from 2.6 to 4.9, then holds it for
5 time units; contact as the engine (kw = 50 per unit length, test points <= 0.5 apart). This shows how much
of the gel's push-back in the engine is drag of the fiber nodes rather than elasticity. One row per time
unit goes to results/cavity_expansion_dynamic.csv (speed = |mu F| of the nodes; n_moving = nodes faster
than V/10).

Recorded at each R (results/cavity_expansion.csv): P_force; P_work = dE_total/dV between consecutive R
(V = 4/3 pi R^3, a cross-check, with P_force_trap = trapezoid of the contact force over the same
interval); stretch, bend and contact energy; segments touching the sphere; mean and max |l - l0|/l0
of segments within 2 of the sphere; deepest penetration; final gradient (max and 2-norm); how unevenly
the load is spread over cell-sized patches of the sphere (patch_stats).

Run from the repository root (single process; ~4-5 min per clamped case, ~25 min per --periodic case
with the coarser radii below):
    python3 experiments/ecm_connectivity/cavity_expansion.py --gradcheck        # check the gradient first
    python3 experiments/ecm_connectivity/cavity_expansion.py                    # p 1.0 and 0.7, kb 0.001 and 0.5
    python3 experiments/ecm_connectivity/cavity_expansion.py --cases p0.8:0.001 p0.9:0.001
    python3 experiments/ecm_connectivity/cavity_expansion.py --cases p1.0:0.001 --R-far 28   # boundary check
    python3 experiments/ecm_connectivity/cavity_expansion.py --cases p1.0:0.001 --periodic --R 3.0 6.6 0.6 --substeps 2
    python3 experiments/ecm_connectivity/cavity_expansion.py --cases p0.7:0.001 --periodic --R 3.0 6.6 0.6 --substeps 2
    python3 experiments/ecm_connectivity/cavity_expansion.py --cases p1.0:0.001 p0.7:0.001 --dynamic 0.033  # ~4 min each
    python3 experiments/ecm_connectivity/cavity_expansion.py --cases p1.0:0.001 --dynamic 0.066            # speed check
    python3 experiments/ecm_connectivity/cavity_expansion.py --plot-only
Rows are appended to the csv (a case run again replaces its old rows); the png is redrawn from the csv.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
import growthviz as gv  # noqa: E402

KS, L0, BOX = 10.0, 2.8284271247461903, 64.0
RANGE = 0.2            # contact range, as Energy/CellFiberContact.cpp
BEND = "original"      # --bend wlc: kb (1 + cos theta), the engine's "fiber ks l0 kb 1" (worm-like chain)
R_HOLE = 3.3           # nearest gel node at the start: zero cavity strain
CSV = os.path.join(HERE, "results", "cavity_expansion.csv")
PNG = os.path.join(HERE, "results", "cavity_expansion.png")
FIELDS = ["case", "p", "seed", "kb", "boundary", "R_far", "kw", "R", "strain", "P_force", "P_half", "P_work",
          "P_force_trap", "E_total", "E_stretch", "E_bend", "E_contact", "n_touch",
          "stretch_mean_near", "stretch_max_near", "n_near", "pen_max", "grad_max", "grad_norm",
          "n_patch", "frac_loaded", "patch_max_over_mean", "top10_share",
          "n_iter", "n_free", "status", "wall_s"]
CSV_DYN = os.path.join(HERE, "results", "cavity_expansion_dynamic.csv")
FIELDS_DYN = ["case", "p", "seed", "kb", "v", "t", "R", "strain", "P_force", "E_stretch", "E_bend", "E_contact",
              "n_touch", "pen_max", "speed_mean", "speed_max", "n_moving", "wall_s"]


def scatter(idx, vec, n):
    """Sum rows of vec (m x 3) into an n x 3 array at rows idx."""
    return np.stack([np.bincount(idx, vec[:, k], minlength=n) for k in range(3)], axis=1)


class Gel:
    """Fiber network with clamped / free nodes, energy and gradient as in the engine plus a sphere."""

    def __init__(self, path, kb, R_far=20.0, clamp_width=L0, periodic=False, kw=500.0, h=0.5):
        nodes, _, fibers = gv.read_ecm(path)
        ids = sorted(nodes)
        idx = {i: k for k, i in enumerate(ids)}
        X = np.array([nodes[i] for i in ids])
        bonds = np.array([(idx[a], idx[b]) for f in fibers for a, b in zip(f[:-1], f[1:])])
        bends = np.array([(idx[a], idx[b], idx[c]) for f in fibers
                          for a, b, c in zip(f[:-2], f[1:-1], f[2:])])
        if periodic:
            free = np.ones(len(X), bool)
        else:
            free = np.linalg.norm(X, axis=1) < R_far - clamp_width
        keep_b = free[bonds].any(axis=1)
        keep_t = free[bends].any(axis=1)
        used = np.unique(np.concatenate([bonds[keep_b].ravel(), bends[keep_t].ravel(), np.flatnonzero(free)]))
        loc = -np.ones(len(X), int)
        loc[used] = np.arange(len(used))
        self.X0 = X[used].copy()
        self.free = free[used]
        self.fidx = np.flatnonzero(self.free)
        self.bonds = loc[bonds[keep_b]]
        self.bends = loc[bends[keep_t]]
        self.kb, self.kw, self.h, self.periodic = kb, kw, h, periodic
        self.n = len(used)
        self.R = 0.0
        self.set_samples(np.arange(len(self.bonds)))

    # ---- geometry helpers
    def wrap(self, d):
        if self.periodic:
            d = d - BOX * np.round(d / BOX)
        return d

    def full(self, x):
        Y = self.X0.copy()
        Y[self.fidx] = x.reshape(-1, 3)
        return Y

    def set_samples(self, segs):
        """Contact test points on the segments `segs` (indices into self.bonds), fixed in the fiber."""
        ns = int(np.ceil(L0 / self.h))
        k = np.arange(ns)
        self.s_seg = np.repeat(segs, ns)
        self.s_w = np.tile((k + 0.5) / ns, len(segs))
        self.s_ds = L0 / ns
        self.ns = ns

    def select_samples(self, Y, R, margin=3.0):
        """Keep test points only on segments that pass within R + RANGE + margin of the centre."""
        d = self.wrap(Y[self.bonds[:, 1]] - Y[self.bonds[:, 0]])
        dist = closest_approach(Y[self.bonds[:, 0]], d)
        self.set_samples(np.flatnonzero(dist < R + RANGE + margin))

    # ---- energy and gradient
    def parts(self, Y, R, terms="sbc"):
        """Return (E_stretch, E_bend, E_contact, gradient n x 3, contact info); terms picks
        which of stretch (s), bend (b), contact (c) go into the gradient (all by default)."""
        gi, gv_ = [], []                      # gradient pieces: node indices, n x 3 values
        i, j = self.bonds[:, 0], self.bonds[:, 1]
        d = self.wrap(Y[j] - Y[i])
        l = np.sqrt(np.einsum("ij,ij->i", d, d))
        Es = 0.5 * KS * np.sum((l - L0) ** 2)
        if "s" in terms:
            f = (KS * (l - L0) / l)[:, None] * d
            gi += [j, i]
            gv_ += [f, -f]
        Eb = 0.0
        if self.kb > 0 and len(self.bends) and "b" in terms:
            a, b, c = self.bends[:, 0], self.bends[:, 1], self.bends[:, 2]
            u = self.wrap(Y[a] - Y[b])
            v = self.wrap(Y[c] - Y[b])
            lu2, lv2 = np.einsum("ij,ij->i", u, u), np.einsum("ij,ij->i", v, v)
            luv = np.sqrt(lu2 * lv2)
            cos = np.einsum("ij,ij->i", u, v) / luv
            if BEND == "wlc":    # E = kb (1 + cos), dE/dcos = kb
                Eb = self.kb * np.sum(cos + 1)
                pref = self.kb * np.ones_like(cos)
            else:                # E = 0.5 kb (1 + cos)^2, dE/dcos = kb (1 + cos)
                Eb = 0.5 * self.kb * np.sum((cos + 1) ** 2)
                pref = self.kb * (cos + 1)
            du = (pref / luv)[:, None] * v - (pref * cos / lu2)[:, None] * u
            dv = (pref / luv)[:, None] * u - (pref * cos / lv2)[:, None] * v
            gi += [a, c, b]
            gv_ += [du, dv, -(du + dv)]
        # contact with the sphere (centre 0, radius R, range RANGE)
        si, sj = self.bonds[self.s_seg, 0], self.bonds[self.s_seg, 1]
        w = self.s_w
        x = Y[si] + w[:, None] * self.wrap(Y[sj] - Y[si])
        r = np.sqrt(np.einsum("ij,ij->i", x, x))
        pen = R + RANGE - r
        on = pen > 0
        Ec, Fr, pen_max, touch = 0.0, 0.0, 0.0, np.zeros(0, int)
        xc, fc = np.zeros((0, 3)), np.zeros(0)
        if on.any() and "c" in terms:
            pen, x, r, w, si, sj = pen[on], x[on], r[on], w[on], si[on], sj[on]
            Ec = 0.5 * self.kw * self.s_ds * np.sum(pen ** 2)
            fmag = self.kw * self.s_ds * pen          # outward force on the fiber at x
            Fr = float(np.sum(fmag))                  # = total radial force on the sphere
            dEdx = -(fmag / r)[:, None] * x
            gi += [si, sj]
            gv_ += [(1 - w)[:, None] * dEdx, w[:, None] * dEdx]
            pen_max = float(pen.max())
            touch = np.unique(self.s_seg[on])
            xc, fc = x, fmag
        g = scatter(np.concatenate(gi), np.concatenate(gv_), self.n) if gi else np.zeros_like(Y)
        return Es, Eb, Ec, g, dict(F_r=Fr, pen_max=pen_max, touch=touch, x=xc, f=fc)

    def fun(self, x):
        Y = self.full(x)
        Es, Eb, Ec, g, _ = self.parts(Y, self.R)
        return Es + Eb + Ec, g[self.fidx].ravel()


def patch_stats(x, f, R, a_cell=np.pi * 0.6 ** 2):
    """How unevenly the gel load is spread over cell-sized patches of the sphere.
    The sphere is cut into n = 4 pi R^2 / a_cell patches (a_cell = face of a cell of diameter 1.2;
    Fibonacci points, each contact force goes to the nearest point). Returns the fraction of patches
    with any load, the largest patch pressure / mean pressure, and the share of the total force
    carried by the most loaded 10% of patches."""
    n = max(1, int(round(4 * np.pi * R ** 2 / a_cell)))
    k = np.arange(n) + 0.5
    z = 1 - 2 * k / n
    phi = np.pi * (1 + 5 ** 0.5) * k
    c = np.stack([np.sqrt(1 - z ** 2) * np.cos(phi), np.sqrt(1 - z ** 2) * np.sin(phi), z], axis=1)
    if len(f) == 0 or f.sum() <= 0:
        return dict(n_patch=n, frac_loaded=0.0, patch_max_over_mean=np.nan, top10_share=np.nan)
    u = x / np.linalg.norm(x, axis=1)[:, None]
    load = np.bincount(np.argmax(u @ c.T, axis=1), f, minlength=n)
    srt = np.sort(load)[::-1]
    return dict(n_patch=n, frac_loaded=float(np.mean(load > 0)),
                patch_max_over_mean=float(srt[0] / load.mean()),
                top10_share=float(srt[:max(1, n // 10)].sum() / load.sum()))


def closest_approach(A, d):
    """Distance from the origin to segments A + s d, s in [0, 1]."""
    s = np.clip(-np.sum(A * d, axis=1) / np.sum(d * d, axis=1), 0.0, 1.0)
    return np.linalg.norm(A + s[:, None] * d, axis=1)


def gradcheck(path, kb, R=4.0, n_check=40, seed=1):
    """Compare the analytic gradient of each energy term with central finite differences,
    on a randomly perturbed network (bent fibers, stretched segments, many points inside the sphere)."""
    gel = Gel(path, kb)
    rng = np.random.default_rng(seed)
    Y = gel.X0.copy()
    Y[gel.fidx] += 0.3 * rng.standard_normal((len(gel.fidx), 3))
    _, _, _, _, info = gel.parts(Y, R)
    touch_nodes = np.unique(gel.bonds[info["touch"]].ravel())
    touch_nodes = touch_nodes[gel.free[touch_nodes]]
    pos = {k: n for n, k in enumerate(gel.fidx)}
    cand_t = np.array([3 * pos[k] + m for k in touch_nodes for m in range(3)])
    pick = np.concatenate([rng.choice(cand_t, n_check // 2, replace=False),
                           rng.choice(3 * len(gel.fidx), n_check // 2, replace=False)])
    out = {}
    for t, k in [("s", 0), ("b", 1), ("c", 2)]:
        g = gel.parts(Y, R, terms=t)[3][gel.fidx].ravel()
        worst, gmax = 0.0, np.abs(g[pick]).max()
        for q in pick:
            node, m = gel.fidx[q // 3], q % 3
            eps = 1e-5
            Yp, Ym = Y.copy(), Y.copy()
            Yp[node, m] += eps
            Ym[node, m] -= eps
            fd = (gel.parts(Yp, R)[k] - gel.parts(Ym, R)[k]) / (2 * eps)
            worst = max(worst, abs(fd - g[q]) / max(abs(fd), abs(g[q]), 1e-3 * gmax, 1e-300))
        out[t] = (worst, gmax)
    print(f"gradcheck {os.path.basename(os.path.dirname(path))} kb={kb}: {len(info['touch'])} segments "
          f"touching; worst relative error over {len(pick)} coordinates: "
          + ", ".join(f"{n} {out[t][0]:.1e} (|g| up to {out[t][1]:.2g})"
                      for t, n in [("s", "stretch"), ("b", "bend"), ("c", "contact")]), flush=True)
    return out


def run_case(path, kb, radii, R_far, periodic, kw, h, gtol, maxiter, maxiter_sub, substeps, label):
    gel = Gel(path, kb, R_far=R_far, periodic=periodic, kw=kw, h=h)
    x = gel.X0[gel.fidx].ravel().copy()
    rows, prev = [], None
    print(f"[{label}] {len(gel.fidx)} free nodes, {gel.n - len(gel.fidx)} clamped, {len(gel.bonds)} segments, "
          f"{len(gel.bends)} bends", flush=True)
    R_prev = None
    for R in radii:
        tic = time.time()
        steps = [R] if R_prev is None else list(np.linspace(R_prev, R, substeps + 1)[1:])
        nit, status, P_half = 0, "", np.nan
        for k, Rs in enumerate(steps):
            final = k == len(steps) - 1
            mi = maxiter if final else maxiter_sub
            gel.R = Rs
            gel.select_samples(gel.full(x), Rs)
            count, half = [0], {}

            def cb(xk):
                count[0] += 1
                if final and count[0] == mi // 2:      # pressure half-way: a convergence check
                    half["P"] = gel.parts(gel.full(xk), Rs)[4]["F_r"] / (4 * np.pi * Rs ** 2)

            res = minimize(gel.fun, x, jac=True, method="L-BFGS-B", callback=cb,
                           options=dict(maxiter=mi, maxfun=2 * mi, gtol=gtol, ftol=1e-16,
                                        maxcor=30, maxls=50))
            x = res.x
            nit += res.nit
            status = "converged" if res.success else ("maxiter" if res.nit >= mi else str(res.message)[:40])
            P_half = half.get("P", np.nan)
        R_prev = R
        Y = gel.full(x)
        # all test points (no pre-selection) for the reported numbers
        gel.set_samples(np.arange(len(gel.bonds)))
        Es, Eb, Ec, g, info = gel.parts(Y, R)
        gf = g[gel.fidx]
        d = gel.wrap(Y[gel.bonds[:, 1]] - Y[gel.bonds[:, 0]])
        l = np.linalg.norm(d, axis=1)
        near = closest_approach(Y[gel.bonds[:, 0]], d) < R + 2.0
        strain = np.abs(l - L0) / L0
        E = Es + Eb + Ec
        V = 4.0 / 3.0 * np.pi * R ** 3
        row = dict(case=label, p=label.split(":")[0], seed=7, kb=kb,
                   boundary="periodic" if periodic else "clamped", R_far=R_far, kw=kw,
                   R=round(R, 4), strain=(R - R_HOLE) / R_HOLE,
                   P_force=info["F_r"] / (4 * np.pi * R ** 2), P_half=P_half, P_work=np.nan, P_force_trap=np.nan,
                   E_total=E, E_stretch=Es, E_bend=Eb, E_contact=Ec, n_touch=len(info["touch"]),
                   stretch_mean_near=float(strain[near].mean()) if near.any() else np.nan,
                   stretch_max_near=float(strain[near].max()) if near.any() else np.nan,
                   n_near=int(near.sum()), pen_max=info["pen_max"], **patch_stats(info["x"], info["f"], R),
                   grad_max=float(np.abs(gf).max()), grad_norm=float(np.linalg.norm(gf)),
                   n_iter=nit, n_free=len(gel.fidx), status=status, wall_s=round(time.time() - tic, 1))
        if prev is not None:
            dV = V - prev["V"]
            row["P_work"] = (E - prev["E"]) / dV
            row["P_force_trap"] = 0.5 * (info["F_r"] + prev["F"]) * (R - prev["R"]) / dV
        prev = dict(V=V, E=E, F=info["F_r"], R=R)
        rows.append(row)
        print(f"[{label}] R={R:.2f} e={row['strain']:+.2f} P_force={row['P_force']:.3e} (half-way {P_half:.3e}) "
              f"P_work={row['P_work']:.3e} Es={Es:.3e} Eb={Eb:.3e} Ec={Ec:.2e} touch={row['n_touch']} "
              f"str_mean={row['stretch_mean_near']:.2e} str_max={row['stretch_max_near']:.2e} "
              f"pen={row['pen_max']:.3f} |g|max={row['grad_max']:.1e} it={nit} {status} "
              f"{row['wall_s']}s", flush=True)
    return rows


def run_dynamic(path, kb, v, R0, R1, hold, dt, label, kw=50.0, h=0.5):
    """Engine dynamics (overdamped, mu = 1, explicit Euler step dt) of the periodic gel, all nodes free, around
    a sphere growing at speed v from R0 to R1 and then held for `hold` time units. One row per time unit."""
    gel = Gel(path, kb, periodic=True, kw=kw, h=h)
    x = gel.X0[gel.fidx].ravel().copy()
    T_grow = (R1 - R0) / v
    n_steps = int(round((T_grow + hold) / dt))
    every = int(round(1.0 / dt))
    rows, tic = [], time.time()
    for k in range(n_steps + 1):
        t = k * dt
        R = R0 + v * min(t, T_grow)
        if k % 20 == 0:
            gel.select_samples(gel.full(x), R)
        gel.R = R
        E, g = gel.fun(x)
        if k % every == 0:
            Es, Eb, Ec, _, info = gel.parts(gel.full(x), R)
            sp = np.linalg.norm(g.reshape(-1, 3), axis=1)          # node speed |mu F|, mu = 1
            row = dict(case=label, p=label.split(":")[0], seed=7, kb=kb, v=v, t=round(t, 3), R=round(R, 4),
                       strain=(R - R_HOLE) / R_HOLE, P_force=info["F_r"] / (4 * np.pi * R ** 2), E_stretch=Es,
                       E_bend=Eb, E_contact=Ec, n_touch=len(info["touch"]), pen_max=info["pen_max"],
                       speed_mean=float(sp.mean()), speed_max=float(sp.max()), n_moving=int((sp > 0.1 * v).sum()),
                       wall_s=round(time.time() - tic, 1))
            rows.append(row)
            if (k // every) % 5 == 0:
                print(f"[{label}] t={t:6.1f} R={R:.3f} P={row['P_force']:.3e} Es={Es:.3e} Eb={Eb:.3e} "
                      f"pen={row['pen_max']:.3f} touch={row['n_touch']} moving={row['n_moving']} "
                      f"{row['wall_s']}s", flush=True)
        x = x - dt * g                                             # dr = mu F dt, F = -grad E
    return rows


def save_rows(rows, path=CSV, fields=FIELDS):
    old = []
    if os.path.exists(path):
        with open(path) as f:
            old = [r for r in csv.DictReader(f)]
    labels = {r["case"] for r in rows}
    old = [r for r in old if r["case"] not in labels]
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for r in old + rows:
            wr.writerow({k: r.get(k, "") for k in fields})


def plot(marks=()):
    """P_force vs cavity strain. Left: the clamped (R_far = 20) cases; right: boundary checks.
    marks: (label, surface radius) pairs drawn as vertical lines (where the organoid runs are)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with open(CSV) as f:
        rows = list(csv.DictReader(f))
    cases = []
    for r in rows:
        if r["case"] not in cases:
            cases.append(r["case"])
    colors = {"p1.0": "#1f4e9a", "p0.9": "#2a9d8f", "p0.8": "#e9a23b", "p0.7": "#c0392b"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    lo = 1e-7
    for c in cases:
        rr = [r for r in rows if r["case"] == c]
        e = np.array([float(r["strain"]) for r in rr])
        P = np.array([float(r["P_force"]) for r in rr])
        p, kb, bnd, rfar = rr[0]["p"], float(rr[0]["kb"]), rr[0]["boundary"], float(rr[0]["R_far"])
        kw = float(rr[0]["kw"])
        main = bnd == "clamped" and rfar == 20 and kw == 500
        Pm = np.maximum(P, 1.3 * lo)          # P below the axis floor is drawn on the floor
        lab = f"{p.replace('p', 'p = ')}, kb = {kb:g}"
        wlc = c.endswith(":wlc")              # --bend wlc rows: kb (1 + cos theta) bending
        if wlc:
            lab += ", bend kb (1 + cos)"
        if main:
            ls, mk = (":", "D") if wlc else (("-", "o") if kb < 0.01 else ("--", "s"))
            axes[0].plot(e, Pm, ls, marker=mk, ms=3.5, lw=1.6, color=colors.get(p, "k"), label=lab)
            if p in ("p1.0", "p0.7") and not wlc:   # faint copy on the right, to compare with the periodic box
                axes[1].plot(e, Pm, "-" if kb < 0.01 else "--", marker="o" if kb < 0.01 else "s", ms=2.5,
                             lw=1.0, alpha=0.4, color=colors.get(p, "k"), label=lab + ", clamped at r > 17.2")
        else:
            extra = "periodic box, all nodes free" if bnd == "periodic" else f"clamped at r > {rfar - L0:.1f}"
            if kw != 500:
                extra += f", kw = {kw:g}"
            ls, mk = ((":", "^") if kb < 0.01 else ("-.", "v")) if bnd == "periodic" else ("--", "x")
            axes[1].plot(e, Pm, ls, marker=mk, ms=4, lw=1.8, color=colors.get(p, "k"), label=f"{lab}, {extra}")
    if os.path.exists(CSV_DYN):                # engine dynamics (--dynamic): thick lines, right panel
        with open(CSV_DYN) as f:
            drows = list(csv.DictReader(f))
        for c in dict.fromkeys(r["case"] for r in drows):
            rr = [r for r in drows if r["case"] == c]
            e = np.array([float(r["strain"]) for r in rr])
            P = np.array([float(r["P_force"]) for r in rr])
            p, kb, v = rr[0]["p"], float(rr[0]["kb"]), float(rr[0]["v"])
            ls = ("-" if kb < 0.01 else (0, (4, 1, 1, 1))) if v < 0.05 else (0, (6, 2))
            axes[1].plot(e, np.maximum(P, 1.3 * lo), ls=ls, lw=2.6, alpha=0.8,
                         color=colors.get(p, "k"),
                         label=f"{p.replace('p', 'p = ')}, kb = {kb:g}, engine dynamics (periodic), speed {v:g}")
    eg = np.linspace(0.02, 1, 50)
    for ax in axes:
        for y, txt in [(6.0, "P_h = 6: growth stops"), (0.6, "0.6: growth 10% slower"),
                       (0.06, "0.06: 1% slower")]:
            ax.axhline(y, color="0.35", lw=1.0 if y > 0.1 else 0.7, ls="-" if y > 0.1 else "--")
            ax.text(-0.09, y * 1.15, txt, fontsize=7.5, color="0.25")
        ax.plot(eg, 4 * 1e-6 * eg, color="0.55", lw=1.0, ls="-.", label="linear elastic 4 G e, G = 1e-6")
        for i, (txt, Rs) in enumerate(marks):
            es = (Rs - R_HOLE) / R_HOLE
            ax.axvline(es, color="0.7", lw=0.8)
            ax.text(es + 0.008, 2.2e-7 if i % 2 == 0 else 1.2, txt, rotation=90, fontsize=7, color="0.4",
                    va="bottom")
        ax.set_yscale("log")
        ax.set_ylim(lo, 30)
        ax.set_xlim(-0.12, 1.03)
        ax.set_xlabel("cavity strain e = (R - 3.3) / 3.3   (R = rigid sphere radius)")
        ax.grid(alpha=0.25, which="both", lw=0.5)
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, frameon=False)
    axes[0].set_ylabel("gel pressure on the sphere  P_force")
    for ax in axes:
        ax.text(0.99, 0.012, "points on the floor: P < 1e-7", transform=ax.transAxes, ha="right",
                fontsize=6.5, color="0.45")
    axes[0].set_title("Gel alone, clamped at r > 17.2: connectivity p and bending kb", fontsize=10)
    axes[1].set_title("Boundary check: clamped (faint), periodic box with all nodes free (as in the engine),\n"
                      "and the engine's own dynamics (thick, sphere growing at the organoid's speed)", fontsize=9)
    fig.tight_layout()
    fig.savefig(PNG, dpi=160, bbox_inches="tight")
    print("wrote", PNG)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", nargs="+", default=["p1.0:0.001", "p0.7:0.001", "p1.0:0.5", "p0.7:0.5"],
                    help="gel:kb pairs; gel = p1.0, p0.9, p0.8, p0.7 (seed 7 inputs)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--R-far", type=float, default=20.0)
    ap.add_argument("--periodic", action="store_true", help="free every node, periodic box (no clamp)")
    ap.add_argument("--kw", type=float, default=500.0)
    ap.add_argument("--h", type=float, default=0.4, help="largest spacing of contact test points")
    ap.add_argument("--R", type=float, nargs=3, default=[3.0, 6.6, 0.3], metavar=("START", "STOP", "STEP"))
    ap.add_argument("--substeps", type=int, default=3)
    ap.add_argument("--gtol", type=float, default=1e-9)
    ap.add_argument("--maxiter", type=int, default=3000, help="L-BFGS iterations at each recorded R")
    ap.add_argument("--maxiter-sub", type=int, default=600, help="L-BFGS iterations at each sub-step")
    ap.add_argument("--dynamic", type=float, nargs="+", metavar="V",
                    help="engine dynamics at these sphere speeds instead of minimization (periodic gel)")
    ap.add_argument("--R-dyn", type=float, nargs=2, default=[2.6, 4.9], metavar=("R0", "R1"))
    ap.add_argument("--hold", type=float, default=5.0, help="time units at R1 after the growth (--dynamic)")
    ap.add_argument("--dt", type=float, default=0.005, help="time step of --dynamic (engine: 0.005)")
    ap.add_argument("--gradcheck", action="store_true")
    ap.add_argument("--bend", choices=["original", "wlc"], default="original",
                    help="bending energy: original 0.5 kb (1+cos)^2, or wlc kb (1+cos) (engine 'fiber ks l0 kb 1')")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--mark", nargs="*", default=["p1.0 t=80:3.25", "p1.0 t=130:4.90", "p0.7 t=130:4.98"],
                    help="label:R vertical lines; default = organoid surface_radius read from runs/*/frames.pkl "
                         "on 2026-11-18 (ecm_p1.0_s7 t=80, long_ecm_p1.0_s7 and long_ecm_p0.7_s7 t=130)")
    a = ap.parse_args()
    BEND = a.bend   # module-level switch read by Gel.energy_grad (this block runs at module level)
    inp = os.path.join(HERE, "inputs")
    if a.gradcheck:
        for p in ["p1.0", "p0.7"]:
            for kb in [0.001, 0.5]:
                gradcheck(os.path.join(inp, f"{p}_seed{a.seed}", "ECM.topo"), kb)
        sys.exit(0)
    if a.dynamic and not a.plot_only:
        for c in a.cases:
            p, kb = c.split(":")
            for v in a.dynamic:
                rows = run_dynamic(os.path.join(inp, f"{p}_seed{a.seed}", "ECM.topo"), float(kb), v, a.R_dyn[0],
                                   a.R_dyn[1], a.hold, a.dt, f"{p}:{float(kb):g}:v{v:g}" + (":wlc" if BEND == "wlc" else ""))
                for r in rows:
                    r["seed"] = a.seed
                save_rows(rows, CSV_DYN, FIELDS_DYN)
    elif not a.plot_only:
        radii = np.round(np.arange(a.R[0], a.R[1] + 1e-9, a.R[2]), 4)
        for c in a.cases:
            p, kb = c.split(":")
            label = f"{p}:{float(kb):g}"
            if a.periodic:
                label += ":periodic"
            elif a.R_far != 20.0:
                label += f":Rfar{a.R_far:g}"
            if a.kw != 500.0:
                label += f":kw{a.kw:g}"
            if BEND == "wlc":
                label += ":wlc"
            rows = run_case(os.path.join(inp, f"{p}_seed{a.seed}", "ECM.topo"), float(kb), radii, a.R_far,
                            a.periodic, a.kw, a.h, a.gtol, a.maxiter, a.maxiter_sub, a.substeps, label)
            for r in rows:
                r["seed"] = a.seed
            save_rows(rows)
    plot([(m.rsplit(":", 1)[0], float(m.rsplit(":", 1)[1])) for m in a.mark])
