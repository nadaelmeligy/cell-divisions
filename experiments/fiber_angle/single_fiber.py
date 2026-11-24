#!/usr/bin/env python3
"""
single_fiber.py -- isolation test for experiments/fiber_angle: how hard does ONE fiber push back on a
growing organoid, depending on its angle psi to the organoid surface? (psi = 0: the fiber points straight at
the organoid; psi = 90: it lies flat on it.)

The organoid is a rigid sphere at the origin growing from R0 = 4.0 (its size at t = 80-130). The fiber has
the engine's energy: stretch 1/2 ks (l - l0)^2 (ks = 10, l0 = 2.83) and, at every inner node, bend
1/2 kb (1 + cos)^2 ("original", Energy/FiberElasticity.cpp) or kb (1 + cos) ("wlc"). Contact as
Energy/CellFiberContact.cpp: test points along each segment, ds apart; a point closer than range = 0.2 to
the sphere is pushed out with force k ds (range - d). F = total force of the fiber on the sphere.

Geometries
  bridge n   the fiber passes the sphere and touches it at its middle node (psi = 90); both ends pinned,
             n segments either side (pinned = held by crosslinks).
  tip psi    the fiber ends at the sphere (like the fibers cut by the organoid's hole), its tip just
             touching at angle psi; 3 segments, the last two nodes fixed (base clamped), tip and next free.
static   energy minimum at every R (R0 -> R0 + 1 in steps of 0.05; L-BFGS-B, k = 500 so the sphere is
         nearly rigid): elastic push-back only, no drag. Free nodes start 0.001 off the straight line (one
         step of the engine's thermal noise), so a perfectly straight fiber is free to buckle.
dynamic  no pins, 30 segments (a p = 1.0 fiber spans the box), the engine's update
         r += mu F dt + sqrt(2 mu T dt) xi (mu = 1, dt = 0.005, T = 1e-4: Run.cpp), contact k = 50 with
         test points <= 0.5 apart (the engine's), sphere growing at 0.033 (the organoid's surface speed,
         t = 80-130) for 30 time units. Push-back is then drag as well. 5 noise seeds each.
Output: results/single_fiber.csv, results/single_fiber.png and a summary table on stdout.
    python3 experiments/fiber_angle/single_fiber.py
"""
import csv
import os

import numpy as np
from scipy.optimize import minimize

KS, L0, RANGE, R0 = 10.0, 2.8284271247461903, 0.2, 4.0
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
BENDS = [("original", 0.001), ("wlc", 0.001), ("wlc", 0.5)]
GEOMS = [("bridge", 1), ("bridge", 3), ("tip", 0), ("tip", 30), ("tip", 60), ("tip", 89)]


def forces(Y, R, kb, bend, k, ns):
    """Energy, gradient (n x 3) and total contact force on the sphere, for the chain of nodes Y."""
    g = np.zeros_like(Y)
    d = Y[1:] - Y[:-1]
    l = np.linalg.norm(d, axis=1)
    E = 0.5 * KS * np.sum((l - L0) ** 2)
    f = (KS * (l - L0) / l)[:, None] * d
    g[1:] += f
    g[:-1] -= f
    if kb > 0 and len(Y) > 2:
        u, v = Y[:-2] - Y[1:-1], Y[2:] - Y[1:-1]
        lu2, lv2 = np.sum(u * u, 1), np.sum(v * v, 1)
        luv = np.sqrt(lu2 * lv2)
        cos = np.sum(u * v, 1) / luv
        if bend == "wlc":
            E += kb * np.sum(1 + cos)
            pref = kb * np.ones_like(cos)
        else:
            E += 0.5 * kb * np.sum((1 + cos) ** 2)
            pref = kb * (1 + cos)
        du = (pref / luv)[:, None] * v - (pref * cos / lu2)[:, None] * u
        dv = (pref / luv)[:, None] * u - (pref * cos / lv2)[:, None] * v
        g[:-2] += du
        g[2:] += dv
        g[1:-1] -= du + dv
    w = (np.arange(ns) + 0.5) / ns
    x = Y[:-1, None, :] + w[None, :, None] * d[:, None, :]          # test points (segment, ns, 3)
    r = np.linalg.norm(x, axis=2)
    pen = np.maximum(R + RANGE - r, 0.0)
    ds = L0 / ns
    E += 0.5 * k * ds * np.sum(pen ** 2)
    fx = (k * ds * pen / r)[..., None] * x                           # outward push on the fiber; dE/dx = -fx
    g[:-1] -= np.sum((1 - w)[None, :, None] * fx, axis=1)
    g[1:] -= np.sum(w[None, :, None] * fx, axis=1)
    return E, g, float(np.sum(k * ds * pen))


def chain(kind, a, n_dyn=None):
    """Start positions and fixed-node mask. Static: 'bridge' (a = segments either side) or 'tip' (a = psi in
    degrees). Dynamic (n_dyn segments, nothing fixed): 'bridge' = flat, touching at its middle node;
    'tip' = end-on."""
    if n_dyn:
        if kind == "bridge":
            k = np.arange(n_dyn + 1) - n_dyn // 2
            Y = np.stack([k * L0, 0.0 * k, np.full(len(k), R0 + RANGE)], axis=1)
        else:
            Y = np.outer(R0 + RANGE + np.arange(n_dyn + 1) * L0, [0.0, 0.0, 1.0])
        return Y, np.zeros(len(Y), bool)
    if kind == "bridge":
        k = np.arange(-a, a + 1)
        Y = np.stack([k * L0, 0.0 * k, np.full(len(k), R0 + RANGE)], axis=1)
        fixed = np.zeros(len(Y), bool)
        fixed[[0, -1]] = True
    else:
        psi = np.radians(a)
        t = np.array([np.sin(psi), 0.0, np.cos(psi)])
        Y = np.array([0.0, 0.0, R0 + RANGE]) + np.arange(4)[:, None] * L0 * t[None, :]
        fixed = np.zeros(4, bool)
        fixed[[2, 3]] = True
    return Y, fixed


def run_static(kind, a, bend, kb, rng):
    Y, fixed = chain(kind, a)
    free = ~fixed
    Y[free] += 1e-3 * rng.standard_normal((free.sum(), 3))
    Y0, x, rows = Y.copy(), Y[free].ravel(), []
    for R in np.round(np.arange(R0, R0 + 1.0 + 1e-9, 0.05), 4):
        def fun(z):
            Yz = Y.copy()
            Yz[free] = z.reshape(-1, 3)
            E, g, _ = forces(Yz, R, kb, bend, 500.0, 8)
            return E, g[free].ravel()
        res = minimize(fun, x, jac=True, method="L-BFGS-B",
                       options=dict(maxiter=20000, maxfun=40000, gtol=1e-12, ftol=1e-16, maxcor=30, maxls=50))
        x = res.x
        Yz = Y.copy()
        Yz[free] = x.reshape(-1, 3)
        E, g, F = forces(Yz, R, kb, bend, 500.0, 8)
        l = np.linalg.norm(Yz[1:] - Yz[:-1], axis=1)
        rows.append(dict(mode="static", kind=kind, a=a, bend=bend, kb=kb, seed=0, R=R, delta=round(R - R0, 4),
                         t=np.nan, F=F, strain_max=float(np.max(np.abs(l - L0)) / L0),
                         tip_lateral=float(np.linalg.norm((Yz[0] - Y0[0])[:2])),
                         grad_max=float(np.abs(g[free]).max()), status="ok" if res.success else "maxiter",
                         # the sphere has reached a pinned node: the push-back is then set by the pin, not the fiber
                         pins_overrun=bool(R + RANGE >= np.linalg.norm(Y[fixed], axis=1).min())))
    return rows


def run_dynamic(kind, bend, kb, seed, n=30, v=0.033, t_push=30.0, T=1e-4, dt=0.005):
    rng = np.random.default_rng(seed)
    Y, _ = chain(kind, None, n_dyn=n)
    ns = int(np.ceil(L0 / 0.5))
    n_steps, every = int(round(t_push / dt)), int(round(1.0 / dt))
    rows, acc = [], []
    for s in range(n_steps):
        t = s * dt
        R = R0 + v * t
        _, g, F = forces(Y, R, kb, bend, 50.0, ns)
        acc.append(F)
        Y = Y - dt * g + np.sqrt(2 * T * dt) * rng.standard_normal(Y.shape)
        if (s + 1) % every == 0:                       # mean force over the last time unit
            l = np.linalg.norm(Y[1:] - Y[:-1], axis=1)
            rows.append(dict(mode="dynamic", kind=kind, a=n, bend=bend, kb=kb, seed=seed, R=round(R, 4),
                             delta=round(R - R0, 4), t=round(t + dt, 3), F=float(np.mean(acc)),
                             strain_max=float(np.max(np.abs(l - L0)) / L0),
                             tip_lateral=float(np.linalg.norm(Y[0, :2])) if kind == "tip" else np.nan,
                             grad_max=np.nan, status="ok", pins_overrun=False))
            acc = []
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for bend, kb in BENDS:
        for kind, a in GEOMS:
            rows += run_static(kind, a, bend, kb, np.random.default_rng(1))
        for kind in ("bridge", "tip"):
            for seed in range(5):
                rows += run_dynamic(kind, bend, kb, seed)
    with open(os.path.join(OUT, "single_fiber.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = pd.DataFrame(rows)
    st = d[(d["mode"] == "static") & ~d["pins_overrun"].astype(bool)]
    print("STATIC (no drag): F at push delta = 0.25 / 0.5 / 1.0 ('pinned': the sphere has reached a pinned node); "
          "fails = minimizer not converged")
    for (bend, kb), gb in st.groupby(["bend", "kb"], sort=False):
        for (kind, a), gg in gb.groupby(["kind", "a"], sort=False):
            F = [gg.loc[np.isclose(gg["delta"], x), "F"] for x in (0.25, 0.5, 1.0)]
            print(f"  {bend:8s} kb={kb:<6g} {kind:6s} {a:>3}: "
                  + " / ".join(f"{x.iloc[0]:.2e}" if len(x) else "pinned  " for x in F)
                  + f"   strain_max {gg['strain_max'].iloc[-1]:.3f}  fails {int((gg['status'] != 'ok').sum())}")
    dy = d[d["mode"] == "dynamic"]
    print("DYNAMIC (drag, 5 seeds): mean F over t = 0-10 / 10-20 / 20-30 (sphere speed 0.033)")
    for (bend, kb, kind), gg in dy.groupby(["bend", "kb", "kind"], sort=False):
        m = [gg.loc[(gg["t"] > lo) & (gg["t"] <= lo + 10), "F"].mean() for lo in (0, 10, 20)]
        name = "flat" if kind == "bridge" else "end-on"
        print(f"  {bend:8s} kb={kb:<6g} {name:6s}: " + " / ".join(f"{x:.3f}" for x in m))

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    colors = dict(zip([f"{k} {a}" for k, a in GEOMS], plt.get_cmap("viridis")(np.linspace(0, 0.9, len(GEOMS)))))
    for ax, (bend, kb) in zip(axs[:2], [("original", 0.001), ("wlc", 0.5)]):
        for (kind, a), gg in st[(st["bend"] == bend) & (st["kb"] == kb)].groupby(["kind", "a"], sort=False):
            lab = f"flat, held ±{a} seg" if kind == "bridge" else f"ends at organoid, psi = {a}°"
            ax.semilogy(gg["delta"], np.maximum(gg["F"], 1e-7), "o-", ms=3, color=colors[f"{kind} {a}"], label=lab)
        ax.set_title(f"one fiber, no drag: {bend} bend, kb = {kb:g}")
        ax.set_xlabel("how far the organoid has pushed in, δ")
        ax.set_ylabel("push-back force F on the organoid")
        ax.set_ylim(1e-7, 3)
    axs[0].legend(fontsize=7)
    ax = axs[2]
    for (bend, kb, kind), gg in dy.groupby(["bend", "kb", "kind"], sort=False):
        mm = gg.groupby("t")["F"].mean()
        ls = "-" if kind == "tip" else "--"
        lab = f"{'end-on' if kind == 'tip' else 'flat'}, {bend} kb = {kb:g}"
        ax.plot(mm.index * 0.033, mm.values, ls, label=lab)
    ax.axhline(31 * 0.033, color="grey", lw=0.8, ls=":")
    ax.text(0.02, 31 * 0.033 * 1.02, "drag of the whole fiber (31 nodes x 0.033)", fontsize=7, color="grey")
    ax.set_title("one free 30-segment fiber, with drag (speed 0.033)")
    ax.set_xlabel("how far the organoid has pushed in, δ")
    ax.set_ylabel("push-back force F (mean of 5 noise seeds)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "single_fiber.png"), dpi=130)


if __name__ == "__main__":
    main()
