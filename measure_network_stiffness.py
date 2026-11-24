#!/usr/bin/env python3
"""
measure_network_stiffness.py -- small-strain shear modulus G of the ECM fiber network, in model units.

Energy, the same as the engine (Energy/FiberElasticity.cpp):
    E = sum_segments 1/2 ks (l - l0)^2  +  sum_bends 1/2 kb (cos(theta) + 1)^2
Test: shear the WHOLE periodic box (Lees-Edwards boundary: a bond that crosses the top/bottom
face of the box is shifted sideways by gamma * L), let every fiber node relax to minimum energy,
and use E(gamma) - E(0) = 1/2 G gamma^2 L^3. There are no clamped walls, so G is a bulk property
of the network. The script also prints the "affine" G (every node moved with the shear, no
relaxation), which is the upper limit.

Result for the networks in this repository (ks = 10, l0 = 2.83, kb = 0.001): G is about 1e-6
or less, against an affine G of about 2. The diluted network has on average 3.2 bonds per node,
far below the 6 needed for a stretching-rigid network in 3D, so at small strain it deforms by
bending alone, and bending is almost free with kb = 0.001. The gel therefore has essentially no
shear stiffness at small strain; making it Matrigel-like needs a larger kb (or more bonds).

    python3 measure_network_stiffness.py sample_init_29cells/ECM.topo --kb 0.001 0.1 1
L-BFGS stops after --maxiter steps, so the printed G is an upper bound (it can only go down).
"""
import argparse
import time
import numpy as np
from scipy.optimize import minimize
import growthviz as gv


def build(path, L):
    nodes, _, fibers = gv.read_ecm(path)
    ids = sorted(nodes)
    idx = {i: k for k, i in enumerate(ids)}
    X = np.array([nodes[i] for i in ids])
    bonds = np.array([(idx[a], idx[b]) for f in fibers for a, b in zip(f[:-1], f[1:])])
    bends = np.array([(idx[a], idx[b], idx[c]) for f in fibers for a, b, c in zip(f[:-2], f[1:-1], f[2:])])
    # which periodic image each bond / bend arm crosses into, taken from the unsheared network
    nb = np.round((X[bonds[:, 1]] - X[bonds[:, 0]]) / L)
    nu = np.round((X[bends[:, 0]] - X[bends[:, 1]]) / L)
    nv = np.round((X[bends[:, 2]] - X[bends[:, 1]]) / L)
    return X, bonds, bends, nb, nu, nv


def image_shift(n, gamma, L):
    """Lattice vector of periodic image n in a box sheared by gamma (x moves with z)."""
    s = n * L
    s[:, 0] += gamma * L * n[:, 2]
    return s


def energy_grad(x, gamma, L, bonds, bends, nb, nu, nv, ks, l0, kb):
    Y = x.reshape(-1, 3)
    g = np.zeros_like(Y)
    d = Y[bonds[:, 1]] - Y[bonds[:, 0]] - image_shift(nb, gamma, L)
    l = np.linalg.norm(d, axis=1)
    E = 0.5 * ks * np.sum((l - l0) ** 2)
    f = (ks * (l - l0) / l)[:, None] * d
    np.add.at(g, bonds[:, 1], f)
    np.add.at(g, bonds[:, 0], -f)
    if kb > 0 and len(bends):
        u = Y[bends[:, 0]] - Y[bends[:, 1]] - image_shift(nu, gamma, L)
        v = Y[bends[:, 2]] - Y[bends[:, 1]] - image_shift(nv, gamma, L)
        lu, lv = np.linalg.norm(u, axis=1), np.linalg.norm(v, axis=1)
        cos = np.sum(u * v, axis=1) / (lu * lv)
        E += 0.5 * kb * np.sum((cos + 1) ** 2)
        pref = (kb * (cos + 1))[:, None]
        du = pref * (v / (lu * lv)[:, None] - (cos / lu ** 2)[:, None] * u)
        dv = pref * (u / (lu * lv)[:, None] - (cos / lv ** 2)[:, None] * v)
        np.add.at(g, bends[:, 0], du)
        np.add.at(g, bends[:, 2], dv)
        np.add.at(g, bends[:, 1], -du - dv)
    return E, g.ravel()


def shear_modulus(path, ks=10.0, l0=2.8284271, kb=0.001, L=64.0, gamma=0.01, maxiter=2000):
    X, bonds, bends, nb, nu, nv = build(path, L)
    args = (L, bonds, bends, nb, nu, nv, ks, l0, kb)
    V = L ** 3
    # reference: relax the unsheared network first (an input taken from a run is not at rest)
    r0 = minimize(energy_grad, X.ravel(), args=(0.0,) + args, jac=True, method="L-BFGS-B",
                  options=dict(maxiter=maxiter, gtol=1e-12, ftol=1e-16, maxcor=20))
    X0 = r0.x.reshape(-1, 3)
    Xs = X0.copy()
    Xs[:, 0] += gamma * Xs[:, 2]          # affine start
    E_affine = energy_grad(Xs.ravel(), gamma, *args)[0]
    r1 = minimize(energy_grad, Xs.ravel(), args=(gamma,) + args, jac=True, method="L-BFGS-B",
                  options=dict(maxiter=maxiter, gtol=1e-12, ftol=1e-16, maxcor=20))
    G = 2 * (r1.fun - r0.fun) / (gamma ** 2 * V)
    G_affine = 2 * (E_affine - r0.fun) / (gamma ** 2 * V)
    return G, G_affine, len(X), 2 * len(bonds) / len(X)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ecm", nargs="?", default="sample_init_29cells/ECM.topo")
    ap.add_argument("--ks", type=float, default=10.0)
    ap.add_argument("--kb", type=float, nargs="+", default=[0.001])
    ap.add_argument("--l0", type=float, default=2.8284271)
    ap.add_argument("--box", type=float, default=64.0, help="periodic box size (conf 'box', 64)")
    ap.add_argument("--maxiter", type=int, default=2000)
    a = ap.parse_args()
    for kb in a.kb:
        tic = time.time()
        G, Ga, n, z = shear_modulus(a.ecm, a.ks, a.l0, kb, a.box, maxiter=a.maxiter)
        print(f"ks={a.ks:g} kb={kb:g}: G <= {G:.3g}   (affine G {Ga:.3g}; {n} nodes, "
              f"{z:.2f} bonds per node; {time.time() - tic:.0f}s)", flush=True)
