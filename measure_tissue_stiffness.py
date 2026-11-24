#!/usr/bin/env python3
"""
measure_tissue_stiffness.py -- ROUGH estimate of the shear modulus G of the cell tissue, in model units.

Energy, close to the engine's for real cells: E = sum_cells kv (V - V0)^2 + (S - S0)^2, with
every cell in the file (the medium "empty" cells too) treated as a real cell with V0 = 1 and
S0 = shape_index. Differences from the engine: face centres are plain corner means (the engine
weights by edge length), and the topology is held fixed (no neighbour swaps).
Test: relax with the outer vertices clamped, then shear by +gamma and by -gamma (x += gamma z,
outer vertices held at the sheared positions) and relax again;
    G = (E(+gamma) + E(-gamma) - 2 E(0)) / (gamma^2 V),
which removes the part linear in gamma that the leftover stress of the clamped reference gives.

LIMITATION: the clamped boundary is part of the result. On the files we tried the estimate
ranged from about 0.9 to 3.8 depending on the input, so it is not yet a material property.
A periodic tissue would be needed for a proper bulk value; do not convert it to pascals.

    python3 measure_tissue_stiffness.py sample_init_29cells/sample.topo
"""
import argparse
import numpy as np
from scipy.optimize import minimize
import growthviz as gv


def build(path):
    V, E, P, C = gv.read_topo(path)
    vids = sorted(V)
    idx = {v: i for i, v in enumerate(vids)}
    X = np.array([V[v] for v in vids])
    pc = gv.polygon_cells(C)
    # outer vertices: on faces that belong to only one cell
    outer = np.zeros(len(X), bool)
    loops = {p: [idx[v] for v in gv.polygon_loop(P[p], E)] for p in P if p in pc}
    for p, owners in pc.items():
        if len(owners) == 1:
            outer[loops[p]] = True
    cells = list(C)
    tri = []  # (cell index, a, b, polygon id) with outward orientation sign
    poly_ids = sorted(loops)
    pidx = {p: i for i, p in enumerate(poly_ids)}
    for ci, c in enumerate(cells):
        cen = X[[v for p in C[c][0] for v in loops[p]]].mean(axis=0)
        for p in C[c][0]:
            L = loops[p]
            pcen = X[L].mean(axis=0)
            # one orientation per (cell, face): from the face's total vector area, not per triangle
            area_vec = sum(np.cross(X[L[i]] - pcen, X[L[(i + 1) % len(L)]] - pcen) for i in range(len(L)))
            s = np.sign(np.dot(area_vec, pcen - cen))
            for i in range(len(L)):
                tri.append((ci, L[i], L[(i + 1) % len(L)], pidx[p], s))
    tri = np.array(tri)
    members = [np.array(loops[p]) for p in poly_ids]
    return X, outer, len(cells), tri, members


def energy_grad(x, Xfix, free, nc, tri, members, kv, v0, s0):
    X = Xfix.copy()
    X[free] = x.reshape(-1, 3)
    pcen = np.array([X[m].mean(axis=0) for m in members])
    ci, a, b, pi, s = tri[:, 0].astype(int), tri[:, 1].astype(int), tri[:, 2].astype(int), tri[:, 3].astype(int), tri[:, 4]
    A, B, Cc = X[a], X[b], pcen[pi]
    cr = np.cross(A - Cc, B - Cc)
    vol_t = s * np.einsum("ij,ij->i", A, np.cross(B, Cc)) / 6.0
    area_t = 0.5 * np.linalg.norm(cr, axis=1)
    Vc = np.bincount(ci, vol_t, nc)
    Sc = np.bincount(ci, area_t, nc)
    E = np.sum(kv * (Vc - v0) ** 2 + (Sc - s0) ** 2)
    dV = (2 * kv * (Vc - v0))[ci][:, None] * s[:, None] / 6.0
    dS = (2 * (Sc - s0))[ci][:, None]
    gA = dV * np.cross(B, Cc)
    gB = dV * np.cross(Cc, A)
    gC = dV * np.cross(A, B)
    n = cr / (2 * area_t[:, None] + 1e-30)  # unit normal
    gA += dS * 0.5 * np.cross(B - Cc, n)
    gB += dS * 0.5 * np.cross(Cc - A, n)
    gC += dS * 0.5 * np.cross(A - B, n)
    g = np.zeros_like(X)
    np.add.at(g, a, gA)
    np.add.at(g, b, gB)
    gp = np.zeros_like(pcen)
    np.add.at(gp, pi, gC)
    for k, m in enumerate(members):
        g[m] += gp[k] / len(m)
    return E, g[free].ravel()


def relax(X, free, *args):
    r = minimize(energy_grad, X[free].ravel(), args=(X, free) + args, jac=True, method="L-BFGS-B",
                 options=dict(maxiter=50000, gtol=1e-9))
    Y = X.copy()
    Y[free] = r.x.reshape(-1, 3)
    return Y, r.fun, r.success


def shear_modulus(path, kv=10.0, v0=1.0, shape_index=5.2, gamma=0.01):
    X, outer, nc, tri, members = build(path)
    free = ~outer
    s0 = shape_index * v0 ** (2.0 / 3.0)
    args = (nc, tri, members, kv, v0, s0)
    X0, E0, ok0 = relax(X, free, *args)
    energies, ok = [], ok0
    for g in (gamma, -gamma):
        Xs = X0.copy()
        Xs[:, 0] += g * Xs[:, 2]
        X1, E1, ok1 = relax(Xs, free, *args)
        energies.append(E1)
        ok = ok and ok1
    pcen = np.array([X0[m].mean(axis=0) for m in members])
    s = tri[:, 4]
    a, b, pi = tri[:, 1].astype(int), tri[:, 2].astype(int), tri[:, 3].astype(int)
    Vtot = np.sum(s * np.einsum("ij,ij->i", X0[a], np.cross(X0[b], pcen[pi])) / 6.0)
    return (energies[0] + energies[1] - 2 * E0) / (gamma ** 2 * Vtot), nc, ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("topo")
    ap.add_argument("--shape-index", type=float, nargs="+", default=[5.2])
    a = ap.parse_args()
    for si in a.shape_index:
        G, nc, ok = shear_modulus(a.topo, shape_index=si)
        print(f"shape index {si}: tissue G about {G:.3g} (model units, rough; see docstring), cells = {nc}, converged = {ok}")
