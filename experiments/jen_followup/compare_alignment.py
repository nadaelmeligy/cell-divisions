"""Compare radial fiber alignment on identical, saved simulation snapshots."""
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import growthviz as gv


def segment_fields(ecm_path):
    nodes, _, fibers = gv.read_ecm(ecm_path)
    pairs = [(a, b) for fiber in fibers for a, b in zip(fiber, fiber[1:])]
    a, b = (np.array([nodes[pair[j]] for pair in pairs]) for j in (0, 1))
    delta = b - a
    lengths = np.linalg.norm(delta, axis=1)
    keep = (lengths > 1e-12) & (lengths < 10)
    return a[keep], b[keep], delta[keep] / lengths[keep, None], lengths[keep]


def original_center(topo_path):
    """Recreate the cellCenter.txt origin used by Shabeeb, from the saved mesh."""
    vertices, edges, faces, cells = gv.read_topo(topo_path)
    centers = []
    for face_ids, cell_type in cells.values():
        if cell_type > 0:
            ids = {v for face_id in face_ids for edge_id in faces[face_id] for v in edges[edge_id]}
            points = np.array([vertices[v] for v in sorted(ids)])
            offset = points - points[0]
            offset -= 64 * np.floor(offset / 64 + 0.5)
            center = points[0] + offset.mean(axis=0)
            centers.append(center - 64 * np.floor(center / 64 + 0.5))
    return np.asarray(centers).mean(axis=0)


def projection(a, b, direction, center, *, midpoint):
    if midpoint:
        position = (a + b) / 2
    else:
        nearer = np.linalg.norm(a - center, axis=1) <= np.linalg.norm(b - center, axis=1)
        position = np.where(nearer[:, None], a, b)
    radial = position - center
    r = np.linalg.norm(radial, axis=1)
    cos2 = (np.einsum("ij,ij->i", radial, direction) / r) ** 2
    return r, cos2


def measure(r, cos2, radius, lengths=None):
    in_shell = (r >= radius) & (r < radius + 2)
    if not in_shell.any():
        return np.nan, 0
    weights = None if lengths is None else lengths[in_shell]
    return 1.5 * np.average(cos2[in_shell], weights=weights) - 0.5, in_shell.sum()


def compare(run, time):
    with (run / "frames.pkl").open("rb") as f:
        frame = next(fr for fr in pickle.load(f) if fr["t"] == time)
    tag = f"{time:07d}"
    folder = run / f"seg_{tag}"
    a, b, u, lengths = segment_fields(folder / f"{tag}.ECM.topo")
    center, radius = frame["center"], frame["surface_radius"]
    old_center = original_center(folder / f"{tag}.sample.topo")
    mid_r, mid_cos2 = projection(a, b, u, center, midpoint=True)
    end_r, end_cos2 = projection(a, b, u, center, midpoint=False)
    old_r, old_cos2 = projection(a, b, u, old_center, midpoint=False)
    video, n_mid = measure(mid_r, mid_cos2, radius)
    weighted_mid, _ = measure(mid_r, mid_cos2, radius, lengths)
    end_same_set, _ = measure(mid_r, end_cos2, radius, lengths)
    end_same_center, n_end = measure(end_r, end_cos2, radius, lengths)
    reconstructed, n_old = measure(old_r, old_cos2, radius, lengths)
    np.testing.assert_allclose(video, frame["order_near"], atol=2e-6, rtol=0)
    # Original default absolute bins assign an edge to the nearest supplied radius.
    bins = np.linspace(5, 29, 8)
    bin_index = np.argmin(np.abs(old_r[:, None] - bins), axis=1)
    nearest_shell = np.argmin(np.abs(bins - (radius + 1)))
    original_bin = (bin_index == nearest_shell) & (old_r >= bins[0]) & (old_r <= bins[-1])
    original_s = (1.5 * np.average(old_cos2[original_bin], weights=lengths[original_bin]) - 0.5
                  if original_bin.any() else np.nan)
    return (video, weighted_mid, end_same_set, end_same_center, reconstructed,
            n_mid, n_end, n_old, radius, bins[nearest_shell], original_s)


if __name__ == "__main__":
    print("run,t,midpoint_equal,midpoint_length_weighted,endpoint_angle_same_segments,"
          "endpoint_selection_same_center,shabeeb_same_radial_bounds,N_midpoint,N_endpoint,"
          "N_shabeeb,R,shabeeb_nearest_default_bin_radius,shabeeb_nearest_default_bin_S")
    for p in ("1.0", "0.9", "0.8", "0.7"):
        for time in ((20, 40, 60, 80, 130, 160) if p in ("1.0", "0.7") else (80,)):
            prefix = "extended_" if time > 130 else "long_" if time > 80 else ""
            run = ROOT / "runs" / f"{prefix}ecm_p{p}_s7"
            print(",".join([f"p{p}", str(time), *(f"{value:.6g}" for value in compare(run, time))]))
