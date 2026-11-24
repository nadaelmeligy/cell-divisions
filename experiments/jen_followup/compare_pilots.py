"""Read the one-change-at-a-time continuations of the seed-7, p=1.0 checkpoint."""
import csv
import pickle
import sys

import numpy as np

from compare_alignment import ROOT, gv, projection


START = 80
CASES = ("baseline", "no_link_force", "fast_growth", "no_contact")


def read_metrics(folder):
    with (folder / "metrics.csv").open() as file:
        return {int(float(row["t"])): {key: float(row[key]) for key in (
            "n_cells", "n_links", "n_div", "mean_V", "mean_P", "order_near",
            "fiber_points_inside", "E_link", "surface_radius",
        )} for row in csv.DictReader(file)}


def total_volume(row):
    return row["mean_V"] * row["n_cells"]


def read_segments(path):
    nodes, _, fibers = gv.read_ecm(path)
    pairs = np.array([(min(a, b), max(a, b)) for f in fibers for a, b in zip(f, f[1:])])
    pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
    a, b = (np.array([nodes[i] for i in pairs[:, j]]) for j in (0, 1))
    return pairs, a, b


def fixed_cohort(start_file, end_file, start_frame, end_frame):
    id0, a0, b0 = read_segments(start_file)
    id1, a1, b1 = read_segments(end_file)
    np.testing.assert_array_equal(id0, id1)
    d0, d1 = b0 - a0, b1 - a1
    len0, len1 = np.linalg.norm(d0, axis=1), np.linalg.norm(d1, axis=1)
    keep = (len0 > 1e-12) & (len1 > 1e-12) & (len0 < 10) & (len1 < 10)
    a0, b0, a1, b1 = (x[keep] for x in (a0, b0, a1, b1))
    u0, u1 = d0[keep] / len0[keep, None], d1[keep] / len1[keep, None]
    r0, first = projection(a0, b0, u0, start_frame["center"], midpoint=True)
    _, rotated = projection(a0, b0, u1, start_frame["center"], midpoint=True)
    r1, final = projection(a1, b1, u1, end_frame["center"], midpoint=True)
    radius = start_frame["surface_radius"]
    cohort = (r0 >= radius) & (r0 < radius + 2)
    radius = end_frame["surface_radius"]
    retained = (r1 >= radius) & (r1 < radius + 2)
    return (1.5 * first[cohort].mean() - 0.5, 1.5 * rotated[cohort].mean() - 0.5,
            1.5 * final[cohort].mean() - 0.5, np.mean(retained[cohort]), cohort.sum())


def frame_at(run, time):
    with (run / "frames.pkl").open("rb") as file:
        return next(frame for frame in pickle.load(file) if frame["t"] == time)


if __name__ == "__main__":
    end = int(sys.argv[1]) if len(sys.argv) > 1 else 94
    cases = sys.argv[2:] or list(CASES)   # the first case is the reference for the matched volume
    runs = {case: ROOT / "runs" / f"jen_{case}_p1_s7" for case in cases}
    metrics = {case: read_metrics(run) for case, run in runs.items()}
    baseline_volume = total_volume(metrics[cases[0]][end])
    start_dir = ROOT / "runs" / "ecm_p1.0_s7"
    start_source = start_dir / "seg_0000080" / "0000080.ECM.topo"
    start_frame = frame_at(start_dir, START)
    for case, data in metrics.items():
        row = data[end]
        divisions = sum(data[t]["n_div"] for t in range(START + 1, end + 1))
        match = min(range(START, end + 1), key=lambda t: abs(total_volume(data[t]) - baseline_volume))
        end_source = runs[case] / f"seg_{end:07d}" / f"{end:07d}.ECM.topo"
        cohort = fixed_cohort(start_source, end_source, start_frame, frame_at(runs[case], end))
        print(f"{case:14s} t={end} cells={row['n_cells']:.0f} divisions={divisions:.0f} "
              f"links={row['n_links']:.0f} V={total_volume(row):.2f} P={row['mean_P']:.3f} S={row['order_near']:.4f} "
              f"inside_samples={row['fiber_points_inside']:.0f} E_link={row['E_link']:.2f}; "
              f"closest to baseline final V at t={match} "
              f"(V={total_volume(data[match]):.2f}, S={data[match]['order_near']:.4f}); "
              f"fixed cohort N={cohort[4]} S(start, rotation only, final)={cohort[0]:.4f}, "
              f"{cohort[1]:.4f}, {cohort[2]:.4f}, {cohort[3]:.1%} stay in shell")
