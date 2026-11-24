"""Which force makes the fibers wrap around the growing tissue?

Four runs from t = 0 on the current engine (Tao's face-centre forces, seeded noise, full-precision restarts),
p = 1.0 gel seed 7, growth twice as fast as before (doubling time 10), t = 0-80:
    all_on         contact + linkers with rest length 1.5 (they attach stretched and pull)
    passive_link   linkers with rest length = their length when attaching (no built-in pull)
    no_link_force  linkers attach but have zero stiffness
    no_contact     linkers as all_on, no cell-fiber contact force
Alignment S = mean (3 cos^2 - 1)/2 of segment vs radial direction from the tissue centre to the segment
midpoint (-0.5 = wrapped, 0 = random, +1 = radial).
  shell S      all segments with midpoints between R and R+2 at that time (the earlier video measure)
  tracked S    the same segments, those in the R to R+2 shell at t_ref, followed over time
Writes results/alignment_mechanism.csv and .png, and (with --video) a 4-panel video.
    python3 experiments/jen_followup/alignment_mechanism.py [--t-ref 20] [--video]
"""
import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np

import growthviz as gv

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "results"
CASES = [("all_on", "all forces (linkers pull)", "#ffffff"),
         ("passive_link", "passive linkers", "#ff8b70"),
         ("no_link_force", "no linker force", "#70b4ff"),
         ("no_contact", "no contact force", "#69e2cd")]


def run_dir(case):
    return ROOT / "runs" / f"tao_{case}_p1_s7"


def segments(ecm_path):
    nodes, _, fibers = gv.read_ecm(ecm_path)
    pairs = sorted({(min(a, b), max(a, b)) for f in fibers for a, b in zip(f, f[1:])})
    a = np.array([nodes[i] for i, _ in pairs])
    b = np.array([nodes[j] for _, j in pairs])
    return {p: k for k, p in enumerate(pairs)}, a, b


def S_of(a, b, center):
    d = b - a
    m = 0.5 * (a + b) - center
    ld, lm = np.linalg.norm(d, axis=1), np.linalg.norm(m, axis=1)
    ok = (ld > 1e-12) & (ld < 10)                      # skip segments that cross the periodic box
    c = np.einsum("ij,ij->i", d, m) / (ld * lm + 1e-30)
    return ok, lm, (3 * c ** 2 - 1) / 2


def analyse(case, t_ref):
    folder = run_dir(case)
    frames = {round(f["t"]): f for f in pickle.load(open(folder / "frames.pkl", "rb"))}
    snaps = sorted(int(p.name[4:]) for p in folder.glob("seg_*") if (p / f"{p.name[4:]}.ECM.topo").exists())
    rows, cohort = [], None
    for t in snaps:
        fr = frames[t]
        center = np.asarray(fr.get("center_vol", fr["center"]))
        R = fr["surface_radius"]
        ids, a, b = segments(folder / f"seg_{t:07d}" / f"{t:07d}.ECM.topo")
        ok, r, s = S_of(a, b, center)
        shell = ok & (r >= R) & (r < R + 2)
        if t == t_ref:
            cohort = [p for p, k in ids.items() if shell[k]]
        tracked = np.nan
        if cohort is not None:
            k = np.array([ids[p] for p in cohort if p in ids])
            k = k[ok[k]]
            tracked = float(s[k].mean())
        rows.append(dict(case=case, t=t, R=R, n_cells=fr["n_cells"], shell_S=float(s[shell].mean()),
                         n_shell=int(shell.sum()), tracked_S=tracked, n_tracked=len(cohort or [])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t-ref", type=int, default=20)
    ap.add_argument("--video", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    rows = [r for case, _, _ in CASES for r in analyse(case, a.t_ref)]
    with open(OUT / "alignment_mechanism.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    by = {c: [r for r in rows if r["case"] == c] for c, _, _ in CASES}
    print(f"tracked fibers: those in the R to R+2 shell at t = {a.t_ref}")
    print(f"{'run':15s} {'R':>5s} {'cells':>6s} {'shell S':>8s} {'tracked S':>10s}   (at the last time; tracked S at t_ref first)")
    for c, lab, _ in CASES:
        last, ref = by[c][-1], next(r for r in by[c] if r["t"] == a.t_ref)
        print(f"{c:15s} {last['R']:5.2f} {last['n_cells']:6d} {last['shell_S']:8.3f} "
              f"{ref['tracked_S']:7.3f} -> {last['tracked_S']:.3f}  (N = {ref['n_tracked']})")
    plt = gv._style()
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    for c, lab, col in CASES:
        d = by[c]
        t = [r["t"] for r in d]
        axs[0].plot(t, [r["n_cells"] for r in d], color=col, label=lab)
        axs[1].plot(t, [r["shell_S"] for r in d], color=col, label=lab)
        axs[2].plot([r["t"] for r in d if r["t"] >= a.t_ref], [r["tracked_S"] for r in d if r["t"] >= a.t_ref],
                    color=col, marker=".", label=lab)
    for ax, title in zip(axs, ("number of cells (growth)", "alignment of fibers near the tissue (R to R+2)",
                               f"alignment of the same fibers, tracked from t = {a.t_ref}")):
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xlabel("simulation time")
    axs[1].set_ylabel("S  (-0.5 = wrapped around the tissue, 0 = random)")
    axs[0].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "alignment_mechanism.png", dpi=120)
    print("wrote", OUT / "alignment_mechanism.png")
    if a.video:
        video(by, a.t_ref)


def video(by, t_ref):
    from matplotlib.animation import FFMpegWriter
    plt = gv._style()
    plt.rcParams["animation.ffmpeg_args"] = ["-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    gv.P_LIM = (0, 6)
    frames = {c: {round(f["t"]): f for f in pickle.load(open(run_dir(c) / "frames.pkl", "rb"))} for c, _, _ in CASES}
    t_end = min(max(f) for f in frames.values())
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    clock = fig.text(0.5, 0.955, "", fontsize=22, ha="center")
    views = [fig.add_axes([0.005 + 0.25 * k, 0.42, 0.24, 0.47], projection="3d") for k in range(4)]
    info = []
    for k, (c, lab, col) in enumerate(CASES):
        fig.text(0.125 + 0.25 * k, 0.905, lab, fontsize=16, ha="center", color=col)
        info.append(fig.text(0.125 + 0.25 * k, 0.88, "", fontsize=11, ha="center", color="#e6e6ee"))
    fig.text(0.5, 0.405, "Cells coloured by pressure (brighter = more squeezed). Yellow lines are linkers. "
             "Same gel, same start; growth doubling time 10.", fontsize=11, ha="center", color="#c8c8d0")
    ax_s = fig.add_axes([0.06, 0.08, 0.40, 0.27])
    ax_t = fig.add_axes([0.55, 0.08, 0.40, 0.27])
    for ax, title in ((ax_s, "Fibers near the tissue (R to R+2): alignment"),
                      (ax_t, f"The same fibers, tracked from t = {t_ref}: alignment")):
        ax.set_title(title, loc="left", fontsize=13)
        ax.set_xlim(0, t_end)
        ax.set_xlabel("simulation time")
        ax.set_ylabel("S  (-0.5 = wrapped, 0 = random)")
        ax.grid(axis="y", color="#31313a", lw=0.6)
    vals_s = [r["shell_S"] for c in by for r in by[c]]
    vals_t = [r["tracked_S"] for c in by for r in by[c] if r["t"] >= t_ref]
    ax_s.set_ylim(min(vals_s) - 0.02, max(vals_s) + 0.02)
    ax_t.set_ylim(min(vals_t) - 0.02, max(vals_t) + 0.02)
    lines_s = [ax_s.plot([], [], color=col, lw=2.2)[0] for _, _, col in CASES]
    lines_t = [ax_t.plot([], [], color=col, lw=2.2, marker=".")[0] for _, _, col in CASES]
    half = max(f["radius"] for c in frames for f in frames[c].values()) + 2.15
    out = OUT / "alignment_mechanism_4runs.mp4"
    writer = FFMpegWriter(fps=8, codec="h264", bitrate=8000)
    with writer.saving(fig, str(out), dpi=100):
        for t in range(1, t_end + 1):
            clock.set_text(f"t = {t}")
            for k, (c, _, _) in enumerate(CASES):
                fr = frames[c][t]
                views[k].cla()
                gv.draw_3d(views[k], fr, half)
                info[k].set_text(f"{fr['n_cells']} cells  ·  {fr['n_links']} linkers")
                d = [r for r in by[c] if r["t"] <= t]
                lines_s[k].set_data([r["t"] for r in d], [r["shell_S"] for r in d])
                dt = [r for r in d if r["t"] >= t_ref]
                lines_t[k].set_data([r["t"] for r in dt], [r["tracked_S"] for r in dt])
            for _ in range(10 if t == t_end else 1):
                writer.grab_frame()
    fig.savefig(OUT / "alignment_mechanism_4runs_last.png", dpi=60)
    print("wrote", out)


if __name__ == "__main__":
    main()
