"""Video: fixed vs passive linkers, t = 80 to 94 (p = 1.0, seed 7, same checkpoint).

Adapted from the earlier renderer for runs/jen_combined_p1_s7/baseline_vs_combined_t80_to_94.mp4 (same 3D view,
pressure colours and division-rate panels). The density panel is replaced by the mean linker tension
ks (length - rest), since that is what passive linkers change.
    python3 experiments/jen_followup/render_linker_video.py
"""
import json
import math
import os
import pickle
from pathlib import Path
from statistics import mean

import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D

import growthviz as gv

start, end = 80, int(os.environ.get("COMPARE_END", 94))
folders = (Path("runs/jen_baseline_v2_p1_s7"), Path("runs/jen_passive_link_p1_s7"))
out = folders[1] / f"fixed_vs_passive_linkers_t{start}_to_{end}.mp4"
batch_color = ("#70b4ff", "#ff8b70")
division_color = ("#ffd166", "#69e2cd")
labels = ("Fixed linkers", "Passive linkers")
edit_labels = ("Rest length 1.5: linkers attach stretched and pull",
               "Rest length = length when attaching: no built-in pull")
KS = json.loads((folders[0] / "config.json").read_text())["link"][1]
COHORT = "Same 239 fibers tracked from t = 80, alignment at t = 94:  fixed -0.258  ·  passive -0.203  (start -0.175)"


def link_tension(fr):
    strain = np.asarray(fr["link_strain"], float)
    rest = fr.get("link_rest_each")
    rest = np.asarray(rest, float) if rest is not None and len(rest) == len(strain) else float(fr.get("link_rest", 1.5))   # frames before Nov 19 had the fixed 1.5
    return float(np.mean(KS * rest * strain)) if len(strain) else 0.0


def load(folder):
    with (folder / "frames.pkl").open("rb") as f:
        saved = pickle.load(f)
    frames = [frame for frame in saved if 71 <= frame["t"] <= end]
    assert [fr["t"] for fr in frames] == list(range(71, end + 1))
    next_frame = next((fr for fr in saved if fr["t"] == end + 1), None)
    state = json.loads((folder / "state.json").read_text())
    final_ids = {int(cid) for cid in state["V0"]} if next_frame is None else set(next_frame["cells"])
    counts, stats = [], []
    for j, fr in enumerate(frames):
        ids = set(fr["cells"])
        surface = {cid for cid, _ in fr["surf"]}
        later = set(frames[j + 1]["cells"]) if j + 1 < len(frames) else final_ids
        divided = ids - later
        counts.append((len(surface), len(ids - surface), len(surface & divided), len((ids - surface) & divided)))
        stats.append((fr["order_near"], link_tension(fr)))
    counts = np.asarray(counts)
    rates = np.full((len(counts), 2), np.nan)
    for j in range(9, len(counts)):
        exposure = counts[j - 9:j + 1].sum(axis=0)
        rates[j] = 100 * exposure[2:] / exposure[:2]
    return frames[9:], np.asarray(stats)[9:], rates[9:]


runs = tuple(load(path) for path in folders)
assert runs[0][0][0]["cells"] == runs[1][0][0]["cells"]   # shared checkpoint at t = 80
plt = gv._style()
plt.rcParams["animation.ffmpeg_args"] = ["-pix_fmt", "yuv420p", "-movflags", "+faststart"]
gv.P_LIM = (0, 6)
fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
clock = fig.text(0.5, 0.961, "", fontsize=20, ha="center")
views = [fig.add_axes([left, 0.467, 0.485, 0.399], projection="3d") for left in (0.01, 0.505)]
view_data = []
for k, cx in enumerate((0.2525, 0.7475)):
    fig.text(cx, 0.929, labels[k], fontsize=17, ha="center", color=batch_color[k], weight="medium")
    fig.text(cx, 0.906, edit_labels[k], fontsize=11, ha="center", color="#d1d1dd")
    view_data.append(fig.text(cx, 0.882, "", fontsize=11, ha="center", color="#e6e6ee"))
cb = fig.colorbar(ScalarMappable(plt.Normalize(*gv.P_LIM), "magma"),
                  cax=fig.add_axes([0.33, 0.449, 0.34, 0.013]), orientation="horizontal", extend="min")
cb.set_label("Cell pressure P  ·  brighter cells are more compressed", fontsize=11, color="#dedee5")
cb.ax.tick_params(colors="#c6c6cf", labelsize=10)
fig.text(0.5, 0.398, "Yellow lines are linkers.", fontsize=10, ha="center", color="#c8c8d0")
cohort_text = fig.text(0.5, 0.374, "", fontsize=12, ha="center", color="#f2f2f6")

fiber_axes = [fig.add_axes([0.05 + 0.322 * k, 0.093, 0.268, 0.234]) for k in range(2)]
rate_axes = [fig.add_axes([0.694, bottom, 0.268, 0.101]) for bottom in (0.215, 0.093)]
for ax, title in zip(fiber_axes, ("Fiber alignment  ·  R to R+2  (lower = more wrapped)",
                                  "Mean linker tension  ks (length - rest)"), strict=True):
    ax.set_title(title, loc="left", fontsize=13, pad=9)
for ax in [*fiber_axes, *rate_axes]:
    ax.set(xlim=(start, end), xticks=list(range(start, end + 1, 2)))
    ax.set_xlabel("Simulation time", fontsize=10, color="#ccccd4", labelpad=3)
    ax.set_facecolor("#17171c")
    ax.tick_params(colors="#c8c8d0", labelsize=10, length=3)
    ax.grid(axis="y", color="#31313a", lw=0.65)
    for spine in ax.spines.values():
        spine.set_color("#595963")
for j, ax in enumerate(fiber_axes):
    vals = np.concatenate([run[1][:, j] for run in runs])
    pad = 0.08 * (vals.max() - vals.min() + 1e-9)
    ax.set_ylim(min(vals.min(), 0) - pad if j == 1 else vals.min() - pad, vals.max() + pad)
fiber_axes[1].axhline(0, color="#8a8a94", lw=0.8)
rate_max = 1.18 * max(np.nanmax(run[2]) for run in runs)
for k, ax in enumerate(rate_axes):
    ax.set_ylim(0, rate_max)
    ax.text(0.985, 0.87, labels[k], transform=ax.transAxes, ha="right", va="top", color=batch_color[k], fontsize=10,
            bbox={"facecolor": "#17171c", "edgecolor": "none", "pad": 1.2})
rate_axes[0].tick_params(labelbottom=False)
rate_axes[0].set_xlabel("")
fig.text(0.694, 0.361, "Divisions per 100 cells per unit time", ha="left", fontsize=13)
fig.legend(handles=[Line2D([0], [0], color=division_color[0], lw=2.5, label="Surface"),
                    Line2D([0], [0], color=division_color[1], lw=2.5, ls="--", label="Interior")],
           loc="lower left", bbox_to_anchor=(0.694, 0.318), borderaxespad=0, ncol=2, frameon=False, fontsize=10)
fig.text(0.823, 0.035, "Division rates use the last 10 time units.", fontsize=10, ha="center", color="#c8c8d0")
fiber_lines = [[ax.plot([], [], color=c, lw=2.4, marker=".", ms=6)[0] for c in batch_color] for ax in fiber_axes]
rate_lines = [[ax.plot([], [], color=c, lw=2.2, ls=ls, marker=".", ms=5)[0]
               for c, ls in zip(division_color, ("-", "--"), strict=True)] for ax in rate_axes]

half_width = max(fr["radius"] for run in runs for fr in run[0]) + 2.15
time = np.arange(start, end + 1)
writer = FFMpegWriter(fps=6, codec="h264", bitrate=7500)
with writer.saving(fig, str(out), dpi=100):
    for i, t in enumerate(time):
        clock.set_text(f"t = {t}" + ("   ·   shared checkpoint" if t == start else ""))
        cohort_text.set_text(COHORT if t == end else "")
        for k, (frames, stats, rates) in enumerate(runs):
            ax = views[k]
            ax.cla()
            fr = frames[i]
            gv.draw_3d(ax, fr, half_width)
            volume = sum(cell["V"] for cell in fr["cells"].values())
            surface = {cid for cid, _ in fr["surf"]}
            pressure = mean(fr["cells"][cid]["P"] for cid in surface)
            view_data[k].set_text(f"{fr['n_cells']:,} cells  ·  tissue volume {volume:.1f}  ·  "
                                  f"mean surface pressure {pressure:.2f}")
            for j in range(2):
                fiber_lines[j][k].set_data(time[:i + 1], stats[:i + 1, j])
                rate_lines[k][j].set_data(time[:i + 1], rates[:i + 1, j])
        if t == end:
            fig.savefig(out.with_suffix(".png"), dpi=60)
        for _ in range(12 if t in (start, end) else 3):
            writer.grab_frame()
print(out, flush=True)
plt.close(fig)
