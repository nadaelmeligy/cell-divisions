#!/usr/bin/env python3
"""Render the +x bud and its round control from the relaxed simulation states."""

import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from bud_cavity import AXES, INP, OUTPUT, SUPPORT_COS, TIP_COS, BudGel, ce, distance, measure, profile, relax


TIP = "#af411f"
FLANKS = "#166b83"
INK = "#142d3a"
PAPER = "#f5f7f8"
FLAT = "#879da7"


def fiber_slice(gel, coords, half_width=1.0):
    a, b = (coords[gel.bonds[:, index]] for index in (0, 1))
    delta = b - a
    sloped = np.abs(delta[:, 2]) > 1e-12
    clipped = np.column_stack([np.zeros(len(a)), np.ones(len(a))])
    crossing = (np.array([-half_width, half_width]) - a[sloped, 2, None]) / delta[sloped, 2, None]
    clipped[sloped, 0] = np.maximum(0, crossing.min(axis=1))
    clipped[sloped, 1] = np.minimum(1, crossing.max(axis=1))
    keep = (clipped[:, 1] > clipped[:, 0]) & (sloped | (np.abs(a[:, 2]) <= half_width))
    return (a[keep, None, :] + clipped[keep, :, None] * delta[keep, None, :])[:, :, :2]


def draw_slice(ax, gel, x, height, heading):
    coords = gel.full(x)
    ax.add_collection(LineCollection(fiber_slice(gel, coords), colors=FLAT, linewidths=0.8, alpha=0.65))
    theta = np.linspace(-np.pi, np.pi, 721)
    mu = np.cos(theta)
    radius, _, _ = profile(mu, gel.R, height)
    px, py = radius * mu, radius * np.sin(theta)
    ax.fill(px, py, color="#e3edf0", zorder=2)
    ax.plot(px, py, color=INK, lw=1.6, zorder=3)
    ax.plot(px, np.where(mu >= TIP_COS, py, np.nan), color=TIP, lw=3.2, zorder=4)
    on_flank = (mu >= SUPPORT_COS) & (mu < TIP_COS)
    ax.plot(px, np.where(on_flank, py, np.nan), color=FLANKS, lw=2.8, zorder=4)

    gel.select_samples(coords, gel.R + height)
    i, j = gel.bonds[gel.s_seg].T
    points = coords[i] + gel.s_w[:, None] * (coords[j] - coords[i])
    d, grad, _ = distance(points, gel.axis, gel.R, height)
    touching = (d < ce.RANGE) & (np.abs(points[:, 2]) <= 1.0)
    force = gel.kw * gel.s_ds * (ce.RANGE - d[touching]) * np.linalg.norm(grad[touching], axis=1)
    ax.scatter(*points[touching, :2].T, s=8 + 15 * force, color="#603d2f", edgecolors="white",
               linewidths=0.5, zorder=5)
    ax.set(xlim=(-7.7, 7.7), ylim=(-6.5, 6.5), aspect="equal")
    ax.set_title(heading, loc="left", color=INK, fontsize=18, fontweight="bold", pad=9)
    ax.set_axis_off()


def draw_load(ax, pressures, max_pressure):
    for y, (region, color) in enumerate((("flanks", FLANKS), ("tip", TIP))):
        p = pressures[region]
        ax.barh(y, max_pressure, height=0.40, color="#e4e9ec")
        ax.barh(y, p, height=0.40, color=color)
        ax.text(-0.026 * max_pressure, y, region.capitalize(), ha="right", va="center", color=INK, fontsize=12)
        ax.text(p + 0.025 * max_pressure, y, f"{p:.2f}", va="center", color=INK, fontsize=12)
    ax.set(xlim=(-0.19 * max_pressure, 1.13 * max_pressure), ylim=(-0.5, 1.5))
    ax.set_axis_off()


def render(path, gel, control, bud, max_pressure):
    height, x, load = bud
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=PAPER)
    fig.text(0.065, 0.942, "How the gel loads a small bud", color=INK, fontsize=26, fontweight="bold")
    fig.text(0.065, 0.900, "Same fiber network, with its outer nodes held fixed", color="#4a626d", fontsize=13)
    draw_slice(fig.add_axes([0.065, 0.255, 0.415, 0.575]), gel, control[0], 0, "Round control")
    draw_slice(fig.add_axes([0.53, 0.255, 0.415, 0.575]), gel, x, height, f"Bud height {height:.1f}")
    fig.text(0.065, 0.225, "Contact force per unit area", color=INK, fontsize=13, fontweight="bold")
    draw_load(fig.add_axes([0.065, 0.104, 0.415, 0.103]), control[1], max_pressure)
    draw_load(fig.add_axes([0.53, 0.104, 0.415, 0.103]), load, max_pressure)
    fig.text(0.065, 0.052, "Cross-section through the bud axis. Dots mark fiber contact in the slice. "
             "Bars measure the full 3D surface.", color="#4a626d", fontsize=11)
    fig.savefig(path, dpi=100, facecolor=PAPER)
    plt.close(fig)


def pressures(gel, x, fit):
    return {row["region"]: row["pressure"] for row in measure(gel, x, "", "+x", 2, np.abs(fit.jac).max())}


def main():
    gel = BudGel(Path(INP) / "p1.0_seed7" / "ECM.topo", kb=0.001, R_far=20, kw=500, h=0.4)
    gel.axis = np.array(AXES["+x"])
    x = gel.X0[gel.fidx].ravel().copy()
    for radius in (3, 3.6, 4.2):
        x = relax(gel, x, radius, 0, 500).x
    for _ in range(2):
        fit = relax(gel, x, 4.8, 0, 1000)
        x = fit.x
    control = (x, pressures(gel, x, fit))
    states = [(0, *control)]
    print("Round control captured", flush=True)
    for height, passes, iterations in ((0.3, 1, 700), (0.6, 2, 1000)):
        for _ in range(passes):
            fit = relax(gel, x, 4.8, height, iterations)
            x = fit.x
        states.append((height, x, pressures(gel, x, fit)))
        print(f"Captured height {height}: {states[-1][2]}", flush=True)

    max_pressure = 1.06 * max(p[region] for _, _, p in states for region in ("tip", "flanks"))
    with TemporaryDirectory(prefix="bud-cavity-") as tmp:
        frames = Path(tmp)
        manifest = []
        for index, (state, duration) in enumerate(zip(states, (2.5, 2.5, 5.0), strict=True)):
            name = f"stage-{index}.png"
            render(frames / name, gel, control, state, max_pressure)
            manifest.extend((f"file {name}", f"duration {duration}"))
        manifest.append(f"file {name}")
        (frames / "frames.txt").write_text("\n".join(manifest) + "\n")
        movie = OUTPUT.with_name("bud_cavity.mp4").resolve()
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-i", "frames.txt", "-t", "10",
                        "-vf", "fps=24", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", str(movie)], cwd=frames, check=True)
    print(f"Wrote {movie}", flush=True)


if __name__ == "__main__":
    main()
