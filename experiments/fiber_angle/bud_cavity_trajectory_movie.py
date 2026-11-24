#!/usr/bin/env python3
"""Render the recorded node positions from bud_cavity_dynamics.py as an MP4."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import to_rgb
from matplotlib.ticker import MaxNLocator

from bud_cavity import OUTPUT, SUPPORT_COS, TIP_COS, ce, distance, profile
from bud_cavity_dynamics import GROW_TIME, RADIUS


INK = "#17303c"
PAPER = "#f7f9fa"
TIP = "#ad3f22"
FLANKS = "#246c86"
AXIS = np.array([1, 0, 0])
VIEW = np.array([0.28, 0.16, 1.0])
VIEW /= np.linalg.norm(VIEW)
RIGHT = np.cross([0, 1, 0], VIEW)
RIGHT /= np.linalg.norm(RIGHT)
UP = np.cross(VIEW, RIGHT)
CAMERA = np.stack((RIGHT, UP, VIEW), axis=1)


def surface_faces():
    theta = np.linspace(0, np.pi, 53)
    phi = np.linspace(0, 2 * np.pi, 57)
    s, c = np.sin(theta[:, None]), np.cos(theta[:, None])
    dirs = np.stack((np.broadcast_to(c, (len(theta), len(phi))), s * np.cos(phi), s * np.sin(phi)), axis=-1)
    faces = np.stack([dirs[:-1, :-1], dirs[1:, :-1], dirs[1:, 1:], dirs[:-1, 1:]], axis=2).reshape(-1, 4, 3)
    mu = faces[:, :, 0].mean(axis=1)
    tint = np.full((len(faces), 3), to_rgb("#c4d5dc"))
    tint[mu >= SUPPORT_COS] = to_rgb("#85b2c1")
    tint[mu >= TIP_COS] = to_rgb("#db896c")
    return faces, tint


def draw_surface(artist, faces, tint, height):
    radius, _, _ = profile(faces[:, :, 0], RADIUS, height)
    surface = faces * radius[:, :, None]
    _, normals, _ = distance(surface.mean(axis=1), AXIS, RADIUS, height)
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    projected = surface @ CAMERA
    facing = np.flatnonzero(normals @ VIEW > 0)
    facing = facing[np.argsort(projected[facing, :, 2].mean(axis=1))]
    light = 0.78 * VIEW + 0.53 * UP - 0.34 * RIGHT
    brightness = 0.74 + 0.28 * np.maximum(0, normals[facing] @ light)
    artist.set_verts(projected[facing, :, :2])
    artist.set_facecolor(np.clip(tint[facing] * brightness[:, None], 0, 1))


def draw_fibers(back, front, points, bonds, rest):
    a, b = points[bonds[:, 0]], points[bonds[:, 1]]
    d = b - a
    extension = (np.linalg.norm(d, axis=1) - rest) / rest
    strength = np.clip(extension / 0.045, 0.0, 1.0)
    # Subdivide the lines for front/back ordering. Every piece lies on its actual bond.
    t = np.linspace(0, 1, 13)
    polyline = a[:, None, :] + d[:, None, :] * t[None, :, None]
    pieces = np.stack((polyline[:, :-1], polyline[:, 1:]), axis=2).reshape(-1, 2, 3)
    midpoint = pieces.mean(axis=1)
    projected = pieces @ CAMERA
    show = (np.linalg.norm(midpoint, axis=1) < 7.8) & (np.abs(midpoint @ VIEW) < 6.5)
    weight = np.repeat(strength, len(t) - 1)
    near = midpoint @ VIEW >= 0
    neutral, stretched = np.array(to_rgb("#8a9ba4")), np.array(to_rgb("#137874"))
    color = neutral + weight[:, None] * (stretched - neutral)
    for artist, mask, alpha in ((back, show & ~near, 0.40), (front, show & near, 0.66)):
        artist.set_segments(projected[mask, :, :2])
        opacity = alpha * (0.72 + 0.28 * weight[mask])
        artist.set_color(np.column_stack((color[mask], opacity)))
        artist.set_linewidth(0.40 + 0.90 * weight[mask])


def render(data, path):
    t, frames, heights = (data[k] for k in ("times", "frames", "heights"))
    bud, control = data["bud_load"], data["round_load"]
    bonds, rest = data["bonds"], float(data["l0"])
    starts, ends = frames[0, bonds[:, 0]], frames[0, bonds[:, 1]]
    bonds = bonds[ce.closest_approach(starts, ends - starts) < 8.6]
    faces, tint = surface_faces()

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=PAPER)
    fig.text(0.063, 0.929, "A bud grows into the fiber gel", fontsize=27, fontweight="bold", color=INK)
    fig.text(0.063, 0.884, "The bump grows to a prescribed height. The surrounding fibers move in response.",
             color="#46606c", fontsize=13)

    scene = fig.add_axes([0.043, 0.106, 0.535, 0.734], facecolor=PAPER)
    scene.set(xlim=(-7.85, 7.85), ylim=(-6.9, 6.9), aspect="equal")
    scene.set_axis_off()
    back = LineCollection([], zorder=1)
    surface = PolyCollection([], linewidths=0.35, edgecolors="face", zorder=2)
    front = LineCollection([], zorder=3)
    for artist in (back, surface, front):
        scene.add_collection(artist)

    plot = fig.add_axes([0.625, 0.265, 0.31, 0.46], facecolor=PAPER)
    plot.set_title("Contact force per unit area", loc="left", color=INK, fontsize=16, fontweight="bold", pad=16)
    plot.set(xlim=(0, t[-1]), ylim=(0, 1.15 * max(bud.max(), control.max())), xlabel="Model time")
    plot.set_xticks([0, 6, 12, 18, 24])
    plot.yaxis.set_major_locator(MaxNLocator(5))
    plot.spines[["top", "right"]].set_visible(False)
    plot.spines[["left", "bottom"]].set_color("#9aadb7")
    plot.tick_params(colors="#415c67", labelsize=11)
    plot.xaxis.label.set_color(INK)
    plot.grid(axis="y", color="#dbe4e8", lw=0.65)
    plot.axvspan(GROW_TIME, t[-1], color="#e9eef1", zorder=0)
    for index, color in enumerate((TIP, FLANKS)):
        plot.plot(t, control[:, index], linestyle="--", color=color, alpha=0.50, lw=1.25)
    tip, = plot.plot([], [], color=TIP, lw=2.6)
    flanks, = plot.plot([], [], color=FLANKS, lw=2.6)
    t_line = plot.axvline(0, color=INK, lw=0.85, alpha=0.65)
    plot.legend([tip, flanks], ["Tip", "Flanks"], loc="upper left", frameon=False, labelcolor=INK)
    fig.text(0.625, 0.178, "Dashed lines show the round control.", color="#46606c", fontsize=11)
    fig.text(0.625, 0.143, "Shading marks the hold at full height.", color="#46606c", fontsize=11)
    status = fig.text(0.063, 0.090, "", color=INK, fontsize=13, fontweight="bold")
    fig.text(0.063, 0.045, "Recorded fiber positions every 0.2 model time units. Teal fibers are stretched.",
             color="#46606c", fontsize=11)

    writer = FFMpegWriter(fps=12, codec="libx264", extra_args=["-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    with writer.saving(fig, str(path), dpi=100):
        for index, (time, points, height) in enumerate(zip(t, frames, heights, strict=True)):
            draw_surface(surface, faces, tint, height)
            draw_fibers(back, front, points, bonds, rest)
            tip.set_data(t[:index + 1], bud[:index + 1, 0])
            flanks.set_data(t[:index + 1], bud[:index + 1, 1])
            t_line.set_xdata([time, time])
            status.set_text(f"Model time {time:4.1f}     Bump height {height:.2f}")
            writer.grab_frame()
            if index % 40 == 0:
                print(f"Rendered frame {index + 1}/{len(t)}", flush=True)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    args = parser.parse_args()
    movie = OUTPUT.with_name("bud_cavity_dynamics.mp4")
    with np.load(args.trajectory) as trajectory:
        render(trajectory, movie)
    print(f"Wrote {movie}", flush=True)
