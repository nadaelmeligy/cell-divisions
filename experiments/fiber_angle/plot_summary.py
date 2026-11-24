#!/usr/bin/env python3
"""plot_summary.py -- one figure with the main results of experiments/fiber_angle (reads results/*.csv).
    python3 experiments/fiber_angle/plot_summary.py      -> results/fiber_angle_summary.png
"""
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
BINS = ["0-.2\nflat", ".2-.4", ".4-.6", ".6-.8", ".8-1\nend-on"]
BLUE, ORANGE, GREY = "#2a6fdb", "#e8743b", "#9aa0a6"


def main():
    fig, axs = plt.subplots(2, 2, figsize=(12, 8.6))

    # A: one fiber, no drag
    ax = axs[0, 0]
    s = pd.read_csv(os.path.join(RES, "single_fiber.csv"))
    s = s[(s["mode"] == "static") & np.isclose(s["delta"], 0.5)]
    geoms = [("bridge", 1, "flat, held 1 segment either side"), ("bridge", 3, "flat, held 3 segments either side"),
             ("tip", 0, "ends at organoid, end-on (psi 0)"), ("tip", 30, "ends at organoid, psi 30"),
             ("tip", 60, "ends at organoid, psi 60"), ("tip", 89, "ends at organoid, lying flat (psi 89)")]
    y = np.arange(len(geoms))
    for k, (bend, kb, col, lab) in enumerate([("original", 0.001, BLUE, "our fibers (original bend, kb 0.001)"),
                                              ("wlc", 0.5, ORANGE, "stiffer fibers (wlc bend, kb 0.5)")]):
        F = [s[(s["kind"] == g) & (s["a"] == a) & (s["bend"] == bend) & (s["kb"] == kb)]["F"].iloc[0] for g, a, _ in geoms]
        ax.barh(y + (0.2 if k else -0.2), np.maximum(F, 1e-10), height=0.38, color=col, label=lab)
    ax.set_xscale("log")
    ax.set_xlim(1e-10, 3)
    ax.set_yticks(y, [g[2] for g in geoms], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("push-back force when the organoid has pushed 0.5 in")
    ax.set_title("A. One fiber, no drag: being held on both sides matters, not the angle", fontsize=10)
    ax.legend(fontsize=7, loc="lower right")

    # B: organoid runs, who pushes
    ax = axs[0, 1]
    o = pd.read_csv(os.path.join(RES, "organoid_angle.csv"))
    x = np.arange(5)
    for k, (p, col) in enumerate([("1.0", BLUE), ("0.7", ORANGE)]):
        g = o[o["p"].astype(str) == p].groupby("bin")[["F", "near_len"]].sum()
        ax.bar(x + (k - 0.5) * 0.4, g["F"] / g["F"].sum(), width=0.38, color=col, label=f"share of push-back, p = {p}")
        ax.plot(x, g["near_len"] / g["near_len"].sum(), "o--", color=col, ms=4, alpha=0.8,
                label=f"share of fiber within 1 of the surface, p = {p}")
    ax.set_xticks(x, BINS, fontsize=8)
    ax.set_xlabel("|cos| of the angle between fiber and surface normal")
    ax.set_ylabel("share")
    ax.set_title("B. Organoid runs (t = 90-130): the push comes from flat-lying fibers", fontsize=10)
    ax.legend(fontsize=7)

    # C: linker tension vs pull direction
    ax = axs[1, 0]
    lk = pd.read_csv(os.path.join(RES, "organoid_links.csv"))
    edges = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0 + 1e-9])
    for p, col in [("1.0", BLUE), ("0.7", ORANGE)]:
        g = lk[lk["p"].astype(str) == p]
        b = np.clip(np.digitize(g["align"], edges) - 1, 0, 4)
        m = g.groupby(b)["tension"].agg(["mean", "sem"]).reindex(range(5))
        ax.errorbar(x, m["mean"], yerr=m["sem"], fmt="o-", color=col, capsize=3, label=f"p = {p}")
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_xticks(x, ["0-.2\nsideways", ".2-.4", ".4-.6", ".6-.8", ".8-1\nalong a fiber"], fontsize=8)
    ax.set_xlabel("|cos| of the angle between the linker and the fiber at its node")
    ax.set_ylabel("linker tension (mean ± s.e.)")
    ax.set_title("C. Organoid runs: linkers hold force only when they pull along a fiber", fontsize=10)
    ax.legend(fontsize=8)

    # D: cavity, cutting near-hole segments by angle
    ax = axs[1, 1]
    path = os.path.join(RES, "cavity_angle.csv")
    if os.path.exists(path):
        c = pd.read_csv(path)
        for bend, ls in [("original", "-"), ("wlc", "--")]:
            for arm, col in [("intact", "black"), ("cut_endon", ORANGE), ("cut_flat", BLUE)]:
                case = [q for q in c["case"].unique() if q.startswith(f"static:p1.0:{bend}:") and q.endswith(arm)]
                if not case:
                    continue
                g = c[(c["case"] == case[0]) & (c["bin"] == 0)]
                lab = {"intact": "intact", "cut_endon": "70 end-on segments cut", "cut_flat": "70 flat segments cut"}[arm]
                ax.plot(g["cav_strain"], g["P_total"], ls, color=col, marker="o", ms=3,
                        label=f"{lab} ({'our fibers' if bend == 'original' else 'stiffer, wlc kb 0.5'})")
        ax.set_xlabel("cavity expansion (R - 3.3) / 3.3")
        ax.set_ylabel("gel push-back pressure on the sphere")
        ax.legend(fontsize=7)
    ax.set_title("D. Gel alone, p = 1.0: cut 70 flat vs 70 end-on segments", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fiber_angle_summary.png"), dpi=130)
    print("wrote results/fiber_angle_summary.png")


if __name__ == "__main__":
    main()
