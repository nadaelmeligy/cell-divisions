#!/usr/bin/env python3
"""Record a time-dependent +x bud and its matched round control in gel seed 7.

Starting from the same relaxed round cavity, the bud grows to height 0.6 over
18 model time units and then holds for 6. The gel moves according to the
overdamped Euler dynamics of the round dynamic cavity test. The usual contact
stiffness is 50. The outer gel nodes are held fixed as in the static bud test.

In the earlier relaxation test the bud added more load per area at the tip.
Here the prediction is that the same tip-minus-flank load increment stays
positive during the hold, where the shape has stopped driving the fibers.
Compare each patch with the matching patch of the round control at the same
time. A nonpositive increment at the end of the hold would refute that
prediction for this run.
"""

import argparse
from pathlib import Path

import numpy as np

from bud_cavity import INP, BudGel, ce, check_contact, measure, relax


RADIUS = 4.8
HEIGHT = 0.6
GROW_TIME = 18
HOLD_TIME = 6
DT = 0.005
FRAME_DT = 0.2


def evolve(gel, initial, *, bud):
    x = initial.copy()
    interval = round(FRAME_DT / DT)
    last_step = round((GROW_TIME + HOLD_TIME) / DT)
    frames, heights, loads = [], [], []
    for step in range(last_step + 1):
        t = step * DT
        gel.height = HEIGHT * min(t / GROW_TIME, 1) if bud else 0
        if step % 20 == 0:
            gel.select_samples(gel.full(x), RADIUS + gel.height)
        _, force_gradient = gel.fun(x)
        if step % interval == 0:
            data = {row["region"]: row for row in measure(gel, x, "bud" if bud else "round", "+x", step,
                                                          np.abs(force_gradient).max())}
            frames.append(gel.full(x))
            heights.append(gel.height)
            loads.append([data[region]["pressure"] for region in ("tip", "flanks")])
            gel.select_samples(frames[-1], RADIUS + gel.height)
            if not np.isfinite(frames[-1]).all() or data["tip"]["max_penetration"] >= ce.RANGE:
                raise RuntimeError(f"Invalid coordinates or interior contact at t={t:g}")
            if step % round(6 / DT) == 0:
                print(f"{'Bud' if bud else 'Round'} t={t:g} height={gel.height:.2f} "
                      f"tip={loads[-1][0]:.3f} flanks={loads[-1][1]:.3f}", flush=True)
        x -= DT * force_gradient
    return np.asarray(frames), np.asarray(heights), np.asarray(loads)


def main(output):
    gel = BudGel(Path(INP) / "p1.0_seed7" / "ECM.topo", kb=0.001, R_far=20, kw=50, h=0.5)
    check_contact(gel)
    initial = gel.X0[gel.fidx].ravel().copy()
    for radius in (3.0, 3.6, 4.2, RADIUS):
        initial = relax(gel, initial, radius, 0, 500 if radius < RADIUS else 2000).x
    print("Round starting geometry ready", flush=True)

    # At a fixed shape the Euler step should dissipate elastic energy.
    gel.height = HEIGHT
    gel.select_samples(gel.full(initial), RADIUS + HEIGHT)
    before, gradient = gel.fun(initial)
    if gel.fun(initial - DT * gradient)[0] >= before:
        raise RuntimeError("Fixed-shape Euler step did not reduce energy")

    bud_frames, heights, bud_load = evolve(gel, initial, bud=True)
    _, _, round_load = evolve(gel, initial, bud=False)
    np.testing.assert_allclose(bud_load[0], round_load[0], atol=1e-12)
    np.savez_compressed(output, times=np.arange(len(heights)) * FRAME_DT, frames=bud_frames, heights=heights,
                        bud_load=bud_load, round_load=round_load, bonds=gel.bonds, l0=ce.L0)
    delta_tip, delta_flanks = bud_load[-1] - round_load[-1]
    print(f"Final changes: tip {delta_tip:+.4f}, flanks {delta_flanks:+.4f}; saved {len(heights)} frames", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="path for the recorded trajectory (.npz)")
    main(parser.parse_args().output)
