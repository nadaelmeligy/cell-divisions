# ECM connectivity and gel → organoid feedback (18–19 Nov 2026)

**Question (Jen's loop):** as the organoid grows it pushes the fibers. That changes the gel's stiffness, which changes the pressure on the outer cells, which changes which cells grow or move. Does that happen in the model, and what does the gel's connectivity p (bond probability) do?

Every result below was re-measured by two independent reviewers with their own code. Numbers are the reviewed ones.

## Setup

| | |
|---|---|
| Model | this branch: pressure-limited growth (P_h = 6), biology linker rules, cell–fiber contact (k 50, range 0.2), 29-cell start |
| Gels | `scripts/tvm/ECM64.py --hole-radius 2.5 --p <p> --seed <s>` in `inputs/`. p = 1.0 / 0.9 / 0.8 / 0.7 means 3.88 / 3.59 / 3.20 / 2.82 bonds per node |
| Half gels | `--p 1.0 --p-left 0.7`: well connected on one half, loose on the other. Each seed (7–10) is run twice, mirrored (stiff half at +x, then at −x) |
| Runs | `runs/ecm_p<p>_s<seed>` (t = 0–80); `runs/long_ecm_*` (continued to 130); `runs/fb_*` (other cell rules from t = 80); `runs/half*_s*` (half gels) |

## Results

**1. The gel is reshaped. Looser gels change more.** By t = 80, fibers near the organoid have turned 22° (p = 1.0) to 30° (p = 0.7).
- **Early:** linkers pull fibers so they point at the organoid.
- **When the organoid arrives (t ≈ 60–80):** the fibers at its surface lie flat around it, about twice as dense as before.
- **Ahead of it:** a band of fibers pointing at the organoid moves outward.
- Videos: `results/grid_gel{7,8,9}.mp4`. Plots: `timeseries.png`, `profiles.png`.

**2. Feedback on growth exists but is weak.** Up to t = 80, growth is the same for every p. At t = 130 (3 gels, p = 1.0 vs 0.7):

| | p = 1.0 vs p = 0.7 |
|---|---|
| fiber stretch near the organoid | 3.5× |
| mean cell pressure | +0.13 to +0.17 |
| cells | 1–7% fewer |

Why it's weak:
- **The gel barely pushes back.** In a periodic box with every node free, as in the engine, the relaxed gel barely resists until about 30% expansion. Even at 100% expansion it pushes back only about 0.8 at p = 1.0, and about 0 at p = 0.7 (`cavity_expansion.png`).
- **What the cells feel is mostly drag.** At p = 1.0 nearly every fiber spans the whole box, so pushing it means dragging a long fiber.
- **Cells respond only through their growth rate, with low gain.** Cells sit at P ≈ 2, while growth stops at P_h = 6.
- Figures: `feedback_long.png`. Video: `long_p1.0_vs_p0.7_gel7.mp4`.

**3. Durotaxis (new): the organoid moves toward the well-connected half of a half gel.**

| t | drift toward the stiff half, 4 mirror pairs (95% CI) | vs. the y/z noise |
|---|---|---|
| 20 | +0.08 (0.03 – 0.13) | z = 4.5 |
| 40 | +0.13 (0.10 – 0.16) | z = 5.4 |
| 80 | +0.20 (0.10 – 0.29), 6% of the radius; 7 of 8 runs | z = 7.8 |

- **When:** it starts at once (t = 3), well before the organoid touches the gel.
- **Mechanism:** linkers on the connected half stay tense. The fiber nodes they hold sit on long fibers (about 45 segments) and barely get pulled in. The net linker pull accounts for 90–110% of the drift.
- **After contact:** the drift slows (+0.05 ± 0.03 over t = 80–130) unless the cells pull (+0.12 ± 0.02).
- Video: `durotaxis_gel7.mp4` (fixed view; + = start, dot = centre). Plot: `half_gel.png`.

**4. Cells that pull (linkers shrink from t = 80) raise their own pressure.** Surface-cell pressure goes up by 0.4–0.7 and growth slows slightly.
- It is **not** a tightening net. The connected gel pulls outward on the organoid.
- The rise comes from the surface cells being deformed. More area means more cortex tension, and that raises their pressure.
- Video: `pull_p1.0_vs_p0.7_gel7.mp4`. Plot: `feedback_screen.png`.

## What is missing for Jen's loop

1. **A gel that pushes back on the cells' pressure scale at the strains the organoid reaches.** The original bending energy, ½·kb·(1 + cos θ)², has no stiffness against small bends, so raising kb barely helps (at most +41% where push-back is above 0.01). The new option `--fiber-bend wlc` uses kb·(1 + cos θ) instead. With kb = 0.5 in the clamped push-back test, it resists earlier:
   - p = 1.0: 9× more at 9% expansion, 1.6× at 45%, +10% at 100%
   - p = 0.7: from about 0 to small (0.013 at 45% expansion)

   It is not yet used in a growth run.
2. **A gel that touches the organoid from the start.** The lattice leaves a 1.4 gap (nearest node at 3.3; organoid radius 1.9).
3. **Cell rules that read the gel beyond growth rate:** fate, movement, sensing of linker tension.
4. **A finer gel.** A fiber segment is 2.83 long and a cell is 1.2 across, so the whole organoid is about one gel pore wide.

## Model issues found (not changed; to discuss)

- **Medium force:** the medium (empty-cell) forces skip organoid surface vertices with type ≥ 3 (`EMPTY CELL CHECK` in `Energy/Volume.cpp` and `Energy/Interface.cpp`), so they don't conserve momentum.
  - **Size:** the organoid gets spurious net kicks of 0.1–3 per snapshot, about the same as the linker forces. This probably causes the drift that all uniform-gel runs share (+0.11 in x over t = 80–130).
  - **Effect on these results:** the mirror pairs cancel it.
  - **To decide:** is it intended?
- **Unseeded noise:** the engine seeds its thermal noise from `std::random_device`, so runs never repeat exactly. "Same seed" means the same starting tissue, gel and Python random numbers, not the same engine noise. `--temperature 0` makes runs repeatable.

## Code changes for this work

| File | Change |
|---|---|
| `Energy/FiberElasticity.*`, `tvm.cpp` | optional 4th number on the `fiber` line: `0` = original bending (default), `1` = worm-like chain. Default output is byte-identical to before (checked at T = 0) |
| `Run/Run.cpp` | the linker rest length never drops below the floor l1 (0.3 used to become 0.1) |
| `run_growth.py` | new options `--fiber-ks`, `--fiber-kb`, `--fiber-bend`, `--temperature`, `--ecm-snapshots`. `--start-from` now keeps the source run's physics (it used to fall back to the defaults). When cells pull, each segment's conf now has the rest length already due (the engine used to replay the decreases, leaving linkers slack for 1–5 steps per segment). NaN check also catches `-nan` and `inf` |
| `growthviz.py` | **bug fix:** `polygon_loop` now rejects a face made of two loops. A division plane through a bent cell produced one and crashed the engine. Also: signed cell volumes (bent cells were overestimated), a volume-weighted centre `center_vol` in each frame, the true linker rest length `link_rest`, a fixed-view grid video (`--fixed-view`), and `--series` |
| `scripts/tvm/ECM64.py` | `--p-left` (one-sided gel), with bonds on the two middle planes split evenly so mirror gels really mirror. The loop-cutting step is now deterministic, so the same seed gives the same gel |
| this folder | `half_gel.py` (mirror-pair drift statistics), `feedback_screen.py`, `cavity_expansion.py` (gel push-back test, `--bend wlc`), `analyse.py` |

## Reproduce

```bash
bash experiments/ecm_connectivity/run_sweep.sh          # 12 uniform-gel runs, t = 0-80 (~30 min on 8 cores)
python3 experiments/ecm_connectivity/analyse.py         # timeseries, profiles, summary.csv, feedback_long.png
python3 experiments/ecm_connectivity/half_gel.py        # durotaxis statistics + half_gel.png
python3 experiments/ecm_connectivity/feedback_screen.py # cell-rule screen
python3 experiments/ecm_connectivity/cavity_expansion.py --periodic --cases p1.0:0.001   # gel push-back
```

Half-gel runs: `python3 run_growth.py --name half_<side>_s<s> --init-dir experiments/ecm_connectivity/inputs/half_<side>_seed<s> --seed <s> --t-start 0 --t-link 1 --t-end 80 --no-movie`. Continue with `--start-from runs/<name>` (add `--shrink-start 80 --shrink-speed 0.2` for pulling cells).

Note: the inputs in `inputs/` were made before the two ECM64.py fixes of 19 Nov. Regenerating them gives the same bond dilution, but about 3% of segments differ in the loop-cutting step, and the half gels' middle planes are now split evenly.
