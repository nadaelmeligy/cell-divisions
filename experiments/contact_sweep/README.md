# Contact-force sweep (24 Nov 2026)

**Question (Nada):** contact is what lays the fibers flat (Result 1). Does that depend on the exact shape of
our contact force, E = ½ k Δs max(0, δ − s)² per fiber test point (`Energy/CellFiberContact.cpp`)?

**Fast test:** the gel alone, no cells or linkers. A rigid sphere grows quasi-statically in the organoid's hole
(R = 3.0 → 6.6, gel clamped at R_far = 20), with the engine's fiber energies (ks = 10, original bend kb = 0.001),
as in `experiments/fiber_angle/cavity_angle.py` "static". Only k and δ change. Script: `sweep.py K DELTA`;
results in `results/k{K}_d{DELTA}.csv` (one row per radius) and `results/log_*.txt`.

**Measured** at each R: S of segments with midpoints in the shell R..R+2 (as Result 1, −½ = all flat), the
share of the push carried by flat fibers (|cos ψ| < 0.2, as Result 2), the push-back pressure, and the deepest
fiber point inside the sphere.

| k | δ | S at R = 3.0 | R = 4.8 | R = 6.6 | flat share of push | push (R = 6.6) | deepest overlap |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.1 | −0.05 | −0.22 | −0.25 | 0.98 | 0.77 | 0.26 |
| 10 | 0.2 | −0.07 | −0.24 | −0.29 | 0.98 | 0.82 | 0.27 |
| 10 | 0.4 | −0.12 | −0.26 | −0.32 | 0.97 | 0.98 | 0.29 |
| 50 | 0.1 | −0.05 | −0.23 | −0.29 | 1.00 | 0.83 | 0.11 |
| 50 | 0.2 | −0.07 | −0.24 | −0.30 | 1.00 | 0.89 | 0.11 |
| 50 | 0.4 | −0.12 | −0.27 | −0.32 | 1.00 | 1.06 | 0.12 |
| 250 | 0.1 | −0.05 | −0.23 | −0.29 | 1.00 | 0.84 | 0.04 |
| 250 | 0.2 | −0.07 | −0.24 | −0.30 | 1.00 | 0.92 | 0.04 |
| 250 | 0.4 | −0.12 | −0.27 | −0.32 | 1.00 | 1.10 | 0.05 |

**Answer:** the flattening does not depend on the contact's stiffness or range. Over a 25× range in k and 4× in δ,
S at R = 6.6 stays between −0.25 and −0.32, and flat fibers carry 97–100% of the push in every case. Softer
contact lets fibers sink deeper (0.27 at k = 10 vs 0.04 at k = 250) and lowers the push by at most ~20%; a longer
range acts like a slightly bigger sphere (a bit more flattening and push). Prediction written before the run:
robust to k and δ, softest contact sinks deeper and pushes less. Both held.

Not covered: friction (kept frictionless, as agreed with Jen), other force laws (Hertz, hard wall), and the full
organoid runs with cells and linkers.

## Does the flattening come from how the force is spread along a segment? (24 Nov 2026)

**Question (Nada):** the contact pushes test points along each segment and shares each push between the two nodes
like a lever, (1 − w) and w, so a segment touching the ball with one end is pushed harder there and turns flat.
Is every fiber doomed to lie flat because of that, i.e. is the flattening an artefact of the contact law?

**Test (`node_only.py`):** remove that torque completely. Contact acts only on the nodes (test points at w = 0 and
1 only), so no contact force can rotate a segment. Same ball test, same gels as the full runs, S of the same strands
(midpoints starting 3.15–5.15 from the centre), as `experiments/matrigel/ball_cohort.py`. Prediction written before
the run: they still flatten, because an expanding ball squeezes the gel along the radius and stretches it around the
ball, and that deformation alone rotates strands toward flat.

| R | 3.6 | 4.2 | 4.8 | 5.4 | 6.0 |
|---|---:|---:|---:|---:|---:|
| collagen-like, contact along the segment (engine) | −0.17 | −0.30 | −0.41 | −0.47 | −0.49 |
| collagen-like, contact on nodes only | −0.12 | −0.25 | −0.39 | −0.46 | −0.49 |
| Matrigel-like mesh (ks 10), along the segment | −0.11 | −0.23 | −0.37 | −0.48 | −0.50 |
| Matrigel-like mesh (ks 10), nodes only | −0.07 | −0.17 | −0.32 | −0.43 | −0.50 |

**Answer:** the lever torque only speeds the flattening up a little; without it the same strands end just as flat in
both gels. The flattening comes from the geometry of a growing ball pushing on a connected gel (radial squeeze, hoop
stretch), not from how the contact force is spread. It matches what is seen around non-invasive tumours (collagen
"stretched around the tumor", Provenzano 2008, BMC Medicine); radial fibers there come from cells pulling, which our
model does not have (linkers are passive springs).
