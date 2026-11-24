#!/usr/bin/env python3
"""Relax the existing clamped gel around a round cavity and a prescribed small bud.

Prediction: at comparable tangential fiber tension, the sharper tip turns the
fiber forces inward more strongly than the flanks. Test whether the tip gains
more contact load per unit area than the flanks, relative to the same patches
of a sphere in the same gel. A consistent zero or negative difference would
reject the prediction; opposite signs across axes indicate location sensitivity.

The body radius is fixed during budding. The protrusion is 0.6 model units
high and smoothly vanishes 45 degrees from its axis. The contact potential
uses a first-order normal distance to the surface. There are no cells or
linkers in this cavity test. Run from the repo root with Python 3.
"""

import csv
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from cavity_angle import INP, ce


SUPPORT_COS = np.cos(np.deg2rad(45))
TIP_COS = np.cos(np.deg2rad(15))
AXES = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+z": (0, 0, 1), "-z": (0, 0, -1)}
OUTPUT = Path(__file__).parent / "results" / "bud_cavity.csv"


def profile(mu, radius, height):
    """Radius and its first two derivatives with respect to cos(polar angle)."""
    width = 1 - SUPPORT_COS
    u = np.maximum(0, (mu - SUPPORT_COS) / width)
    return radius + height * u**3, 3 * height * u**2 / width, 6 * height * u / width**2


def distance(points, axis, radius, height):
    """Signed radial gap converted to first-order normal distance, and its gradient."""
    r = np.linalg.norm(points, axis=1)
    radial = points / r[:, None]
    mu = radial @ axis
    rho, slope, curvature = profile(mu, radius, height)
    q = slope / rho
    s = np.sqrt(1 + (1 - mu**2) * q**2)
    ds_dmu = ((1 - mu**2) * q * (curvature / rho - q**2) - mu * q**2) / s
    d = (r - rho) / s
    grad_mu = (axis - mu[:, None] * radial) / r[:, None]
    grad = radial / s[:, None] - ((slope + d * ds_dmu) / s)[:, None] * grad_mu
    return d, grad, mu


class BudGel(ce.Gel):
    """Use the cavity experiment's gel mechanics with a modified contact surface."""

    height = 0.0
    axis = np.array(AXES["+x"])

    def contact(self, coords):
        i, j = self.bonds[self.s_seg].T
        w = self.s_w
        points = coords[i] + w[:, None] * (coords[j] - coords[i])
        d, grad, mu = distance(points, self.axis, self.R, self.height)
        on = d < ce.RANGE
        pen = ce.RANGE - d[on]
        energy = 0.5 * self.kw * self.s_ds * np.dot(pen, pen)
        force = (self.kw * self.s_ds * pen)[:, None] * grad[on]
        # Keep measurement and minimization on the same contact law and samples.
        wc = w[on, None]
        g = ce.scatter(i[on], -(1 - wc) * force, self.n) + ce.scatter(j[on], -wc * force, self.n)
        return energy, g, mu[on], np.linalg.norm(force, axis=1), pen

    def fun(self, x):
        coords = self.full(x)
        stretch, bend, _, gradient, _ = self.parts(coords, self.R, terms="sb")
        energy, contact_gradient, _, _, _ = self.contact(coords)
        return stretch + bend + energy, (gradient + contact_gradient)[self.fidx].ravel()


def check_contact(gel):
    """Isolate the sphere limit and test the bud energy's force by differentiation."""
    gel.R = 4.8
    rng = np.random.default_rng(12)
    x = gel.X0[gel.fidx].ravel() + rng.normal(0, 0.02, 3 * len(gel.fidx))
    coords = gel.full(x)
    gel.select_samples(coords, gel.R + 0.6)
    E, g, *_ = gel.contact(coords)
    _, _, sphere_E, sphere_g, _ = gel.parts(coords, gel.R, terms="c")
    np.testing.assert_allclose(E, sphere_E, atol=1e-10)
    np.testing.assert_allclose(g, sphere_g, atol=1e-10)

    gel.height = 0.6
    _, g, *_ = gel.contact(coords)
    direction = g[gel.fidx].ravel()
    direction /= np.linalg.norm(direction)
    eps = 1e-5
    plus = gel.contact(gel.full(x + eps * direction))[0]
    minus = gel.contact(gel.full(x - eps * direction))[0]
    np.testing.assert_allclose((plus - minus) / (2 * eps), g[gel.fidx].ravel() @ direction, rtol=1e-6)
    gel.height = 0.0
    print("Contact checks passed: sphere limit and bud force gradient", flush=True)


def relax(gel, x, radius, height, iterations):
    gel.R, gel.height = radius, height
    gel.select_samples(gel.full(x), radius + height)
    return minimize(
        gel.fun, x, jac=True, method="L-BFGS-B",
        options={"maxiter": iterations, "gtol": 1e-7, "ftol": 1e-15, "maxcor": 25},
    )


def measure(gel, x, shape, axis_name, checkpoint, grad_max):
    # Use all bonds for measurement, including any that moved into the contact shell.
    gel.set_samples(np.arange(len(gel.bonds)))
    _, _, mu, forces, pen = gel.contact(gel.full(x))
    for region, lo, hi in (("tip", TIP_COS, 1), ("flanks", SUPPORT_COS, TIP_COS),
                           ("body", -1, SUPPORT_COS)):
        samples = np.linspace(lo, hi, 257)
        rho, slope, _ = profile(samples, gel.R, gel.height)
        area_density = rho * np.sqrt(rho**2 + (1 - samples**2) * slope**2)
        area = 2 * np.pi * np.trapezoid(area_density, samples)
        in_region = (mu >= lo) & (mu < hi) if hi < 1 else mu >= lo
        load = forces[in_region].sum()
        yield dict(shape=shape, axis=axis_name, checkpoint=checkpoint, radius=gel.R,
                   height=gel.height, region=region, area=area, force=load,
                   pressure=load / area, samples=int(in_region.sum()),
                   max_penetration=float(pen.max(initial=0)), grad_max=grad_max)


def main():
    path = Path(INP) / "p1.0_seed7" / "ECM.topo"
    gel = BudGel(path, kb=0.001, R_far=20.0, kw=500.0, h=0.4)
    check_contact(gel)
    x = gel.X0[gel.fidx].ravel().copy()
    for radius in (3.0, 3.6, 4.2):
        x = relax(gel, x, radius, 0, 500).x
        print(f"Round warm-up: R={radius:g}", flush=True)

    round_states = []
    for checkpoint in (1, 2):
        fit = relax(gel, x, 4.8, 0, 1000)
        x = fit.x
        round_states.append((x, float(np.max(np.abs(fit.jac)))))
        print(f"Round control checkpoint {checkpoint}: max force {round_states[-1][1]:.4g}", flush=True)

    rows = []
    for name, axis in AXES.items():
        gel.axis = np.array(axis)
        gel.height = 0.0
        for checkpoint, (control, grad_max) in enumerate(round_states, 1):
            rows.extend(measure(gel, control, "round", name, checkpoint, grad_max))
        x = relax(gel, round_states[-1][0], 4.8, 0.3, 700).x
        for checkpoint in (1, 2):
            fit = relax(gel, x, 4.8, 0.6, 1000)
            x = fit.x
            rows.extend(measure(gel, x, "bud", name, checkpoint, float(np.max(np.abs(fit.jac)))))
        control = {row["region"]: row["pressure"] for row in rows
                   if row["axis"] == name and row["shape"] == "round" and row["checkpoint"] == 2}
        load = {row["region"]: row["pressure"] for row in rows[-3:]}
        delta_tip, delta_flank = (load[key] - control[key] for key in ("tip", "flanks"))
        print(f"{name}: Δ tip={delta_tip:+.4f}, Δ flanks={delta_flank:+.4f}, "
              f"difference={delta_tip - delta_flank:+.4f}", flush=True)
        with OUTPUT.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
