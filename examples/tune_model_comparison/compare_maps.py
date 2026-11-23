#!/usr/bin/env python
"""Compare Citlali intensity FITS on their unchanged common sky pixels.

The inputs must use equal pixel scales and an integer-aligned linear offset
grid. This comparison never shifts a source, interpolates pixels, or rescales
one signal to resemble the other. Configuration/input equivalence is a separate
prerequisite: the image comparison alone cannot establish it.
"""

# ruff: noqa: PLR2004

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


def read_map(path):
    """Read an intensity map and its linear azimuth/elevation grid."""
    with fits.open(path) as hdus:
        h = hdus["signal_I"].header.copy()
        signal = np.squeeze(hdus["signal_I"].data).copy()
        weight = np.squeeze(hdus["weight_I"].data).copy()
        coverage = np.squeeze(hdus["coverage_I"].data).copy()
    if signal.ndim != 2:
        msg = "Expected one two-dimensional intensity map"
        raise ValueError(msg)
    for axis, kind in [(1, "AZOFFSET"), (2, "ELOFFSET")]:
        if h[f"CTYPE{axis}"] != kind or h[f"CUNIT{axis}"].strip() != "arcsec":
            msg = "This comparison requires linear arcsecond offset axes"
            raise ValueError(msg)
    coords = [
        (np.arange(n) + 1 - h[f"CRPIX{axis}"]) * h[f"CDELT{axis}"] + h[f"CRVAL{axis}"]
        for axis, n in [(1, signal.shape[1]), (2, signal.shape[0])]
    ]
    return {
        "path": path,
        "signal": signal,
        "weight": weight,
        "coverage": coverage,
        "x": coords[0],
        "y": coords[1],
        "header": h,
    }


def shared_axis(left, right):
    """Find exactly shared pixels; reject fractional offsets and unequal scales."""
    left_order, right_order = np.argsort(left), np.argsort(right)
    left, right = left[left_order], right[right_order]
    dl, dr = np.diff(left), np.diff(right)
    if not (np.allclose(dl, dl[0]) and np.allclose(dr, dl[0])):
        msg = "Pixel scales differ"
        raise ValueError(msg)
    shift = (right[0] - left[0]) / dl[0]
    if not np.isclose(shift, round(shift), atol=1e-8, rtol=0):
        msg = "Pixel grids have a fractional offset"
        raise ValueError(msg)
    li = np.arange(len(left))
    ri = li - round(shift)
    keep = (ri >= 0) & (ri < len(right))
    if not keep.any():
        msg = "No shared pixels"
        raise ValueError(msg)
    return left_order[li[keep]], right_order[ri[keep]]


def compare(left, right, *, radius=40.0, coverage_fraction=0.1):
    """Measure residuals using the same finite, covered pixels in each map."""
    if left["header"].get("UNIT") != right["header"].get("UNIT"):
        msg = "Map units differ"
        raise ValueError(msg)
    lx, rx = shared_axis(left["x"], right["x"])
    ly, ry = shared_axis(left["y"], right["y"])
    a, b = left["signal"][np.ix_(ly, lx)], right["signal"][np.ix_(ry, rx)]
    x, y = left["x"][lx], left["y"][ly]
    mask = np.isfinite(a) & np.isfinite(b)
    for item, ix, iy in [(left, lx, ly), (right, rx, ry)]:
        weight = item["weight"][np.ix_(iy, ix)]
        cov = item["coverage"][np.ix_(iy, ix)]
        mask &= (weight > 0) & np.isfinite(weight)
        mask &= cov >= coverage_fraction * np.nanmax(item["coverage"])
    aperture = (y[:, None] ** 2 + x[None, :] ** 2) <= radius**2
    valid = mask & aperture
    if valid.sum() < 3:
        msg = "Insufficient shared aperture coverage"
        raise ValueError(msg)
    d = b - a
    metrics = {
        "common_pixels": int(mask.sum()),
        "aperture_pixels": int(valid.sum()),
        "radius_arcsec": radius,
        "coverage_fraction": coverage_fraction,
        "unit": left["header"].get("UNIT"),
        "signal_correlation": float(np.corrcoef(a[valid], b[valid])[0, 1]),
        "difference_rms": float(np.sqrt(np.mean(d[valid] ** 2))),
        "difference_max_abs": float(np.max(np.abs(d[valid]))),
        "left_peak": float(np.max(a[valid])),
        "right_peak": float(np.max(b[valid])),
        "pointing": {},
    }
    for key in ["X_T", "Y_T", "AMP", "A_FWHM", "B_FWHM"]:
        values = [
            float(m["header"].get("POINTING." + key, np.nan)) for m in [left, right]
        ]
        metrics["pointing"][key.lower()] = {
            "left": values[0],
            "right": values[1],
            "delta": values[1] - values[0],
        }
    return x, y, np.where(mask, a, np.nan), np.where(mask, b, np.nan), metrics


def main():
    """Write matched-coordinate figures, source hashes and metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "left", type=Path, help="Directory containing the three array FITS"
    )
    parser.add_argument("right", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--left-label", default="v2")
    parser.add_argument("--right-label", default="v3")
    parser.add_argument("--arrays", nargs="+", default=["a1100", "a1400", "a2000"])
    parser.add_argument("--difference-label", default="v3 - v2")
    parser.add_argument("--radius", type=float, default=40.0)
    parser.add_argument("--title", default="Matched reduction comparison")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        len(args.arrays),
        3,
        figsize=(13, 4 * len(args.arrays)),
        layout="constrained",
        squeeze=False,
    )
    rows = []
    for i, array in enumerate(args.arrays):
        paths = []
        for folder in [args.left, args.right]:
            found = sorted(folder.glob(f"*_{array}_*.fits"))
            if len(found) != 1:
                msg = f"Expected one {array} FITS in {folder}; found {found}"
                raise ValueError(msg)
            paths.append(found[0])
        left = read_map(paths[0])
        right = read_map(paths[1])
        x, y, a, b, metrics = compare(left, right, radius=args.radius)
        metrics["array"] = array
        metrics["wcs_pixel_scales"] = [
            {axis: m["header"][axis] for axis in ["CDELT1", "CDELT2"]}
            for m in [left, right]
        ]
        metrics["sources"] = [
            {
                "path": str(p.resolve()),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
            for p in paths
        ]
        rows.append(metrics)
        central = (np.abs(y[:, None]) <= args.radius) & (
            np.abs(x[None, :]) <= args.radius
        )
        vmax = float(np.nanmax(np.r_[a[central], b[central]]))
        vmin = float(np.nanmin(np.r_[a[central], b[central]]))
        limit = max(float(np.nanmax(np.abs((b - a)[central]))), vmax * 1e-6)
        dx, dy = x[1] - x[0], y[1] - y[0]
        extent = [x[0] - dx / 2, x[-1] + dx / 2, y[0] - dy / 2, y[-1] + dy / 2]
        for j, data in enumerate([a, b, b - a]):
            ax = axes[i, j]
            im = ax.imshow(
                data,
                origin="lower",
                extent=extent,
                cmap="RdBu_r" if j == 2 else "viridis",
                vmin=-limit if j == 2 else vmin,
                vmax=limit if j == 2 else vmax,
            )
            ax.set(
                xlim=(-args.radius, args.radius),
                ylim=(-args.radius, args.radius),
                xlabel="Azimuth offset (arcsec)",
                ylabel="Elevation offset (arcsec)",
            )
            ax.set_title(
                f"{array} · "
                + [args.left_label, args.right_label, args.difference_label][j]
            )
            fig.colorbar(im, ax=ax, shrink=0.8, label=metrics["unit"])
    fig.suptitle(
        args.title + "\nShared signal scale; no image registration or signal rescaling",
        fontsize=14,
    )
    fig.savefig(args.output / "maps.png", dpi=170)
    fig.savefig(args.output / "maps.pdf")
    (args.output / "metrics.json").write_text(json.dumps(rows, indent=2))
    sys.stdout.write(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
