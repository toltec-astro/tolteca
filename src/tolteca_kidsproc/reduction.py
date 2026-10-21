"""Repetition averaging for KIDs I/Q samples."""

# ruff: noqa: E741, N803 -- Preserve physical I/Q names.
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

import numpy as np
import xarray as xr

__all__ = ["reduce_sweep_samples"]


def reduce_sweep_samples(
    I: xr.DataArray,
    Q: xr.DataArray,
    *,
    sample_dim: str,
    sample_slices: Sequence[slice],
    compute_uncertainty: bool = True,
) -> xr.Dataset:
    """Average I/Q repetitions within explicitly selected sample intervals.

    Parameters
    ----------
    I, Q : xarray.DataArray
        In-phase and quadrature samples with identical dimensions and shape.
        Units are read through the ``u`` accessor and retained on means and
        uncertainties. Bare arrays produce bare results.
    sample_dim : str
        Dimension to average. All other dimensions and their coordinates remain.
    sample_slices : sequence of slice
        Nonempty intervals relative to the input arrays, one per output probe
        step. Each slice has integer start and exclusive stop bounds with no
        stride. Output steps follow this sequence; frequency coordinates are
        supplied by the caller.
    compute_uncertainty : bool, default True
        Include the population standard deviation at each output step.

    Returns
    -------
    xarray.Dataset
        Eager float64 I/Q means and, when enabled, population standard
        deviations (ddof=0). The final dimension is ``sweep``. NaNs propagate.

    Raises
    ------
    ValueError
        I/Q dimensions differ, the sample dimension is absent, or a slice is
        empty, strided, or outside the sample bounds.

    Examples
    --------
    >>> I = xr.DataArray([1., 3., 5., 7.], dims="sample").u.set("adu")
    >>> Q = -I
    >>> reduced = reduce_sweep_samples(
    ...     I, Q, sample_dim="sample", sample_slices=[slice(0, 2), slice(2, 4)])
    >>> reduced.I.values.tolist()
    [2.0, 6.0]
    """
    if I.dims != Q.dims or I.shape != Q.shape or sample_dim not in I.dims:
        msg = "I/Q must share dimensions and shape, including the sample dimension"
        raise ValueError(msg)
    size = I.sizes[sample_dim]
    if not sample_slices or any(
        part.step not in {None, 1}
        or part.start is None
        or part.stop is None
        or not 0 <= part.start < part.stop <= size
        for part in sample_slices
    ):
        msg = "Sample slices must be nonempty, unstrided intervals within the input"
        raise ValueError(msg)
    dims = tuple(dim for dim in I.dims if dim != sample_dim)
    coords = {
        name: coord for name, coord in I.coords.items() if sample_dim not in coord.dims
    }
    result = xr.Dataset(coords=coords)
    # Load each signal once, preserving its native reduction-axis ordering.
    axis = I.get_axis_num(sample_dim)
    for array, mean_field, unc_field in (
        (I, "I", "unc_I"),
        (Q, "Q", "unc_Q"),
    ):
        values = array.values
        means = np.empty((*[array.sizes[d] for d in dims], len(sample_slices)))
        uncertainties = np.empty_like(means) if compute_uncertainty else None
        for step, part in enumerate(sample_slices):
            index = [slice(None)] * values.ndim
            index[axis] = part
            samples = values[tuple(index)]
            means[..., step] = np.mean(samples, axis=axis)
            if uncertainties is not None:
                uncertainties[..., step] = np.std(samples, axis=axis)
        attrs = {"units": array.u.unit_str} if array.u.unit is not None else {}
        out_dims = (*dims, "sweep")
        result[mean_field] = xr.DataArray(means, dims=out_dims, attrs=attrs)
        if uncertainties is not None:
            result[unc_field] = xr.DataArray(
                uncertainties,
                dims=out_dims,
                attrs=attrs,
            )
    return result
