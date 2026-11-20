"""Transport scalar numeric ECSV tables in native model-header slots."""

from io import StringIO

import numpy as np
import xarray as xr
from astropy.table import Table

_STACK_NDIM = 3
_TABLE_NDIM = 2

MODEL_FIELDS = ("Header.Toltec.ModelParams", "Header.Toltec.ModelParamsHeader")


def pack_model_tables(raw, reports):
    """Encode channel tables with their complete ECSV headers, without model logic.

    Parameters
    ----------
    raw : xarray.Dataset
        Acquisition supplying block/channel dimensions and existing model slots.
    reports : mapping of int to astropy.table.Table
        Destination block to report. Each report has one row per acquired channel
        in acquisition order. Unmasked scalar numeric columns are required.

    Returns
    -------
    xarray.Dataset
        Model values and column names. Each selected slot carries its complete
        ECSV header, including column types, units and arbitrary table metadata.

    Raises
    ------
    ValueError
        A report cannot be represented exactly in the numeric header layout.
    """
    if not reports:
        msg = "Supply at least one model table to encode"
        raise ValueError(msg)
    params, header = MODEL_FIELDS
    block_dim, chan_dim = raw["Header.Toltec.ToneFreq"].dims
    names = []
    if (params in raw) != (header in raw):
        msg = "Acquisition contains an incomplete model header group"
        raise ValueError(msg)
    if header in raw:
        names = [
            n.decode() if isinstance(n, bytes) else str(n) for n in raw[header].values
        ]
    if len(set(names)) != len(names) or any(not n for n in names):
        msg = "Model header has empty or duplicate column names"
        raise ValueError(msg)
    original = len(names)
    for block, table in reports.items():
        if not 0 <= block < raw.sizes[block_dim] or len(table) != raw.sizes[chan_dim]:
            msg = "Model table must match the destination block and channel count"
            raise ValueError(msg)
        _validate_columns(table)
        names.extend(n for n in table.colnames if n not in names)
    values = np.full((raw.sizes[block_dim], len(names), raw.sizes[chan_dim]), np.nan)
    attrs = {}
    if params in raw:
        values[:, :original] = raw[params].values
        attrs = dict(raw[params].attrs)
    for block, table in reports.items():
        values[block] = np.nan
        for name in table.colnames:
            values[block, names.index(name)] = table[name]
        stream = StringIO()
        table[:0].write(stream, format="ascii.ecsv")
        attrs[f"ecsv_header_{block}"] = stream.getvalue()
    width = max([32, *map(len, names)])
    result = xr.Dataset(
        {
            params: xr.DataArray(
                values, dims=(block_dim, "modelParamsNum", chan_dim), attrs=attrs
            ),
            header: xr.DataArray(
                np.asarray(names, dtype=f"S{width}"), dims="modelParamsNum"
            ),
        }
    )
    result[header].encoding["char_dim_name"] = "modelParamsHeaderLen"
    return result


def _validate_columns(table):
    for col in table.itercols():
        a = np.asarray(col)
        if a.ndim != 1 or a.dtype.kind not in "iuf" or np.ma.getmaskarray(col).any():
            msg = "Model header transport requires unmasked scalar numeric columns"
            raise ValueError(msg)
        if a.dtype.kind in "iu" and np.any(np.abs(a.astype(object)) > 2**53):
            msg = "Integer model column is not exactly representable in float64"
            raise ValueError(msg)


def read_model_table(data, *, block=0):
    """Decode the same ECSV table from a selected native model slot.

    Parameters
    ----------
    data : xarray.Dataset
        Raw acquisition or stub containing the model-header variable pair.
    block : int, optional
        Destination model slot, default zero.

    Returns
    -------
    astropy.table.Table
        Loaded report with its column order, types, units and metadata restored.

    Raises
    ------
    ValueError
        The ECSV header or numeric payload is inconsistent.
    """
    params, header = MODEL_FIELDS
    descriptor = data[params].attrs.get(f"ecsv_header_{block}")
    if not descriptor:
        msg = "Model slot has no ECSV header"
        raise ValueError(msg)
    table = Table.read(descriptor.splitlines(), format="ascii.ecsv")
    names = [
        n.decode() if isinstance(n, bytes) else str(n) for n in data[header].values
    ]
    payload = data[params]
    if payload.ndim == _STACK_NDIM:
        payload = payload.isel({payload.dims[0]: block})
    if payload.ndim != _TABLE_NDIM or len(table):
        msg = "Model payload requires two axes and an empty ECSV header template"
        raise ValueError(msg)
    values = payload.values
    if len(set(names)) != len(names) or values.shape[0] != len(names):
        msg = "Model columns disagree with their header"
        raise ValueError(msg)
    columns = []
    for name in table.colnames:
        if name not in names:
            msg = "ECSV column is absent from ModelParamsHeader"
            raise ValueError(msg)
        a = values[names.index(name)]
        dtype = table[name].dtype
        if dtype.kind not in "iuf":
            msg = "Model header transport requires numeric ECSV columns"
            raise ValueError(msg)
        if dtype.kind in "iu" and (
            not np.isfinite(a).all()
            or np.any(a != np.floor(a))
            or np.any(np.abs(a) > 2**53)
            or np.any(a < np.iinfo(dtype).min)
            or np.any(a > np.iinfo(dtype).max)
        ):
            msg = "Integer model column contains an unrepresentable value"
            raise ValueError(msg)
        if dtype.kind == "f" and np.any(
            np.isfinite(a) & (np.abs(a) > np.finfo(dtype).max)
        ):
            msg = "Floating model column exceeds its declared range"
            raise ValueError(msg)
        column = table[name]
        columns.append(
            type(column)(
                a.astype(dtype),
                name=name,
                unit=column.unit,
                description=column.description,
                meta=column.meta,
            )
        )
    return Table(columns, meta=table.meta)
