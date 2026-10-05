"""Reusable LMT observation functions."""

import pandas as pd

__all__ = [
    "make_quartet",
    "make_uid_columns",
    "make_uid_raw_obs",
    "make_uid_raw_obs_file",
]


def make_quartet(
    master: str | None, obsnum: int | None, subobsnum: int | None, scannum: int | None
) -> str | None:
    """Construct an observation identifier.

    Parameters
    ----------
    master : str or None
        Acquisition master label.
    obsnum : int or None
        Observation number.
    subobsnum : int or None
        Sub-observation number.
    scannum : int or None
        Scan number.

    Returns
    -------
    str or None
        Observation identifier, or None if any input is None.
    """
    parts = (master, obsnum, subobsnum, scannum)
    if any(part is None for part in parts):
        return None
    return "-".join(map(str, parts))


def make_uid_raw_obs(
    master: str | None, obsnum: int | None, subobsnum: int | None, scannum: int | None
) -> str | None:
    """Construct a raw-observation identifier.

    Parameters
    ----------
    master : str or None
        Acquisition master label.
    obsnum : int or None
        Observation number.
    subobsnum : int or None
        Sub-observation number.
    scannum : int or None
        Scan number.

    Returns
    -------
    str or None
        Raw-observation identifier, or None if any input is None.
    """
    return make_quartet(master, obsnum, subobsnum, scannum)


def make_uid_raw_obs_file(
    master: str | None,
    obsnum: int | None,
    subobsnum: int | None,
    scannum: int | None,
    interface: str | None,
) -> str | None:
    """Construct a raw-observation file identifier.

    Parameters
    ----------
    master : str or None
        Acquisition master label.
    obsnum : int or None
        Observation number.
    subobsnum : int or None
        Sub-observation number.
    scannum : int or None
        Scan number.
    interface : str or None
        Readout interface identifier.

    Returns
    -------
    str or None
        Raw-observation file identifier, or None if any input is None.
    """
    uid_raw_obs = make_uid_raw_obs(master, obsnum, subobsnum, scannum)
    if uid_raw_obs is None or interface is None:
        return None
    return f"{uid_raw_obs}-{interface}"


def make_uid_columns(
    master: pd.Series,
    obsnum: pd.Series,
    subobsnum: pd.Series,
    scannum: pd.Series,
    interface: pd.Series,
) -> pd.DataFrame:
    """Construct observation identifier columns.

    Parameters
    ----------
    master : pandas.Series
        Acquisition master labels.
    obsnum : pandas.Series
        Observation numbers.
    subobsnum : pandas.Series
        Sub-observation numbers.
    scannum : pandas.Series
        Scan numbers.
    interface : pandas.Series
        Readout interface identifiers.

    Returns
    -------
    pandas.DataFrame
        Nullable string columns ``quartet``, ``uid_raw_obs`` and
        ``uid_raw_obs_file``, preserving the input index and row order.
        An identifier is null if any of its components is missing.
        Input Series are not modified.

    Raises
    ------
    ValueError
        If the input Series do not have equal indices in the same order.

    Examples
    --------
    >>> obs = pd.DataFrame({
    ...     "master": ["tcs"], "obsnum": [145609], "subobsnum": [0],
    ...     "scannum": [2], "interface": ["tel"],
    ... })
    >>> uids = make_uid_columns(
    ...     obs.master, obs.obsnum, obs.subobsnum, obs.scannum, obs.interface
    ... )
    >>> uids.uid_raw_obs_file.tolist()
    ['tcs-145609-0-2-tel']
    """
    columns = {
        "master": master,
        "obsnum": obsnum,
        "subobsnum": subobsnum,
        "scannum": scannum,
        "interface": interface,
    }
    if any(not master.index.equals(column.index) for column in columns.values()):
        msg = "UID input columns must have equal indices in the same order"
        raise ValueError(msg)
    data = pd.DataFrame(columns).astype(object)
    rows = list(data.where(data.notna(), None).itertuples(index=False, name=None))
    return pd.DataFrame(
        {
            name: pd.array([make_value(*row[:count]) for row in rows], dtype="string")
            for name, make_value, count in (
                ("quartet", make_quartet, 4),
                ("uid_raw_obs", make_uid_raw_obs, 4),
                ("uid_raw_obs_file", make_uid_raw_obs_file, 5),
            )
        },
        index=master.index,
    )
