"""Reusable LMT observation functions."""

import pandas as pd

__all__ = [
    "make_quartet",
    "make_uid_columns",
    "make_uid_obs",
    "make_uid_raw_obs",
    "make_uid_raw_obs_file",
]


def make_quartet(
    master: str | None,
    obsnum: int | None,
    subobsnum: int | None = None,
    scannum: int | None = None,
) -> str | None:
    """Construct an observation identifier.

    Parameters
    ----------
    master : str or None
        Acquisition master label, omitted from the identifier if None.
    obsnum : int or None
        Observation number.
    subobsnum : int or None, optional
        Sub-observation number. Default is None.
    scannum : int or None, optional
        Scan number. Default is None.

    Returns
    -------
    str or None
        Observation identifier with an optional master and trailing numbers.
        None if the observation number is missing or numeric components
        contain a gap.

    Examples
    --------
    >>> make_quartet("tcs", 1, 1, 1)
    'tcs-1-1-1'
    >>> make_quartet(None, 1, 1, 1)
    '1-1-1'
    >>> make_quartet("tcs", 1)
    'tcs-1'
    >>> make_quartet(None, 1)
    '1'
    >>> make_quartet("tcs", 1, None, 3) is None
    True
    """
    if obsnum is None or (subobsnum is None and scannum is not None):
        return None
    parts = (master, obsnum, subobsnum, scannum)
    return "-".join(str(part) for part in parts if part is not None)


def make_uid_obs(master: str | None, obsnum: int | None) -> str | None:
    """Construct an observation identifier.

    Parameters
    ----------
    master : str or None
        Acquisition master label, omitted from the identifier if None.
    obsnum : int or None
        Observation number.

    Returns
    -------
    str or None
        Observation identifier, or None if the observation number is missing.

    Examples
    --------
    >>> make_uid_obs("tcs", 1)
    'tcs-1'
    >>> make_uid_obs(None, 1)
    '1'
    """
    return make_quartet(master, obsnum)


def make_uid_raw_obs(
    master: str | None, obsnum: int | None, subobsnum: int | None, scannum: int | None
) -> str | None:
    """Construct a raw-observation identifier.

    Parameters
    ----------
    master : str or None
        Acquisition master label, omitted from the identifier if None.
    obsnum : int or None
        Observation number.
    subobsnum : int or None
        Sub-observation number.
    scannum : int or None
        Scan number.

    Returns
    -------
    str or None
        Raw-observation identifier, or None if any observation number is missing.
    """
    if any(part is None for part in (obsnum, subobsnum, scannum)):
        return None
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
        Acquisition master label, omitted from the identifier if None.
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
        Raw-observation file identifier, or None if any observation number
        or the interface is missing.
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
        Nullable string columns ``quartet``, ``uid_obs``, ``uid_raw_obs`` and
        ``uid_raw_obs_file``, preserving the input index and row order.
        Quartets allow omitted trailing numbers. Each UID is null if components
        required for its scope are missing; a missing master is omitted.
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
                ("uid_obs", make_uid_obs, 2),
                ("uid_raw_obs", make_uid_raw_obs, 4),
                ("uid_raw_obs_file", make_uid_raw_obs_file, 5),
            )
        },
        index=master.index,
    )
