"""Reusable LMT observation functions."""

__all__ = ["make_quartet", "make_uid_raw_obs", "make_uid_raw_obs_file"]


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
