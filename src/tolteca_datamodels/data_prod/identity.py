"""Observation, acquisition and named-collection product identifiers."""

import re
from typing import cast

from pydantic import TypeAdapter, ValidationError

from ..lmt_base.funcs import make_uid_obs, make_uid_raw_obs
from .types import NamedGroupUid

__all__ = ["make_obs_product_uid", "make_raw_obs_product_uid", "normalize_primary_uid"]


_NAMED_GROUP_UID = TypeAdapter(NamedGroupUid)
_MASTER_LABEL_PATTERN = r"[A-Za-z][A-Za-z0-9]*"


def normalize_primary_uid(primary_uid: str) -> str:
    """Return a canonical observation, acquisition or named-collection UID.

    Parameters
    ----------
    primary_uid : str
        Explicit master followed by an observation number, or by observation,
        subobservation and scan numbers, or a named collection's UUID.
        For example, ``TCS-00145609-000-0001``. A master starts with an ASCII
        letter and contains only ASCII letters and digits.

    Returns
    -------
    str
        Lowercase master and unpadded numbers, such as ``tcs-145609-0-1``.
        The master remains part of the identifier, including for TCS. Named
        collection UUIDs use their canonical string representation.

    Raises
    ------
    ValueError
        The value does not identify one observation or acquisition with an
        explicit master or a named collection. Directory labels with an omitted
        master are not UIDs.

    Examples
    --------
    >>> from tolteca_datamodels.data_prod import normalize_primary_uid
    >>> normalize_primary_uid("TCS-00145609-000-0001")
    'tcs-145609-0-1'
    >>> normalize_primary_uid("ICS-00001")
    'ics-1'
    """
    if (
        re.fullmatch(rf"{_MASTER_LABEL_PATTERN}-[0-9]+(?:-[0-9]+-[0-9]+)?", primary_uid)
        is None
    ):
        try:
            return str(_NAMED_GROUP_UID.validate_python(primary_uid))
        except ValidationError:
            msg = (
                f"Invalid primary UID {primary_uid!r}: expected an observation or "
                "acquisition with an explicit master, or a named collection UUID"
            )
            raise ValueError(msg) from None
    master, *numbers = primary_uid.split("-")
    return "-".join((master.lower(), *(str(int(number)) for number in numbers)))


_ENTITY_UID_PARTS = {"dp_obs": 2, "dp_raw_obs": 4}


def _normalize_entity_uid(uid, data_prod_type=None):
    """Return a canonical observation or acquisition UID, never a collection's.

    With a declared type, the UID must have that type's components.
    """
    match = re.fullmatch(rf"{_MASTER_LABEL_PATTERN}((?:-[0-9]+){{1,3}})", uid)
    n_parts = None if match is None else match.group(1).count("-") + 1
    if n_parts not in (
        _ENTITY_UID_PARTS.values()
        if data_prod_type is None
        else (_ENTITY_UID_PARTS[data_prod_type],)
    ):
        kind = data_prod_type or "observation or acquisition"
        msg = f"Invalid {kind} UID {uid!r}"
        raise ValueError(msg)
    return normalize_primary_uid(uid)


def _validate_observation_identity(master, *numbers):
    """Require a master label and nonnegative integer observation numbers."""
    if (
        not isinstance(master, str)
        or re.fullmatch(_MASTER_LABEL_PATTERN, master) is None
        or any(not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in numbers)
    ):
        msg = (
            "An observation requires a master starting with an ASCII letter and "
            "containing only ASCII letters and digits, and nonnegative integer "
            "observation numbers"
        )
        raise ValueError(msg)


def make_obs_product_uid(master: str, obsnum: int) -> str:
    """Return the location-independent identifier of an observation product.

    Parameters
    ----------
    master : str
        Acquisition master, such as ``tcs`` or ``ics``. Starts with an ASCII
        letter and contains only ASCII letters and digits.
    obsnum : int
        Observation number.

    Returns
    -------
    str
        Lowercase master followed by the unpadded observation number.

    Raises
    ------
    ValueError
        The master is invalid or the observation number is not a nonnegative
        integer.
    """
    _validate_observation_identity(master, obsnum)
    return cast("str", make_uid_obs(master.lower(), obsnum))


def make_raw_obs_product_uid(
    master: str, obsnum: int, subobsnum: int, scannum: int
) -> str:
    """Return the location-independent identifier of a raw-observation product.

    Parameters
    ----------
    master : str
        Acquisition master, such as ``tcs`` or ``ics``. Starts with an ASCII
        letter and contains only ASCII letters and digits.
    obsnum, subobsnum, scannum : int
        Observation, subobservation and scan numbers.

    Returns
    -------
    str
        Acquisition quartet with a lowercase master.

    Raises
    ------
    ValueError
        The master is invalid or an observation number is not a nonnegative
        integer.
    """
    _validate_observation_identity(master, obsnum, subobsnum, scannum)
    return cast("str", make_uid_raw_obs(master.lower(), obsnum, subobsnum, scannum))
