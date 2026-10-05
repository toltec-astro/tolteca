"""Opt-in synthetic instrument for testing TolTECA integrations."""

__all__ = ["register"]

_registered = False


def register() -> None:
    """Enable dummy filename parsing, channel selection and file discovery.

    Notes
    -----
    Call once before using the dummy file store or accessor. Repeated calls
    are harmless. Importing this package alone does not enable the instrument.

    Examples
    --------
    >>> from tolteca_dummy_instru import register
    >>> register()
    >>> from tolteca_dummy_instru.file import DummyInstruFileAccessor
    >>> table = DummyInstruFileAccessor.from_sources([
    ...     "chan01-tcs-1-0-2-20251031T120000.nc",
    ...     "tel_dummy_instru_2025-10-31_000001_00_0002.nc",
    ... ])
    >>> table.interface.tolist()
    ['chan01', 'tel']
    """
    global _registered  # noqa: PLW0603 -- explicit, process-local opt-in.
    if _registered:
        return
    from tolteca_datamodels.input_discovery import InputDiscoveryConfig  # noqa: PLC0415
    from tolteca_datamodels.lmt.file_store import LmtFileStore  # noqa: PLC0415

    from . import file  # noqa: PLC0415, F401 -- register the pandas/xarray accessors.
    from .file_store import DummyInstruFileStore  # noqa: PLC0415
    from .input_discovery import DummyInstruInputDiscovery  # noqa: PLC0415
    from .types import DummyInstruInfo  # noqa: PLC0415

    LmtFileStore.register_substore(DummyInstruInfo.instru, DummyInstruFileStore)
    InputDiscoveryConfig.register_input(
        "data_dummy_instru_path", DummyInstruInputDiscovery
    )
    _registered = True
