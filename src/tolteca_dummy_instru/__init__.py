"""Opt-in synthetic instrument for testing TolTECA integrations."""

__all__ = ["register"]

_registered = False


def register() -> None:
    """Enable dummy support in generic file access and configured discovery.

    Notes
    -----
    Call before using dummy support in generic file access or configured
    discovery. Repeated calls are harmless. Importing this package does not
    construct or register its data-model definition.

    Examples
    --------
    >>> from tolteca_dummy_instru import register
    >>> register()
    >>> from tolteca_dummy_instru.file import DummyInstruFileAccessor
    >>> file_accessor = DummyInstruFileAccessor.from_sources([
    ...     "chan01-tcs-1-0-2-20251031T120000.nc",
    ...     "tel_dummy_instru_2025-10-31_000001_00_0002.nc",
    ... ])
    >>> file_accessor.table.interface.tolist()
    ['chan01', 'tel']
    """
    global _registered  # noqa: PLW0603 -- explicit, process-local opt-in.
    if _registered:
        return
    from tolteca_datamodels.data_model import data_models  # noqa: PLC0415
    from tolteca_datamodels.input_discovery import InputDiscoveryConfig  # noqa: PLC0415
    from tolteca_datamodels.lmt.file_store import LmtFileStore  # noqa: PLC0415

    from .data_model import data_model_spec  # noqa: PLC0415
    from .file_store import DummyInstruFileStore  # noqa: PLC0415
    from .input_discovery import DummyInstruInputDiscovery  # noqa: PLC0415
    from .types import DummyInstruInfo  # noqa: PLC0415

    data_models.register(data_model_spec)
    LmtFileStore.register_substore(
        DummyInstruFileStore, DummyInstruInfo.instru, name="dummy_instru"
    )
    InputDiscoveryConfig.register_input(
        "data_dummy_instru_path", DummyInstruInputDiscovery
    )
    _registered = True
