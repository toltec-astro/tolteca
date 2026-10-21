"""Configured input discovery with built-in LMT and TolTEC entries."""

from . import data_model as _data_model  # noqa: F401 -- enable built-in capabilities.
from .core.input_discovery import InputDiscovery, InputDiscoveryConfig, query_file_store
from .lmt.input_discovery import LmtInputDiscovery as _LmtInputDiscovery
from .lmt_base.input_discovery import SourcesInputDiscovery as _SourcesInputDiscovery
from .toltec.input_discovery import ToltecInputDiscovery as _ToltecInputDiscovery

__all__ = ["InputDiscovery", "InputDiscoveryConfig", "query_file_store"]

InputDiscoveryConfig.register_input("data_lmt_path", _LmtInputDiscovery)
InputDiscoveryConfig.register_input("data_toltec_path", _ToltecInputDiscovery)
InputDiscoveryConfig.register_input("sources", _SourcesInputDiscovery)
