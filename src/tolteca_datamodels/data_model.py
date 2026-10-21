"""Registered data models and workflow selection.

Look up built-in capabilities with ``data_models.toltec`` or
``data_models.lmt``. Use ``data_models[name]`` for any registered name and
`DataModelSet.from_select` to select capabilities for a workflow.
Optional extensions appear after they register.
"""

from tollan.utils.registry import RegistryView as _RegistryView

from .core.data_model import DataModelSet
from .core.data_model import DataModelSpec as _DataModelSpec
from .core.data_model import data_models as _registered_data_models
from .lmt.data_model import data_model as _lmt
from .toltec.data_model import data_model as _toltec

__all__ = ["DataModelSet", "data_models"]

_lmt.register()
_toltec.register()


class _DataModels(_RegistryView[str, _DataModelSpec]):
    @property
    def lmt(self) -> _DataModelSpec:
        """Registered LMT telescope capabilities."""
        return self[_lmt.name]

    @property
    def toltec(self) -> _DataModelSpec:
        """Registered TolTEC capabilities, including telescope companions."""
        return self[_toltec.name]


data_models = _DataModels(_registered_data_models)
"""Read-only lookup of currently registered data models by name."""
