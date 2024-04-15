from tollan.config.types import AbsDirectoryPath

from .pipeline import StepConfig

__all__ = [
    "OutputConfig",
    "OutputMixin",
]


class OutputConfig(StepConfig):
    """A base model for output config."""

    rootpath: None | AbsDirectoryPath = None

    def make_path(self, name, rootpath=None):
        """Return path."""
        rootpath = rootpath or self.rootpath
        return rootpath.joinpath(name)


class OutputMixin:
    """A mixin class for output step."""
