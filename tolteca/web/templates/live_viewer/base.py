#!/usr/bin/env python

from tollan.utils import getobj


def _add_from_name_factory(cls):
    """A helper decorator to add ``from_name`` factory method to class."""
    cls._subclasses = dict()

    def _init_subclass(cls, name):
        cls._subclasses[name] = cls
        cls.name = name

    cls.__init_subclass__ = classmethod(_init_subclass)

    def from_name(cls, name):
        """Return the site instance for `name`."""
        # try load the object in this current folder
        getobj(f"{__package__}.{name}")
        if name not in cls._subclasses:
            raise ValueError(f"invalid obs site name {name}")
        subcls = cls._subclasses[name]
        return subcls()

    cls.from_name = classmethod(from_name)

    return cls


class LiveViewerModule(object):
    """Base class for modules in live viewer."""

    def make_viewer(self, container, **kwargs):
        return container.child(self.ViewerPanel(self, **kwargs))

    def make_viewer_controls(self, container, **kwargs):
        return container.child(self.ViewerControlPanel(self, **kwargs))

    def make_controls(self, container, **kwargs):
        return container.child(self.ControlPanel(self, **kwargs))

    def make_info_display(self, container, **kwargs):
        return container.child(self.InfoPanel(self, **kwargs))

    InfoPanel = NotImplemented
    """
    Subclass should implement this as a `ComponentTemplate` to generate UI
    for display read-only custom info for this module.
    """

    ControlPanel = NotImplemented
    """
    Subclass should implement this class as a ComponentTemplate
    to generate UI for collecting user inputs.

    The template should have an attribute ``info_store`` of type
    dcc.Store, which is updated when user input changes.
    """

    ViewerPanel = NotImplemented
    """
    Subclass should implement this as a `ComponentTemplate` to generate UI
    for presenting the live views.
    """

    ViewerControlPanel = NotImplemented
    """
    Subclass should implement this as a `ComponentTemplate` to generate UI
    for operating with the live views.
    """

    display_name = NotImplemented
    """Name to display as title of the module."""


@_add_from_name_factory
class ObsSite(LiveViewerModule):
    """A module base class for an observation site"""

    observer = NotImplemented
    """The astroplan.Observer instance for the site."""

    @classmethod
    def get_observer(cls, name):
        return cls._subclasses[name].observer


@_add_from_name_factory
class ObsInstru(LiveViewerModule):
    """A module base class for an observation instrument."""

    @classmethod
    def make_traj_data(cls, exec_config):
        """Subclass should implement this to generate traj_data
        as part of the result of `ObsPlannerExecConfig.make_traj_data`.
        """
        return NotImplemented
