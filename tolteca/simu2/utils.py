#!/usr/bin/env python

import yaml
from collections import UserDict

from tollan.utils import rupdate
from tollan.utils.fmt import pformat_yaml


class PersistentState(UserDict):
    """A class to persist state as YAML file.
    """

    def __init__(self, filepath, init=None, update=None):
        if filepath.exists():
            with open(filepath, 'r') as fo:
                state = yaml.safe_load(fo)
            if update is not None:
                rupdate(state, update)
        elif init is not None:
            state = init
        else:
            raise ValueError("cannot initialize/load persistent state")
        self._filepath = filepath
        super().__init__(state)

    def sync(self):
        """Update the YAML file with the state."""
        with open(self._filepath, 'w') as fo:
            yaml.dump(self.data, fo)
        return self

    def reload(self):
        """Update the state with the YAML file."""
        with open(self._filepath, 'r') as fo:
            state = yaml.load(fo)
        self.data = state

    def __str__(self):
        return pformat_yaml({
            'state': self.data,
            'filepath': self.filepath})

    @property
    def filepath(self):
        return self._filepath
