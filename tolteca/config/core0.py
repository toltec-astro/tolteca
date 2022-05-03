#!/usr/bin/env python

import sys
import shlex
from typing import Sequence
from pathlib import Path

import schema
from schema import Or, Use

from astropy.time import Time

from dataclasses import dataclass, field

from tollan.utils import ensure_abspath
from tollan.utils.sys import get_username, get_hostname

from ..version import version as current_version


class AbsPathSchema(schema.Schema):
    """A schema to validate absolute path."""
    def __init__(self, check_exists=True, type=None):
        super().__init__(str)
        self._check_exists = check_exists
        self._type = type

    @classmethod
    def _check_path_type(cls, p, type):
        if type == 'dir':
            return p.is_dir()
        if type == 'file':
            return p.is_file()
        if type == 'symlink':
            return p.is_symlink()
        raise schema.SchemaError(f"Path is not {type}: {p}")

    def validate(self, data, **kwargs):
        data = ensure_abspath(super().validate(data))
        if self._check_exists:
            if data.exists():
                if self._type is None:
                    return data
                return self._check_path_type(data, self._type)
            raise schema.SchemaError(f"Path does not exist: {data}")
        return data


@dataclass
class ConfigInfo(object):
    """The config info.

    """
    env_files: Sequence[Path] = field(
        default_factory=list,
        metadata={
            'description': 'The list of env files.',
            'schema': [AbsPathSchema(type='file')]
            }
        )
    config_files: Sequence[Path] = field(
        default_factory=list,
        metadata={
            'description': 'The list of config files.',
            'schema': [AbsPathSchema(type='file')]
            }
        )
    runtime_context_dir: Path = field(
        default=None,
        metadata={
            'description': 'The path to load runtime context from.',
            'schema': Or(None, AbsPathSchema(type='dir')),
            }
        )

    class Meta:
        schema = {
            'description': 'The config info.'
            }


@dataclass
class SetupInfo(object):
    """The info saved to setup file.

    """

    created_at: Time = field(
        default_factory=Time.now,
        metadata={
            'description': 'The time of setup.',
            'schema': Use(Time)
            }
        )
    config: dict = field(
        default_factory=dict,
        metadata={
            'description': 'The config to be saved.'
            }
        )

    class Meta:
        schema = {
            'description': 'The info saved to to setup file.'
            }


@dataclass
class RuntimeInfo(object):
    """The runtime info.

    """

    version: str = field(
        default=current_version,
        metadata={
            'description': 'The current version.',
            }
        )
    created_at: Time = field(
        default_factory=Time.now,
        metadata={
            'description': 'The time the runtime info is created.',
            'schema': Use(Time)
            }
        )
    username: str = field(
        default_factory=get_username,
        metadata={
            'description': 'The current username.',
            }
        )
    hostname: str = field(
        default_factory=get_hostname,
        metadata={
            'description': 'The system hostname.',
            }
        )
    python_prefix: Path = field(
        default=ensure_abspath(sys.prefix),
        metadata={
            'schema': AbsPathSchema(type='dir'),
            'description': 'The path to the python installation.'
            }
        )
    exec_path: Path = field(
        default=ensure_abspath(sys.argv[0]),
        metadata={
            'schema': AbsPathSchema(type='file'),
            'description': 'Path to the commandline executable.'
            }
        )
    cmd: str = field(
        default=shlex.join(sys.argv),
        metadata={
            'description': 'The full commandline.'
            }
        )
    bindir: Path = field(
        default=None,
        metadata={
            'schema': Or(None, AbsPathSchema(type='dir')),
            'description': 'The directory to look for external routines.'
            }
        )
    caldir: Path = field(
        default=None,
        metadata={
            'schema': Or(None, AbsPathSchema(type='dir')),
            'description': 'The directory to hold calibration files.'
            }
        )
    config_info: ConfigInfo = field(
        default_factory=ConfigInfo,
        metadata={
            'description': 'The config info dict.',
            }
        )
    setup_info: SetupInfo = field(
        default_factory=SetupInfo,
        metadata={
            'description': 'The setup info dict.',
            }
        )

    class Meta:
        schema = {
            'description': 'The info related to runtime context.'
            }
