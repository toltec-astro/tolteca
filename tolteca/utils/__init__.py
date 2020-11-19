#!/usr/bin/env python

from tollan.utils.sys import touch_file
from tollan.utils.fmt import pformat_yaml
from tollan.utils import rupdate
from tollan.utils.log import get_logger, logit

import astropy.units as u
from yaml.dumper import SafeDumper
import appdirs
from pathlib import Path, PosixPath
from datetime import datetime
from cached_property import cached_property
from schema import Schema, Use
import inspect
import yaml
import re
import os


def get_pkg_data_path():
    """Return the package data path."""
    return Path(__file__).parent.parent.joinpath("data")


def get_user_data_dir():
    return Path(appdirs.user_data_dir('tolteca', 'toltec'))


class RuntimeContextError(Exception):
    """Raise when errors occur in `RuntimeContext`."""
    pass


class RuntimeContext(object):
    """A class that holds runtime context for pipeline."""

    _file_contents = {
            'logdir': 'log',
            'bindir': 'bin',
            'caldir': 'cal',
            'setup_file': '50_setup.yaml'
            }
    _backup_items = ['setup_file', ]
    _backup_time_fmt = "%Y%m%dT%H%M%S"

    logger = get_logger()

    def __init__(self, rootpath):
        if isinstance(rootpath, str):
            rootpath = Path(rootpath)
        self.rootpath = rootpath.resolve()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.rootpath})"

    def __getattr__(self, name, *args):
        if name in self._file_contents:
            return self._get_content_path(self.rootpath, name)
        return super().__getattribute__(name, *args)

    @classmethod
    def _get_content_path(cls, rootpath, item):
        return rootpath.joinpath(cls._file_contents[item])

    @property
    def config_files(self):
        """The list of config files present in the rootpath.

        Files with names match ``\\d+_.+\\.ya?ml$`` in the :attr:`rootpath`
        are returned.
        """
        return sorted(filter(
            lambda p: re.match(r'\d+_.+\.ya?ml$', p.name),
            self.rootpath.iterdir()))

    def to_dict(self):
        """Return a dict representation of the runtime context."""
        return {
                attr: getattr(self, attr)
                for attr in [
                    'rootpath',
                    ] + list(self._file_contents.keys())
                }

    @cached_property
    def config(self):
        """The config dict, created by merging all :attr:`config_files`.

        """
        config_files = self.config_files
        self.logger.debug(
                f"load config from files: {pformat_yaml(config_files)}")
        if len(config_files) == 0:
            raise RuntimeContextError('no config file found.')
        cfg = dict()
        for f in self.config_files:
            with open(f, 'r') as fo:
                rupdate(cfg, yaml.safe_load(fo))
        cfg = self.validate_config(cfg)
        # update runtime info
        cfg['runtime'] = self.to_dict()
        self.logger.debug(f"loaded config: {pformat_yaml(cfg)}")
        return cfg

    @classmethod
    def validate_config(cls, cfg):
        return cls.get_config_schema().validate(cfg)

    @classmethod
    def get_config_schema(cls):
        """Return a `schema.Schema` object that validates the config dict.

        This combines all :attr:`_config_schema` defined in all
        base classes.
        """
        # merge schema
        d = dict()
        for base in reversed(inspect.getmro(cls)):
            if not hasattr(base, 'extend_config_schema'):
                continue
            s = base.extend_config_schema()
            if s is None:
                continue
            if isinstance(s, Schema):
                s = s.schema
            rupdate(d, s)
        return Schema(d)

    @classmethod
    def extend_config_schema(cls):
        # this defines a basic schema to validate the config

        def validate_setup(cfg_setup):
            # TODO implement more logic to verify the settings
            from ..version import version
            # for now we just issue a warning but this will be replaced
            # by actual version comparison.
            if cfg_setup['version'] != version:
                cls.logger.warning(
                        f"mismatch of tolteca version "
                        f"{cfg_setup['version']} -> {version}")
            return cfg_setup

        return {
            'setup': Use(validate_setup),
            object: object
            }

    @cached_property
    def yaml_dumper(self):

        class yaml_dumper(SafeDumper):
            """Yaml dumper that handles some additional types."""
            pass

        yaml_dumper.add_representer(
                PosixPath, lambda s, p: s.represent_str(p.as_posix()))
        yaml_dumper.add_representer(
                u.Quantity, lambda s, q: s.represent_str(q.to_string()))
        return yaml_dumper

    @classmethod
    def _create_backup(cls, path, dry_run=False):
        timestamp = datetime.fromtimestamp(
            path.lstat().st_mtime).strftime(
                cls._backup_time_fmt)
        backup_path = path.with_name(
                f"{path.name}.{timestamp}"
                )
        with logit(cls.logger.info, f"backup {path} -> {backup_path}"):
            if not dry_run:
                os.rename(path, backup_path)
        return backup_path

    @classmethod
    def from_dir(
            cls, dirpath,
            create=False, force=False, overwrite=False, dry_run=False
            ):
        """
        Create `PipelineRuntime` instance from `dirpath`.

        Parameters
        ----------
        path : `pathlib.Path`
            The path to the work directory.

        create : bool
            When set to False, raise `RuntimeContextError` if `path` does not
            already have all content items. Otherwise, create the missing ones.

        force : bool
            When False, raise `RuntimeContextError` if `dirpath` is not empty

        overwrite : bool
            When False, backups for existing files is created.

        dry_run : bool
            If True, no actual file system changed is made.

        kwargs : dict
            Keyword arguments passed directly into the created
            config file.
        """

        path_is_ok = False
        if dirpath.exists():
            if dirpath.is_dir():
                try:
                    next(dirpath.iterdir())
                except StopIteration:
                    # empty dir
                    path_is_ok = True
                else:
                    # nonempty dir
                    if not force:
                        raise RuntimeContextError(
                                f"path {dirpath} is not empty. Set"
                                f" force=True to proceed anyways")
                    path_is_ok = True
            else:
                # not a dir
                raise RuntimeContextError(
                        f"path {dirpath} exists but is not a valid directory."
                        )
        else:
            # non exists
            path_is_ok = True
        assert path_is_ok  # should not fail

        def get_or_create_item_path(item, path, dry_run=False):
            if path.exists():
                cls.logger.debug(
                    f"{'overwrite' if item in cls._backup_items else 'use'}"
                    f" existing {item} {path}")
            else:
                with logit(cls.logger.debug, f"create {item} {path}"):
                    if not dry_run:
                        if item.endswith('dir'):
                            path.mkdir(parents=True, exist_ok=False)
                        elif item.endswith('file'):
                            touch_file(path)
                        else:
                            raise ValueError(f"unknown {item}")

        if create:
            for item in cls._backup_items:
                content_path = cls._get_content_path(dirpath, item)
                if content_path.exists():
                    if not overwrite:
                        cls._create_backup(content_path)

        for item in cls._file_contents.keys():
            content_path = cls._get_content_path(dirpath, item)
            if not create and not content_path.exists():
                raise RuntimeContextError(
                        f"unable to initialize pipeline runtime"
                        f" from {dirpath}:"
                        f" missing {item} {content_path}. Set"
                        f" create=True to create missing items")
            if create:
                get_or_create_item_path(item, content_path)

        return cls(dirpath)
