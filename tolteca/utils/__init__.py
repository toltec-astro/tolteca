#!/usr/bin/env python

from tollan.utils.fmt import pformat_yaml
from tollan.utils import rupdate
from tollan.utils.log import get_logger
from tollan.utils.dirconf import DirConfError, DirConfMixin

from ..version import version
from astropy.time import Time
import appdirs
from pathlib import Path
from cached_property import cached_property
from schema import Use, Optional
from copy import deepcopy
import yaml


def get_pkg_data_path():
    """Return the package data path."""
    return Path(__file__).parent.parent.joinpath("data")


def get_user_data_dir():
    return Path(appdirs.user_data_dir('tolteca', 'toltec'))


class RuntimeContextError(DirConfError):
    """Raise when errors occur in `RuntimeContext`."""
    pass


class RuntimeContext(DirConfMixin):
    """A class to manage runtime contexts.

    This class manages a set of configurations in a coherent way, providing
    per-project persistence for user settings.

    A runtime context can be constructed either from file system,
    using the `tollan.utils.dirconf.DirConfMixin` under the hood, or
    be constructed directly from a configuration dict composed
    programatically. Property :attr:``is_persistent`` is set to True in the
    former case.

    """

    _contents = {
        'bindir': {
            'path': 'bin',
            'type': 'dir',
            'backup_enabled': False
            },
        'caldir': {
            'path': 'cal',
            'type': 'dir',
            'backup_enabled': False
            },
        'logdir': {
            'path': 'log',
            'type': 'dir',
            'backup_enabled': False
            },
        'setup_file': {
            'path': '50_setup.yaml',
            'type': 'file',
            'backup_enabled': True
            },
        }

    logger = get_logger()

    def __init__(self, rootpath=None, config=None):
        if sum([rootpath is None, config is None]) != 1:
            raise RuntimeContextError(
                    "one and only one of rootpath and config has to be set")
        if rootpath is not None:
            # we expect that rootpath is already setup if constructed
            # this way.
            try:
                rootpath = self.populate_dir(
                        rootpath,
                        create=False, force=True)
            except DirConfError:
                raise RuntimeContextError(
                        f'missing runtime context contents in {rootpath}. '
                        f'Use {self.__class__.__name__}.from_dir '
                        f'with create=True instead.'
                        )
        elif config is not None:
            # make our own copy because we may update it
            config = deepcopy(config)
        self._rootpath = rootpath
        # we delay the validating of config to accessing time
        # in property ``config``
        self._config = config
        if not self.is_persistent and not self._config_has_setup(config):
            # when user provide config directly
            # it is likely that its not setup, therefore
            # just do it here if not already
            self.setup()

    @property
    def is_persistent(self):
        """True if this context is created from a valid rootpath."""
        return self._rootpath is not None

    @property
    def rootpath(self):
        if self.is_persistent:
            return self._rootpath
        # the config runtime should always be available,
        # since we add that at the end of the config property getter
        return self.config['runtime']['rootpath']

    def __getattr__(self, name, *args):
        # in case the config is not persistent,
        # we return the content paths from the runtime dict
        # make available the content attributes
        if self.is_persistent:
            return super().__getattr__(name, *args)
        if name in self._contents.keys():
            return self.config['runtime'][name]
        return super().__getattribute__(name, *args)

    def __repr__(self):
        if self.is_persistent:
            return f"{self.__class__.__name__}({self.rootpath})"
        # an extra star to indication is not persistent
        return f"{self.__class__.__name__}(*{self.rootpath})"

    @property
    def config_files(self):
        """The list of config files present in the :attr:`config_files`.

        Returns ``None`` if the runtime context is not persistent.

        """
        if self.is_persistent:
            return self.collect_config_files()
        return None

    @cached_property
    def config(self):
        """The runtime context dict.

        """
        if self.is_persistent:
            cfg = self.collect_config_from_files(
                    self.config_files, validate=True
                    )
            # update runtime info
            cfg['runtime'] = self.to_dict()
        else:
            cfg = self.validate_config(self._config)
            # here we also add the runtime dict if it not already exists
            if 'runtime' not in cfg:
                cfg['runtime'] = {
                        attr: None
                        for attr in self._get_to_dict_attrs()
                        }
        self.logger.debug(f"loaded config: {pformat_yaml(cfg)}")
        return cfg

    @classmethod
    def _validate_setup(cls, cfg_setup):
        # TODO implement more logic to verify the settings
        if cfg_setup is None:
            cfg_setup = {}

        # check version
        from ..version import version
        # for now we just issue a warning but this will be replaced
        # by actual version comparison.
        if 'version' not in cfg_setup:
            cls.logger.warning("no version info found.")
            cfg_setup['version'] = version
        if cfg_setup['version'] != version:
            cls.logger.warning(
                    f"mismatch of tolteca version "
                    f"{cfg_setup['version']} -> {version}")
        return cfg_setup

    @classmethod
    def extend_config_schema(cls):
        # this defines a basic schema to validate the config
        return {
            'setup': Use(cls._validate_setup),
            Optional(object): object
            }

    @classmethod
    def from_dir(
            cls, dirpath, init_config=None, **kwargs
            ):
        """
        Create `RuntimeContext` instance from `dirpath`.

        This is the preferred method to construct `RuntimeContext`
        from arbitrary path.

        For paths that have already setup previous as runtime context,
        use the constructor instead.

        Parameters
        ----------
        dirpath : `pathlib.Path`, str
            The path to the work directory.
        init_config : dict, optional
            The dict to add to setup_file.
        **kwargs : dict, optional
            Additional arguments passed to the underlying
            :meth:`DirConfMixin.populate_dir`.
        """
        dirpath = cls.populate_dir(dirpath, **kwargs)
        if kwargs.get('dry_run', False):
            # when dry_run we just create a in-memory rc
            cfg = {
                    'runtime': {
                        'rootpath': dirpath
                        }
                    }
            if init_config is not None:
                rupdate(cfg, init_config)
            return cls(config=cfg)
        # write init_config to setup_file
        if init_config is not None:
            setup_file = cls._resolve_content_path(dirpath, 'setup_file')
            with open(setup_file, 'r') as fo:
                cfg = yaml.safe_load(fo)
                if cfg is None or not isinstance(cfg, dict):
                    cfg = dict()
                rupdate(cfg, init_config)
            cls.write_config_to_yaml(
                    cfg,
                    setup_file,
                    overwrite=True)
        return cls(rootpath=dirpath)

    @classmethod
    def from_config(cls, *configs):
        """
        Create `RuntimeContext` instance from a set of configs.

        This method allow constructing `RuntimeContext`
        from multiple configuration dicts.

        For a single config dict, use the constructor instead.

        Parameters
        ----------
        *configs : tuple
            The config dicts.
        """
        cfg = deepcopy(configs[0])
        # TODO maybe we nned to make deepcopy of all?
        for c in configs[1:]:
            rupdate(cfg, c)
        return cls(config=cfg)

    def symlink_to_bindir(self, src, link_name=None):
        """Create a symbolic link of of `src` in :attr:`bindir`.

        """
        src = Path(src)
        if link_name is None:
            link_name = src.name
        dst = self.bindir.joinpath(link_name)
        dst.symlink_to(src)  # note this may seem backward but it is the way
        self.logger.debug(f"symlink {src} to {dst}")
        return dst

    @staticmethod
    def _config_has_setup(config):
        """Return True if config has been setup."""
        return isinstance(config, dict) and 'setup' in config

    def setup(self, config=None, overwrite=False):
        """Populate the setup file (50_setup.yaml).

        Parameters
        ----------
        config : dict, optional
            Additional config to add to the setup file.
        overwrite : bool
            Set to True to force overwrite the existing
            setup info. Otherwise a `RuntimeContextError` is
            raised.
        """
        # check if already setup
        if self.is_persistent:
            with open(self.setup_file, 'r') as fo:
                setup_cfg = yaml.safe_load(fo)
        else:
            # use the unvalidated version here so we can setup for the
            # first time
            setup_cfg = self._config
        if self._config_has_setup(setup_cfg):
            if overwrite:
                self.logger.debug(
                    "runtime context is already setup, overwrite")
            else:
                raise RuntimeContextError(
                        'runtime context is already setup, '
                        'use overwrite=True to re-setup.'
                        )
        if config is None:
            config = dict()
        else:
            config = deepcopy(config)
        rupdate(
            config,
            {
                'setup': {
                    'version': version,
                    'created_at': Time.now().isot,
                    }
            })
        if self.is_persistent:
            # write the setup context to the setup_file
            with open(self.setup_file, 'w') as fo:
                yaml.dump(config, fo)
        else:
            # create setup key in the config
            rupdate(self._config, config)
        # invalidate the config cache if needed
        # so later self.config will pick up the new setting
        if 'config' in self.__dict__:
            del self.__dict__['config']
        return self

    def update(self, config, config_file=None, overwrite=False):
        """Populate the `config_file` with `config`.

        Parameters
        ----------
        config : dict
            Config to add to `config_file`.
        config_file : str, `pathlib.Path`, optional
            Config to add to `config_file`. When self is not persistent
            this has to be set to None.
        overwrite : bool
            Set to True to force overwrite the existing
            config. Otherwise a `RuntimeContextError` is
            raised.
        """
        if self.is_persistent:
            with open(config_file, 'r') as fo:
                cfg = yaml.safe_load(fo)
        else:
            cfg = self._config
        if config is None:
            config = dict()
        if isinstance(cfg, dict):
            common_keys = set(cfg.keys()).intersection(set(config.keys()))
            if len(common_keys) > 0:
                if overwrite:
                    self.logger.debug(
                        f"keys {common_keys} exists, overwrite")
                else:
                    raise RuntimeContextError(
                        f'keys {common_keys} already exists'
                        )
        rupdate(cfg, config)
        if self.is_persistent:
            # update the config file
            with open(config_file, 'w') as fo:
                yaml.dump(cfg, fo)
        else:
            self._config = cfg
        # invalidate the config cache if needed
        # so later self.config will pick up the new setting
        if 'config' in self.__dict__:
            del self.__dict__['config']
        return self
