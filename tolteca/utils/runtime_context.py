#!/usr/bin/env python

from tollan.utils import odict_from_list, ensure_abspath, rupdate
from tollan.utils.dirconf import (
    DirConfMixin, DirConfPath, DirConfPathType)
from tollan.utils.dataclass_schema import add_schema
from dataclasses import dataclass, field, replace
from cached_property import cached_property
from copy import deepcopy
from pathlib import Path
from tollan.utils.log import get_logger
from tollan.utils.fmt import pformat_yaml
from tollan.utils.sys import get_username, get_hostname
from .misc import get_user_data_dir, get_pkg_data_path
from ..version import version as tolteca_version
from schema import Use, Or
from typing import Sequence
import sys
import shlex
import inspect
from astropy.time import Time
import collections.abc
import shutil


__all__ = [
    'yaml_load', 'yaml_dump',
    'ConfigInfo', 'SetupInfo', 'RuntimeInfo',
    'ConfigBackendError', 'ConfigBackend', 'DirConf', 'FileConf', 'DictConf',
    'RuntimeContextError', 'RuntimeContext',
    'RuntimeBase', 'RuntimeBaseError'
    ]


@add_schema
@dataclass
class ConfigInfo(object):
    """The info related to config items.

    This instance is dynamically created and present in the `RuntimeInfo`.
    """
    env_files: Sequence[Path] = field(
        default_factory=list,
        metadata={
            'description': 'The list of systemd env files',
            'schema': [Use(ensure_abspath)]
            }
        )
    standalone_config_files: Sequence[Path] = field(
        default_factory=list,
        metadata={
            'description': 'The list of non-default config paths',
            'schema': [Use(ensure_abspath)]
            }
        )
    user_config_path: Path = field(
        default=get_user_data_dir().joinpath('tolteca.yaml'),
        metadata={
            'descritpion':
            'The tolteca config file path for the user account.',
            'schema': Use(ensure_abspath),
            }
        )
    sys_config_path: Path = field(
        default=get_pkg_data_path().joinpath('tolteca.yaml'),
        metadata={
            'descritpion': 'The tolteca built-in config file path.',
            'schema': Use(ensure_abspath),
        })
    load_user_config: bool = field(
        default=True,
        metadata={
            'descritpion': 'If False, disable the loading of the user config.'
        })
    load_sys_config: bool = field(
        default=True,
        metadata={
            'descritpion': 'If False, disable the loading of the sys config.'
        })
    runtime_context_dir: Path = field(
        default=None,
        metadata={
                'descritpion': 'The config dir path to load runtime context',
                'schema': Or(None, Use(ensure_abspath)),
                }
        )

    class Meta:
        schema = {
            'ignore_extra_keys': True,
            'description': 'The info related to config items.'
            }


@add_schema
@dataclass
class SetupInfo(object):
    """The info saved to `DirConf` setup_file.

    The setup info contains a copy of the config (including the runtime info)
    dict, and can be used for subsequent runs to detect changes of versions or
    environments that could lead to changes  to check any changes to the
    runtime context
    """

    created_at: Time = field(
        default_factory=Time.now,
        metadata={
            'description': 'The time the setup is done.',
            'schema': Use(Time)
            }
        )
    config: dict = field(
        default_factory=dict,
        metadata={
            'description': 'The config dict at setup time'
            }
        )

    class Meta:
        schema = {
            'ignore_extra_keys': True,
            'description': 'The info saved when setup config dir.'
            }


@add_schema
@dataclass
class RuntimeInfo(object):
    """The info related to `RuntimeContext`.

    A instance of this class is always dynamically created and present in the
    config dict.
    """

    version: str = field(
        default=tolteca_version,
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
            'schema': Use(ensure_abspath),
            'description': 'The path to the python installation.'
            }
        )
    exec_path: Path = field(
        default=ensure_abspath(sys.argv[0]),
        metadata={
            'schema': Use(ensure_abspath),
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
            'schema': Or(None, Use(ensure_abspath)),
            'description': 'The dir to look for external routines.'
            }
        )
    logdir: Path = field(
        default=None,
        metadata={
            'schema': Or(None, Use(ensure_abspath)),
            'description': 'The dir to hold log files.'
            }
        )
    caldir: Path = field(
        default=get_user_data_dir(),
        metadata={
            'schema': Or(None, Use(ensure_abspath)),
            'description': 'The dir to hold external data files.'
            }
        )
    config_info: ConfigInfo = field(
        default_factory=ConfigInfo,
        metadata={
            'description': 'The dict contains the config info.',
            }
        )
    setup_info: SetupInfo = field(
        default_factory=SetupInfo,
        metadata={
            'description': 'The dict contains the setup info.',
            }
        )

    class Meta:
        schema = {
            'ignore_extra_keys': True,
            'description': 'The info related to runtime context.'
            }


# we need the dir config compatiable yaml loader and dumpers for
# all config backends
yaml_load = DirConfMixin.yaml_load
yaml_dump = DirConfMixin.yaml_dump


class ConfigBackendError(RuntimeError):
    pass


class ConfigBackend(object):
    """A base class that defines the interface for config handling object.

    This class manages three config dicts internally: ``_default_config``,
    ``_config_impl``, and ``_override_config``, and the cached property
    :attr:``config`` returns a merged dict of the three.

    :meth:`reload` can be used to re-build the config dict.
    """
    def __init__(self):
        # build the config cache and runtime info.
        self.load()

    _runtime_info_key = 'runtime_info'
    """The runtime info dict key when serialized."""

    _config_info_key = 'config_info'
    """The config info dict key when serialized."""

    _setup_info_key = 'setup_info'
    """The setup info dict key when serialized."""

    @classmethod
    def _get_runtime_info_from_config(cls, config):
        return RuntimeInfo.schema.load(
            config.get(cls._runtime_info_key, dict()))

    _default_config = NotImplemented
    """Config dict to set defaults.

    This dict gets overridden by the `_config` and `_override_config`."""

    _override_config = NotImplemented
    """Config dict to override `_config`.
    """

    _runtime_info = None
    """The info dict related to runtime context."""

    _config = NotImplemented
    """Config dict to be implemented in subclass."""

    def _make_runtime_info_dict(self, config):
        """Compile the runtime info dict from `config`.

        Subclass may re-implement this to define custom runtime info.
        """
        return config.get(self._runtime_info_key, dict())

    @property
    def runtime_info(self):
        """The info dict related to runtime context.

        This is updated by :meth:`load` automatically. Subclass
        should implement the :meth:`_make_runtime_info_dict` method to control
        what goes in the runtime info dict.
        """
        return self._runtime_info

    @property
    def config_info(self):
        """The config info."""
        return self.runtime_info.config_info

    @property
    def setup_info(self):
        """The setup info.
        """
        return self.runtime_info.setup_info

    @cached_property
    def config(self):
        """
        Cached config dict as composed from `_default_config`, `_config`,
        `_override_config`, and `runtime_info`.

        """
        return self.load()

    def _invalidate_config_cache(self, include_config_impl_cache=True):
        """This has to be called to allow re-build of the config dict."""
        logger = get_logger()
        if 'config' in self.__dict__:
            del self.__dict__['config']
            logger.debug("config cache invalidated")
        if include_config_impl_cache and \
                '_config_impl' in self.__dict__:
            del self.__dict__['_config_impl']
            logger.debug("config impl cache invalidated")

    @cached_property
    def _config_impl(self):
        """The cached config dict implemented in subclasses.
        """
        # This is necessary so that when reload after default_config
        # and override_config changes, we don't have to also reload
        # the config impl.
        return self._config

    def _make_config_from_configs(self):
        """
        Return config dict as composed from `_default_config`, `_config`,
        and `_override_config`.

        """
        cfg = dict()
        rupdate(cfg, self._default_config or dict())
        rupdate(cfg, self._config_impl or dict())
        rupdate(cfg, self._override_config or dict())
        return cfg

    def load(self, update_runtime_info=True, reload_config=True):
        """Build the config dict and the runtime info.

        This invalidate the config cache and update the runtime info
        object accordingly.

        Parameters
        ----------
        update_runtime_info : bool, optional
            If True, the runtime info dict is updated.

        reload_config : bool, optional
            If True, the :attr:`_config_impl` cache is invalidated, which
            triggers a reload of the config.
        """
        self._invalidate_config_cache(include_config_impl_cache=reload_config)
        # this is to just compile the config without dumping the runtimeinfo
        # because we do not know it yet.
        cfg = self._make_config_from_configs()
        # create and update the runtime info
        if update_runtime_info:
            runtime_info = self._runtime_info = RuntimeInfo.schema.load(
                self._make_runtime_info_dict(cfg))
            # dump the runtime info back to the cfg
            rupdate(
                cfg, {
                    self._runtime_info_key: runtime_info.to_dict()
                    }
                )
        return cfg

    def _make_config_for_setup(self, runtime_info_only=False):
        """Create a full config dict with the desired structure containing
        the config dict in the setup info."""
        if runtime_info_only:
            cfg = {
                self._runtime_info_key:
                self.runtime_info.to_dict()
                }
        else:
            cfg = self.config
        # the cfg could already be setup so the setup info
        # may have nested setup info
        # we need to remove that so the size of the dict does not bloat
        cfg[self._runtime_info_key][self._setup_info_key]['config'] = dict()
        return {
                self._runtime_info_key: {
                    self._setup_info_key:
                    replace(self.setup_info, config=cfg).to_dict()
                    }
                }

    def set_default_config(self, cfg):
        """Set the default config dict.

        This will invalidate the config cache.
        """
        self._default_config = cfg
        self.load(reload_config=False, update_runtime_info=False)

    def set_override_config(self, cfg):
        """Set the override config dict.

        This will invalidate the config cache.
        """

        self._override_config = cfg
        self.load(reload_config=False, update_runtime_info=False)

    def update_default_config(self, cfg):
        """Update the default config dict.

        This will invalidate the config cache.
        """
        rupdate(self._default_config, cfg)
        self.load(reload_config=False, update_runtime_info=False)

    def update_override_config(self, cfg):
        """Update the override config dict.

        This will invalidate the config cache.
        """
        rupdate(self._override_config, cfg)
        self.load(reload_config=False, update_runtime_info=False)

    @property
    def is_persistent(self):
        return NotImplemented

    @property
    def rootpath(self):
        return NotImplemented


class DirConf(ConfigBackend, DirConfMixin):
    """The config backend for config files in tolteca workdir.

    A new workdir can be created by calling the factory method
    :meth:`DirConf.from_dir()` with option ``create=True``.

    Once created, it contains a setup.yaml file and a set of
    predefined subdirs ``bin``, ``log``, and ``cal``.

    The setup file is populated with the current runtime context
    info, and saved persistently.

    The user can subsequently add new config files of format
    ``\\d+_.+.ya?ml`` (e.g., 60_simu.yaml). All the found config files
    will be loaded and merged together upon user querying the
    :attr:`config` property.

    The DirConf object is created implicitly when passing a valid path
    which points to a `DirConf` rootpath to the `RuntimeContext` constructor.

    Parameters
    ----------
    rootpath : str, `pathlib.Path`
        The path to the config dir.
    """

    logger = get_logger()

    _contents = odict_from_list([
        DirConfPath(
            label='bindir',
            path_name='bin',
            path_type=DirConfPathType.DIR,
            backup_enabled=True,
            ),
        DirConfPath(
            label='logdir',
            path_name='log',
            path_type=DirConfPathType.DIR,
            backup_enabled=False,
            ),
        DirConfPath(
            label='caldir',
            path_name='cal',
            path_type=DirConfPathType.DIR,
            backup_enabled=False,
            ),
        DirConfPath(
            label='docdir',
            path_name='doc',
            path_type=DirConfPathType.DIR,
            backup_enabled=False,
            ),
        DirConfPath(
            label='setup_file',
            path_name='40_setup.yaml',
            path_type=DirConfPathType.FILE,
            backup_enabled=True,
            ),
        ], key=lambda a: a.label)

    def __init__(self, rootpath):
        self._default_config = dict()
        self._override_config = dict()
        self._rootpath = self.from_populated(dirpath=rootpath)
        super().__init__()

    def _make_runtime_info_dict(self, config):
        # the runtime info for dir config includes the content paths
        runtime_info_dict = super()._make_runtime_info_dict(config)
        rupdate(runtime_info_dict, self.get_content_paths())
        rupdate(
            runtime_info_dict,
            {
                self._config_info_key: {
                    'runtime_context_dir': self.rootpath}
                })
        return runtime_info_dict

    def _make_config_for_setup(self, **kwargs):
        # in dirconf, the setup file contain the
        # file meta description will is better to be removed
        config = super()._make_config_for_setup(**kwargs)
        config[self._runtime_info_key][self._setup_info_key]['config'].pop(
            self._setup_file_meta_key)
        return config

    @property
    def config_files(self):
        """The list of config files in the config dir.

        """
        return self.collect_config_files()

    @property
    def _config(self):
        """The config dict, collected from all the config files.

        """
        cfg = self.collect_config_from_files(
                self.config_files, validate=False
                )
        self.logger.debug(f"collected config: {pformat_yaml(cfg)}")
        return cfg

    @property
    def is_persistent(self):
        return True

    @property
    def rootpath(self):
        return self._rootpath

    @property
    def _setup_file_meta_key(self):
        return self.make_metadata_dict_key(self.setup_file)

    _setup_file_meta_description = inspect.cleandoc(r"""
~~~~~~~~~~~~~~~~~
!!! IMPORTANT !!!
~~~~~~~~~~~~~~~~~

This file is created as part of the tolteca workdir, and the content is
created/managed programatically, therefore should NOT be modified by hand.

This file contains a copy of the config dict when either of the follow was
run the last time:

(1) In python:

```
>>> from tolteca.utils import RuntimeContext
>>> rc = RuntimeContext("/this/path")
>>> rc.setup()
```

(2) In shell:

```
$ cd /this/path
$ tolteca setup
```

User-supplied configurations should be added by creating/updating other
config files in this workdir with filenames matching `\d+_.+.ya?ml`,
e.g., `10_db.yaml``, `60_simu.yaml`, etc. When the same entry exists
in multiple such files, the one with larger leading number takes precedence
(this applies to this 40_setup.yaml file as well).
""")

    @staticmethod
    def _make_setup_meta(rc, desc):
        return {
            'description': desc,
            'created_at': rc.runtime_info.setup_info.created_at,
            'created_in': rc.rootpath,
            'created_by': f"{rc.runtime_info.username}"
                          f"@{rc.runtime_info.hostname}",
            'creator': f'tolteca v{rc.runtime_info.version}',
            }

    def _make_setup_file_meta(self):
        return {self._setup_file_meta_key: self._make_setup_meta(
            self, self._setup_file_meta_description)}

    def update_setup_file(
            self, config, disable_backup=False, merge=True):
        """Update the content of :attr:`setup_file` with `config`.

        This is done with `tollan.utils.rupdate`.

        The setup file will be also populated with some metadata
        if this has not been done yet.

        Parameters
        ----------

        config : dict
            The config dict to add to setup file.

        disable_backup : bool
            If True, no backup file is created.

        merge : bool
            If False, the config overwrites what is in the setup file.
        """
        # here we need to check the content of the setup file so see
        # if it does not have any info yet, which could be the case
        # right after `DirConf.populate_dir(..., create=True)`
        cfg = self.get_config_from_file(self.setup_file)
        if not cfg:
            # in case the cfg contains nothing, we'll populate the
            # setup_file without making a backup
            disable_backup = True
        else:
            if not merge:
                # we need to clean up the cfg so it only contain the runtime
                # info dict
                runtime_info_dict = cfg.get(self._runtime_info_key, dict())
                cfg = self._make_setup_file_meta()
                cfg.update({self._runtime_info_key: runtime_info_dict})
        # we use the DirConfPath to handle the creation of backup.
        filepath = self._contents['setup_file'].create(
            self.rootpath, disable_backup=disable_backup, dry_run=False)
        assert filepath == self.setup_file
        # here we'll check the setup_file for the metadata header
        # and add the meta dict to cfg
        if self._setup_file_meta_key not in cfg:
            rupdate(cfg, self._make_setup_file_meta())
        # update the config to cfg
        rupdate(cfg, config)
        # here we need to overwrite=True to allow write to the existing file
        self.write_config_file(cfg, filepath, overwrite=True)
        # we also need to clear the config cache to make it take effect
        # in the config dict.
        # note that the runtime_info does not change for DirConf
        # in this case because it is not dependent on user config entries.
        # TODO is this really the case?
        # self._invalidate_config_cache()
        self.load(update_runtime_info=True, reload_config=True)

    @classmethod
    def populate_dir(cls, *args, **kwargs):
        dirpath = super().populate_dir(*args, **kwargs)
        # add some custom content to the example folder
        docdir = cls._contents['docdir'].resolve_path(dirpath)
        # reference of all dataclasses.
        with open(
                docdir.joinpath('00_config_dict_references.txt'), 'w') as fo:
            # collect all config types from submodules
            from ..simu import simu_config_item_types
            from ..reduce import redu_config_item_types
            for dcls in [
                    ConfigInfo, SetupInfo, RuntimeInfo
                    ] + simu_config_item_types + redu_config_item_types:
                if hasattr(dcls, 'pformat_schema'):
                    doc = dcls.pformat_schema()
                else:
                    doc = dcls.schema.pformat()
                fo.write(f"\n{doc}\n")
        # example config files
        example_dir = get_pkg_data_path().joinpath('examples')
        for file in [
                '10_db.yaml',
                '60_simu_point_source_lissajous.yaml',
                '61_simu_blank_field_raster.yaml',
                # '62_simu_fits_input_rastajous.yaml',
                '70_redu_simulated_data.yaml'
                ]:
            shutil.copyfile(example_dir.joinpath(file), docdir.joinpath(file))
        # readme file in the rootpath
        shutil.copyfile(
            example_dir.joinpath('workdir_README_template.md'),
            dirpath.joinpath('README.md'))
        return dirpath


class FileConf(ConfigBackend):
    """A config backend for single file.

    """

    def __init__(self, filepath):
        self._default_config = dict()
        self._override_config = dict()
        self._filepath = ensure_abspath(filepath)
        super().__init__()

    @property
    def _config(self):
        try:
            with open(self._filepath, 'r') as fo:
                cfg = yaml_load(fo)
        except Exception as e:
            raise ConfigBackendError(
                f"unable to load YAML config from {self._filepath}: {e}")
        return cfg

    @property
    def is_persistent(self):
        return True

    @property
    def rootpath(self):
        return self._filepath


class DictConf(ConfigBackend):
    """A config backend for python dict.

    """
    def __init__(self, config):
        self._default_config = dict()
        self._override_config = dict()
        self._config = config
        super().__init__()

    @property
    def is_persistent(self):
        return False

    @property
    def rootpath(self):
        # for dict config, the rootpath can be propagated from whatever
        # method that created the config dict. This info is accessible
        # in the runtime_info dict
        return self.runtime_info.config_info.runtime_context_dir


class RuntimeContextError(RuntimeError):
    pass


class RuntimeContext(object):
    """A class to manage configurations.

    This class manages configurations in a coherent way for different
    scenarios. The constructor could take a file path that points to a YAML
    config file, or a tolteca working directory path containing multiple
    config files (most recommended, see below), or an python dict.

    Depending on the type of config source (file/dir/dict), the respective
    `ConfigBackend` object (:attr:`config_backend`) is created and the
    handling of the config is done through it.

    Among the three, the `DirConf` backend provides the most complete features
    and capabilities, via the notion of "tolteca workdir". Using the workdir
    allows saving a snapshot of the full config in the `setup_info` into
    the ``setup.yaml`` file in the workdir. This setup info config is then used
    to check any incompatibility in the software or pipeline versions to ensure
    repeatability of the work. The factory method :meth:`from_dir` can be used
    to create a tolteca workdir from an empty directory. The directory will
    be populated with pre-defined content as outlined in `DirConf._contents`.

    From a runtime context object, the full config dict can be accessed using
    the :attr:`config` property. The properties :attr:`runtime_info`,
    :attr:`config_info`, and :attr:`seup_info` can be used to access the
    information that are related to the current config and setup.

    The class provides a registry that maps itself to subclasses of
    `RuntimeBase`, a consumer class of this class. Objects of
    `RuntimeBase` subclasses can be constructed conveniently using the
    ``__get_item__` interface::

    >>> from tolteca.simu import SimulatorRuntime
    >>> from tolteca.reduce import PipelineRuntime
    >>> rc = RuntimeContext('/path/to/workdir')
    >>> simrt, plrt = rc[DatabaseRuntime, PipelineRuntime]

    Parameters
    ----------

    config : str, `pathlib.Path`, dict
        The config source, can be a file path, a directory path or a
        python dict.
    """
    logger = get_logger()

    yaml_load = yaml_load
    yaml_dump = yaml_dump

    def __init__(self, config):

        if isinstance(config, collections.abc.Mapping):
            self.logger.debug("create runtime context from dict")
            config_path = None
            config_dict = config
            config_backend = DictConf(config=deepcopy(config_dict))
        elif isinstance(config, (str, Path)):
            # just rename it for clarity
            config_path = ensure_abspath(config)
            config_dict = None
            if config_path.is_file():
                self.logger.debug(f"load config from file {config_path}")
                config_backend = FileConf(config_path)
            else:
                self.logger.debug(f"load config from workdir {config_path}")
                config_backend = DirConf(rootpath=config_path)
                # except Exception as e:
                #     raise RuntimeContextError(
                #         f'unable to load config from workdir'
                #         f' {config_path}: {e}. '
                #         f'To create a workdir, use `RuntimeContext.from_dir`'
                #         f'with `create=True` instead'
                #         )
        else:
            raise RuntimeContext(f'invalid config source {config}')
        self._config_path = config_path
        self._config_dict = config_dict
        self._config_backend = config_backend

    def __repr__(self):
        if self.is_persistent:
            return f"{self.__class__.__name__}({self.rootpath})"
        # an extra star to indication is not persistent
        return f"{self.__class__.__name__}(*{self.rootpath})"

    # @classmethod
    # def config_schema(cls):
    #     """The schema of the config.

    #     The :attr:`config` is validated against this schema.
    #     """
    #     return Schema(
    #         {str: object},
    #         description='The base schema that validates any config dict.')

    @property
    def bindir(self):
        """The bin directory."""
        return self.runtime_info.bindir

    @property
    def logdir(self):
        """The log directory."""
        return self.runtime_info.logdir

    @property
    def caldir(self):
        """The cal directory."""
        return self.runtime_info.caldir

    @property
    def config_backend(self):
        """The config backend of this runtime context."""
        return self._config_backend

    @property
    def config(self):
        """The config dict."""
        return self._config_backend.config

    @property
    def is_persistent(self):
        """True if the runtime context is created from a file system path."""
        return self._config_backend.is_persistent

    @property
    def rootpath(self):
        """The path from which the runtime context is created."""
        return self._config_backend.rootpath

    @property
    def runtime_info(self):
        """The runtime info of the context."""
        return self._config_backend.runtime_info

    @property
    def config_info(self):
        """The config info."""
        return self._config_backend.config_info

    @property
    def setup_info(self):
        """The setup info.

        For runtime context created from a tolteca workdir, it makes use
        of the ``setup_file`` (the setup.yaml file) to store a copy of
        the config dict.

        This config dict is consulted in the :meth:`check` function
        to detect any changes in the runtime context which may affect
        the correctness of the program.
        """
        return self._config_backend.setup_info

    def get_setup_rc(self):
        """Return the runtime context constructed from the setup info.
        """
        cfg = self.setup_info.config
        if not cfg:
            return None
        return RuntimeContext(config=cfg)

    def setup(
            self,
            overwrite=False,
            runtime_info_only=False,
            setup_filepath=None,
            overwrite_setup_file=False,
            backup_setup_file=True):
        """Save the config dict to the setup info dict.

        This will behave differently for different type of config backend.

        For `DirConf`, the config dict is saved to the ``setup.yaml`` file,
        so subsequent creation of the `RuntimeContext` from the workdir will
        be able to access the config dict via the :attr:`setup_info`.

        For `FileConf` and `DirConf`, this will only update the in-memory
        setup info dict and it won't get saved persistently, unless the
        `setup_filepath` is specified, in which case the dict is saved in the
        specified path. The setup file can be subsequently used together
        with other config files using ``-c`` option for the tolteca CLI so
        the full :meth:`check` can be performed.

        Parameters
        ----------

        overwrite : bool
            Set to True to force overwrite if previous setup config exists.
            Otherwise a `RuntimeContextError` is raised.
            Note that this is not related to whether to overwrite the
            `setup_filepath`, which can be specified by `overwrite_setup_file`.

        runtime_info_only : bool
            If True, only dump the runtime info dict to the setup config.

        setup_filepath : str, `pathlib.Path`, optional
            If set, and use this filepath to save the config dict.
            For `DirConf` config backend, this will be used instead of the
            default `setup.yaml` file in the workdir if specified. For
            `FileConf` and `DictConf`, this has to be a valid path in order to
            save the config dict.

        overwrite_setup_file : bool
            If True, the setup file is overwritten if exists. Otherwise
            `RuntimeContextError` is raised. This is not to be confused
            with the `overwrite` option, which determines whether allowing
            to overwrite the setup config dict itself.

        backup_setup_file : bool
            If True, a copy of the setup file is created. This applies to
            all three types of the config backends.
        """
        # we check if setup info config is present first and error out
        # early.
        if self.setup_info.config:
            if overwrite:
                self.logger.debug(
                    f"{self} has existing setup config, overwrite")
            else:
                raise RuntimeContextError(
                        f'{self} has existing setup config, to '
                        f'overwrite, use `overwrite=True`.'
                        )
        config_backend = self.config_backend
        # this is the config to be put in the setup info
        cfg = config_backend._make_config_for_setup(
                runtime_info_only=runtime_info_only)
        if isinstance(config_backend, DirConf) and setup_filepath is None:
            # this is the most normal case where we save the setup info
            # to the dirconf setup_file.
            # we need to disable the backup of setup file when requested
            # by backup_setup_file=False
            config_backend.update_setup_file(
                cfg, disable_backup=not backup_setup_file)
            # the above will trigger the reload of self.config.
            return self
        # this is the case for non-persistent setup
        # It will update the override_config dict with the setup info dict and
        # reload. Optionally, the new config is saved to the setup_filepath
        # if available
        if setup_filepath is not None:
            setup_filepath = ensure_abspath(setup_filepath)
            # check if we allow existing files and overwrite is allowed
            if not overwrite_setup_file and setup_filepath.exists():
                raise RuntimeContextError(
                        f'setup file path {setup_filepath} exists, to '
                        f'overwrite, use `overwrite_setup_file=True`.'
                        )
            elif setup_filepath.exists():
                self.logger.debug(
                    f"setup filepath {setup_filepath} exists, overwrite")
            else:
                # the specified setup file does not exist yet.
                pass
            # we can now create with backup handled for the setup file
            # using the DirConfPath machinery similar to DirConf.setup_file
            # this will create the backup file when backup_setup_file
            # is true. however, this case the content of the setup_filepath
            # is not checked.
            setup_filepath = DirConfPath(
                    label='setup_file',
                    path_name=setup_filepath.name,
                    path_type=DirConfPathType.FILE,
                    backup_enabled=True,
                    ).create(
                        setup_filepath.parent,
                        disable_backup=not backup_setup_file,
                        dry_run=False)
            # we can make a file header following that in the DirConf
            # to do this we need to make a copy since we'll need the
            # clean cfg for updating the setup info dict later
            # TODO this metadata will clutter the config dict but I guess it
            # worth it.
            cfg_copy = deepcopy(cfg)
            rupdate(cfg_copy, {
                DirConf.make_metadata_dict_key(setup_filepath):
                DirConf._make_setup_meta(
                    self,
                    desc=inspect.cleandoc("""
~~~~~~~~~~~~~~~~~
!!! IMPORTANT !!!
~~~~~~~~~~~~~~~~~

This file is created by `RuntimeContext.setup`. The content is generated
programatically, therefore should NOT be modified by hand.

This file contains a copy of the full config dict when the setup call we run,
and can be used as input to later runs for checking version compatibilities.
"""))})
            with open(setup_filepath, 'w') as fo:
                yaml_dump(cfg_copy, fo)
        # then update the setup info config via the override dict.
        config_backend.update_override_config(cfg)
        return self

    @classmethod
    def from_dir(cls, dirpath, init_config=None, **kwargs):
        """Create `RuntimeContext` instance from `dirpath`.

        This factory method is preferred when the given directory may not
        be already setup as tolteca workdir, in which case the setup can be
        done by passing ``create=True`` to this method.

        For paths that have already been setup as a tolteca workdir previously,
        use the constructor with ``config=dirpath`` is more convenient.

        Parameters
        ----------
        dirpath : `pathlib.Path`, str
            The path to the work directory.

        init_config : dict, optional
            The dict to add to the setup_file.

        **kwargs : dict
            Additional arguments passed to the underlying
            :meth:`DirConfMixin.populate_dir`.
        """
        try:
            dirconf = DirConf.populate_dir(dirpath, **kwargs)
        except Exception as e:
            raise RuntimeContextError(
                f"Failed create runtime context from dir {dirpath}: {e}")
        if kwargs.get('dry_run', False):
            # in dry run mode we'll just return None since there is nothing
            # created
            return None
        # write init_config to setup_file
        if init_config is None:
            init_config = dict()
        # here we should already did the backup in populate_dir if
        # requested, so we pass disable_backup here
        # we disable merge so the init config is not affected by previous
        # content in setup_file.
        DirConf(rootpath=dirconf).update_setup_file(
            init_config, disable_backup=True, merge=False)
        return cls(dirpath)


class RuntimeBaseError(RuntimeError):
    pass


class RuntimeBase(object):
    """A base class that consumes `RuntimeContext`.

    This class acts as a proxy of an underlying `RuntimeContext` object,
    providing a unified interface for subclasses to managed
    specialized config objects constructed from
    the config dict of the runtime context and its the runtime info.

    Parameters
    ----------
    config : `RuntimeContext`, `pathlib.Path`, str, dict
        The runtime context object, or the config source of it.
    """

    def __init__(self, config):
        if isinstance(config, RuntimeContext):
            rc = config
        else:
            rc = RuntimeContext(config)
        self._rc = rc

    @property
    def rc(self):
        return self._rc

    config_cls = NotImplemented
    """Subclasses implement this to provide specialized config object."""

    @cached_property
    def config(self):
        """The config object of :attr:`config_cls` constructed from the
        runtime context config dict.

        The config dict is validated and the constructed object is cached.
        The config object can be updated by using :meth:`RuntimeBase.update`.
        """
        return self.config_cls.from_config_dict(
            self.rc.config,
            # these are passed to the schema validate method
            rootpath=self.rc.rootpath,
            runtime_info=self.rc.runtime_info)

    def update(self, config, mode='override'):
        """Update the config object with provided config dict.

        Parameters
        ----------
        config : `dict`
            The config dict to apply.
        mode : {"override", "default"}
            Controls how `config` dict is applied, Wether to override the
            config or use as default for unspecified values.
        """
        if mode == 'override':
            self.rc.config_backend.update_override_config(config)
        elif mode == 'default':
            self.rc.config_backend.update_default_config(config)
        else:
            raise ValueError("invalid update mode.")
        # invalidate the cache
        if 'config' in self.__dict__:
            del self.__dict__['config']
