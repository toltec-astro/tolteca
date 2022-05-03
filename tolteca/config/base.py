#!/usr/bin/env python


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
        if include_config_impl_cache and \
                '_config_impl' in self.__dict__:
            del self.__dict__['_config_impl']
            logger.debug("config impl cache invalidated")

        if 'config' in self.__dict__:
            del self.__dict__['config']
            logger.debug("config cache invalidated")

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
        # we update the config cache without re-load the config from file
        self.__dict__['config'] = self.load(
            reload_config=False, update_runtime_info=True)

    def set_override_config(self, cfg):
        """Set the override config dict.

        This will invalidate the config cache.
        """

        self._override_config = cfg
        # we update the config cache without re-load the config from file
        self.__dict__['config'] = self.load(
            reload_config=False, update_runtime_info=True)

    def update_default_config(self, cfg):
        """Update the default config dict.

        This will invalidate the config cache.
        """
        rupdate(self._default_config, cfg)
        # we update the config cache without re-load the config from file
        self.__dict__['config'] = self.load(
            reload_config=False, update_runtime_info=True)

    def update_override_config(self, cfg):
        """Update the override config dict.

        This will invalidate the config cache.
        """
        rupdate(self._override_config, cfg)
        # we update the config cache without re-load the config from file
        self.__dict__['config'] = self.load(
            reload_config=False, update_runtime_info=True)

    @property
    def is_persistent(self):
        return NotImplemented

    @property
    def rootpath(self):
        return NotImplemented



