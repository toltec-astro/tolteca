#!/usr/bin/env python

import argparse

from tollan.utils.log import get_logger
from tollan.utils.fmt import pformat_yaml
from tollan.utils import ensure_abspath, getname
from ..utils import dict_from_cli_args


__all__ = ['load_runtime', ]


def load_runtime(
        runtime_cls, config_loader,
        no_cwd=False, runtime_context_dir_only=False,
        runtime_cli_args=None):
    """Return instance of `runtime_cls`."""

    logger = get_logger()

    workdir = config_loader.runtime_context_dir

    if workdir is None and not no_cwd:
        # in this special case we just use the current directory
        workdir = ensure_abspath('.')

    try:
        rt = runtime_cls(workdir)
    except Exception as e:
        if (
                config_loader.runtime_context_dir is not None
                or runtime_context_dir_only):
            # raise when user explicitly specified the workdir
            # or requested to load only from rc dir.
            raise argparse.ArgumentTypeError(
                f"invalid workdir {workdir}\n"
                f"{getname(e.__class__)}: {e}")
        else:
            logger.debug("no valid runtime context in current directory")
            # create rc from config
            cfg = config_loader.get_config()
            rt = runtime_cls(cfg)
    logger.debug(f"loaded {runtime_cls} instance: {rt}")
    # update with cli args
    if runtime_cli_args is not None:
        _cli_cfg = dict_from_cli_args(runtime_cli_args)
        # note the cli_cfg is under the namespace redu
        cli_cfg = {rt.config_cls.config_key: _cli_cfg}
        if _cli_cfg:
            logger.info(
                f"config specified with commandline arguments:\n"
                f"{pformat_yaml(cli_cfg)}")
        rt.update(cli_cfg, mode='override')
        cfg = rt.config.to_config_dict()
        # here we recursively check the cli_cfg and report
        # if any of the key is ignored by the schema and
        # throw an error

        def _check_ignored(key_prefix, d, c):
            if isinstance(d, dict) and isinstance(c, dict):
                ignored = set(d.keys()) - set(c.keys())
                ignored = [f'{key_prefix}.{k}' for k in ignored]
                if len(ignored) > 0:
                    raise argparse.ArgumentError(
                        f"Invalid config items specified in "
                        f"the commandline: {ignored}")
                for k in set(d.keys()).intersection(c.keys()):
                    _check_ignored(f'{key_prefix}{k}', d[k], c[k])
        _check_ignored('', cli_cfg, cfg)

    return rt
