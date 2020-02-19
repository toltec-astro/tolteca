#! /usr/bin/env python

"""This package contains a set of recipes that make use of `tolteca`."""


from pathlib import Path
import appdirs
from tollan.utils.log import init_log, get_logger, logit


init_log(level='DEBUG')


def get_user_data_dir():
    return Path(appdirs.user_data_dir("tolteca-recipes", "toltec"))


def get_extern_dir():
    logger = get_logger()
    p = get_user_data_dir().joinpath('extern')
    if not p.exists():
        with logit(logger.debug, f"create {p}"):
            p.mkdir(exist_ok=True, parents=True)
    return p
