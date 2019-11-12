#! /usr/bin/env python

from contextlib import ContextDecorator
import logging
import logging.config
import inspect
import functools
import time

from . import deepmerge


def init_logging(overrides):
    """Initialize logging facilities
    """
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s:'
                          ' %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'short': {
                'format': '[%(levelname)s] %(name)s: %(message)s'
            },
        },
        'handlers': {
            'default': {
                'level': 'DEBUG',
                'class': 'logging.StreamHandler',
                'formatter': 'short',
            },
        },
        'loggers': {
            '': {
                'handlers': ['default'],
                'level': 'DEBUG',
                'propagate': False
            },
            'matplotlib': {
                'handlers': ['default'],
                'level': 'WARNING',
                'propagate': False
            },
            'root': {
                'handlers': ['default'],
                'level': 'ERROR',
                'propagate': False
            },
        }
    }
    deepmerge(config, overrides)
    logging.config.dictConfig(config)


def get_logger(name=None):
    if name is None:
        name = inspect.stack()[1][3]
        # code = inspect.currentframe().f_back.f_code
        # func = [obj for obj in gc.get_referrers(code)][0]
        # name = func.__qualname__
    return logging.getLogger(name)


def timeit(func):
    funcname = func.__name__

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger("timeit")
        logger.debug("{} ...".format(funcname))
        s = time.time()
        r = func(*args, **kwargs)
        elapsed = time.time() - s
        logger.debug("{} done in {}".format(
            funcname,
            "{:.0f}ms".format(elapsed * 1e3)))
        return r
    return wrapper


class logit(ContextDecorator):
    def __init__(self, func, msg):
        self.func = func
        self.msg = msg

    def __enter__(self):
        self.func(f"{self.msg} ...")

    def __exit__(self, *args):
        self.func(f'{self.msg} done')
