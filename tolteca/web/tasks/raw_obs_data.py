#!/usr/bin/env python

"""
This module defines tasks to handle raw obs data.
"""


from .dbrt import dbrt

from dasha.web.extensions.celery import celery_app, schedule_task, Q
from dasha.web.extensions.ipc import ipc
from dasha.web import exit_stack
from .. import lmt_ocs3_url
from urllib.parse import urlparse
import socket
import io
from tollan.utils.log import get_logger, timeit
from ...utils import get_pkg_data_path
import functools
import yaml
import re
from tollan.utils import odict_from_list
from wrapt import ObjectProxy
import shlex
from .. import basic_obs_datastore


def update_raw_obs_from_toltecdb():
    logger = get_logger()

    # these are the sqla table objects
    # we need to check both the original table
    # and the repeated file table because the
    # repeater may not be working.
    # entries from both table are added to
    # dp_raw_obs, and a named group is created
    # to state this relation
    t_orig = dbrt['toltec'].tables['toltec']
    t_local = dbrt['toltec'].tables['toltec_r1']
    t_raw_obs = dbrt['tolteca'].tables['dp_raw_obs']
    t_dp_named_group = dbrt['tolteca'].tables['dp_named_group']
    t_dpa_named_group_data_prod = dbrt['tolteca'].tables[
            'dpa_named_group_data_prod']

    # The file info is organized as a pandas table
    # which contains the basic obs identifiers
    # and the local file paths from the local table
    # this table is 
    toltec_file_info_store.set()


if celery_app is not None:

    QueueOnce = celery_app.QueueOnce

    # make the task
    celery_app.task(base=QueueOnce, once={'timeout': 10})(update_raw_obs_from_toltecdb)

    # schedule the task 1 sec interval
    schedule_task(update_toltec_file_info_from_db, schedule=1, args=tuple(), options={'queue': Q.high_priority})
