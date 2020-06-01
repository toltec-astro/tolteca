#! /usr/bin/env python

import celery
from celery_once import QueueOnce
from dasha.web.extensions.celery import schedule_task
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


def get_ocs3_info_store():
    return ipc.get_or_create('rejson', label='ocs_info')


def get_ocs3_server():
    return socket.socket(socket.AF_INET, socket.SOCK_STREAM)


ocs3_info_store = get_ocs3_info_store()
ocs3_server = ObjectProxy(get_ocs3_server())
exit_stack.push(ocs3_server.__exit__)  # ensure the socket is released


def connect():
    logger = get_logger()
    url = urlparse(lmt_ocs3_url)
    addr = (url.hostname, url.port)
    addr_repr = f'{url.hostname}:{url.port}'
    with timeit(f"connect {addr_repr}"):
        try:
            ocs3_server.connect(addr)
        except Exception as e:
            logger.error(
                    f"unable to connect to {addr_repr}: {e}",
                    exc_info=True)
            # replace the socket
            ocs3_server.close()
            ocs3_server.__wrapped__ = get_ocs3_server()
    return ocs3_server


@functools.lru_cache(maxsize=1)
def get_ocs3_api():
    logger = get_logger()
    filepath = get_pkg_data_path().joinpath('ocs3_api.yaml')
    logger.info(f"load ocs3 api from {filepath}")
    with open(filepath, 'r') as fo:
        api = yaml.safe_load(fo)
    for obj in api:
        obj['attrs'] = odict_from_list(obj['attrs'], key='name')
    return odict_from_list(api, key='name')


def filter_obj(obj):
    return True
    # return obj['name'] != 'ToltecDetectors'


def parse_value(obj_name, attr_name, value_str):
    type_ = get_ocs3_api()[obj_name]['attrs'][attr_name]['type']
    if type_ == 'string':
        return value_str
        # assert value_str[0] == '"'
        # assert value_str[-1] == '"'
        # return value_str[1:-1]
    elif type_ == 'int':
        try:
            return int(value_str)
        except ValueError:
            # this is probably due to enum type int.
            return value_str
    elif type_ == 'double':
        return float(value_str)
    elif type_ == 'boolean':
        return value_str == '1'
    raise ValueError("unable to parse value")


def parse_attrs(obj_name, kvs):
    attrs = dict()
    re_name = re.compile(
            r'^-(?P<name>[^\[]+)(?:\[(?P<idx>\d+)\](?:\[(?P<idx2>\d+)\])?)?$')
    for k, v in kvs:
        g = re.match(re_name, k).groupdict()
        attr_name = g['name']
        attr_idx = g.get('idx', None)
        attr_idx2 = g.get('idx2', None)
        # print(f"{k} {obj_name}.{attr_name} -> {v} {g}")
        value = parse_value(obj_name, attr_name, v)
        if attr_idx is not None:
            idx = int(attr_idx)
            if attr_name not in attrs:
                attrs[attr_name] = list()
            else:
                if attr_idx2 is None:
                    assert(idx == len(attrs[attr_name]))
            if attr_idx2 is not None:
                idx2 = int(attr_idx2)
                # make a list of list
                if idx2 == 0:
                    attrs[attr_name].append(list())
                else:
                    assert(idx2 == len(attrs[attr_name][-1]))
                attrs[attr_name][-1].append(value)
            else:
                attrs[attr_name].append(value)
        else:
            attrs[attr_name] = value
    return attrs


def parse_obj(obj_str):
    # parts = re.split(r'\s+', obj_str)
    parts = shlex.split(obj_str)
    obj_name = parts[0]
    obj = dict(
            name=obj_name,
            attrs=parse_attrs(obj_name, zip(*[iter(parts[1:])] * 2))
            )
    return obj


@celery.shared_task(base=QueueOnce, once={'timeout': 10})
def update_ocs_info():
    logger = get_logger()
    api = get_ocs3_api()
    # build the query
    query = []
    for obj in filter(filter_obj, api.values()):
        obj_query = [obj['name'], ]
        for attr in obj['attrs'].values():
            obj_query.append(f'-{attr["name"]}')
        query.append(' '.join(obj_query))
    query = ';'.join(query) + '\n'
    with timeit('send query'):
        # logger.debug(f"query ocs3: {query}")

        def send():
            ocs3_server.sendall(query.encode())

        try:
            send()
        except Exception as e:
            logger.debug(f"failed sending query: {e}", exc_info=True)
            connect()
            send()
    with timeit('get response'):
        buf = io.BytesIO()
        while True:
            data = ocs3_server.recv(1)
            if not data or data == b'\n':
                break
            buf.write(data)
        buf = buf.getvalue()
        logger.debug(f"size of response: {len(buf)}")
        response = buf.decode().strip(';\r\n')
    with timeit('parse objects'):
        objs = [parse_obj(obj) for obj in response.split(';')]
        logger.debug(f"parsed {len(objs)} objects")
    # for obj in objs:
        # obj['attrs'] = odict_from_list(obj['attrs'], key='name')
    ocs3_info_store.set(odict_from_list(objs, key='name'))


# 1 sec interval
schedule_task(update_ocs_info, schedule=1, args=tuple())
