#!/usr/bin/env python

import functools
import socket
import json
from io import BytesIO
from urllib.parse import urlparse

from dasha.web import exit_stack

from tollan.utils.log import get_logger, timeit
from tollan.utils import to_typed


class Ocs3ConsumerMixin(object):
    
    def init_ocs3(self, url):
        self._ocs3_api = Ocs3API(url=url)
        
    @property
    def ocs3_api(self):
        return self._ocs3_api


class Ocs3API(object):
    def __init__(self, url):
        self._url = url
        
    def _create_socket_server(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        return server
    
    @functools.cached_property
    def _socket_server(self):
        return self._create_socket_server()
    
    def __exit__(self, *args, **kwargs):
        if "_socket_server" in self.__dict__:
            self._socket_server.__exit__(*args, **kwargs)

    @functools.lru_cache(maxsize=None)
    def get_or_create_connection(self):
        """Return """
        logger = get_logger()
        url = urlparse(self._url)
        addr = (url.hostname, url.port)
        addr_repr = f'{url.hostname}:{url.port}'
        with timeit(f"connect {addr_repr}"):
            try:
                self._socket_server.connect(addr)
            except Exception as e:
                logger.error(
                        f"unable to connect to {addr_repr}: {e}",
                        exc_info=True)
                # replace the socket server and retry
                if "_socket_server" in self.__dict__:
                    del self._socket_server
                    self._socket_server.connect(addr)
        return self._socket_server

    def query(self, query_str):
        logger = get_logger()
        conn = self.get_or_create_connection()
        if not query_str.endswith("\n"):
            query_str += '\n'
        with timeit('send query'):
            logger.debug(f"query ocs3: {query_str}")

            def send():
               conn.sendall(query_str.encode())

            try:
                send()
            except Exception as e:
                logger.debug(f"failed sending query: {e}", exc_info=True)
                # re-connect and retry
                del self.__dict__['_socket_server']
                conn = self.get_or_create_connection()
                send()

        with timeit('get response'):
            buf = BytesIO()
            while True:
                data = conn.recv(1)
                if not data or data == b'\n':
                    break
                buf.write(data)
            buf = buf.getvalue()
            logger.debug(f"size of response: {len(buf)}")
            response = buf.decode().strip(';\r\n')
            with timeit('parse objects'):
                try:
                    d = json.loads(response)
                except Exception:
                    logger.debug(f"invalid json:\n{response}")
                    return None
                # walk done the tree to convert values to sensible type
                d = to_typed_json(d)
            return d


def to_typed_json(node):
    if isinstance(node, str):
        return to_typed(node)
    if isinstance(node, list):
        return list(map(to_typed_json, node))
    if isinstance(node, dict):
        return {k: to_typed_json(v) for k, v in node.items()}
    return node


