#! /usr/bin/env python

"""Access files via SSH."""

import paramiko
from tollan.utils.log import logit, get_logger
from pathlib import Path
import stat


class SSHAccessor(object):

    logger = get_logger()
    _config = paramiko.SSHConfig()

    def __init__(self, hostname, **kwargs):
        self._hostname = hostname

        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.WarningPolicy())
        with logit(self.logger.debug, f"connect to {self.hostname}"):
            if 'proxycommand' in kwargs:
                proxycommand = kwargs.pop('proxycommand')
                proxycommand = proxycommand.replace("%h", hostname)
                proxycommand = proxycommand.replace(
                        "%p", str(kwargs.get('port', 22)))
                kwargs['sock'] = paramiko.ProxyCommand(proxycommand)
            if 'key_filename' in kwargs:
                kwargs['key_filename'] = Path(
                        kwargs['key_filename']).expanduser().as_posix()
            self.logger.debug(f"use proxycommand {proxycommand}")
            client.connect(
                    hostname=self.hostname,
                    **kwargs
                    )
            self._client = client
            self._sftp = client.open_sftp()

    @property
    def hostname(self):
        return self._hostname

    def __repr__(self):
        return f'{self.__class__.__name__}({self.hostname})'

    def glob(self, dir_, pattern):
        return self._sftp.listdir(dir_.as_posix())

    def walk(
            self, rootpath, recursive=True,
            skip=None, inc=None, return_attr=False):
        for item in self._sftp.listdir_attr(rootpath.as_posix()):
            path = rootpath.joinpath(item.filename)
            if inc is not None:
                if not inc(path):
                    continue
            if skip is not None:
                if skip(path):
                    continue
            if recursive and stat.S_ISDIR(item.st_mode):
                for result in self.walk(
                        path, recursive=True,
                        inc=inc, skip=skip, return_attr=return_attr):
                    yield result
            if return_attr:
                yield path, item
            else:
                yield path
