#! /usr/bin/env python

"""
Inter-op with TolTEC roach manager.
"""

from astropy import log


class TcpListenerMixin(object):
    """Mixin class for services that handle requests issued via TCP socket."""
    def __init__(self):
        pass

    def serve(self, url):
        pass

class EventHandlerMixin(object):
    """Mixin class that handles callbacks"""
    def __init__(self):
        pass

    def serve(self, url):
        pass



class ToltecRoachManagerCommandListener(TcpListenerMixin, EventHandlerMixin):
    """A service to handle TolTEC roach manager commands."""

    def __init__(self, callbacks=None):
        self._callbacks = dict()
        if callbacks is not None:
            for k, v in callbacks.items:
                self.register_callback(k, v)

    def register_callback(self, event, callback):
        if hasattr(self.__class__, event):
            self._callbacks[event].append

    def dispatch(self, cmd):
        if "-r" in cmd:
            self.reduce(cmd)

    @staticmethod
    def reduce(input_):
        log.debug(f"reduce {input_}")

    @staticmethod
    def plot(input_):
        log.debug(f"reduce {input_}")
