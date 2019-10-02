#! /usr/bin/env python
from contextlib import contextmanager
import sys
import time
from astropy import log

try:
    from PyQt5 import QtWidgets, QtCore
except ModuleNotFoundError:

    log.error("PyQt5 is required to run the GUI")
    sys.exit(1)


class QThreadTarget(QtCore.QObject):

    finished = QtCore.pyqtSignal()

    should_keep_running = True

    def __init__(self, interval, target=lambda: None):
        super().__init__()
        self._interval = interval
        self._target = target

    @QtCore.pyqtSlot()
    def start(self):
        while self.should_keep_running:
            time.sleep(self._interval / 1e3)
            self._target()
        self.finished.emit()

    @QtCore.pyqtSlot()
    def stop(self):
        self.should_keep_running = False


def qt5app(args=None):
    if args and args[0] != sys.argv[0]:
        args = sys.argv[:1] + args
    app = QtWidgets.QApplication(args)
    # Let the interpreter run each 500 ms, to process the SIGINT signal
    # http://stackoverflow.com/questions/4938723
    timer = QtCore.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)
    # need to keep this pointer during the app run
    app._keep_python_alive = timer
    return app


@contextmanager
def slot_disconnected(signal, slot):
    try:
        signal.disconnect(slot)
    except Exception:
        pass
    yield
    signal.connect(slot)
