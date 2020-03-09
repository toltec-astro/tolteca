#! /usr/bin/env python
"""The GUI  entry point."""

import yaml
from ..utils import get_pkg_data_path
import sys
from pathlib import Path
import psutil
from collections import OrderedDict
from datetime import datetime
from tollan.utils import rupdate
from tollan.utils.fmt import pformat_yaml
from tollan.utils.qt import qt5app, IntervalTarget
from tollan.utils.qt.colors import Palette
from tollan.utils.log import get_logger, init_log, logit, timeit
import argparse
from ..version import version
import concurrent

from ..db import get_databases
from ..fs.toltec import ToltecDataFileStore

from PyQt5 import uic, QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt
import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import (
        Parameter, ParameterTree, ParameterItem, registerParameterType)


UI_FILE_DIR = Path(__file__).parent.joinpath("ui")

Ui_MainWindow, Ui_MainWindowBase = uic.loadUiType(
        UI_FILE_DIR.joinpath("tolteca.ui"))
Ui_DBStatus, Ui_DBStatusBase = uic.loadUiType(
        UI_FILE_DIR.joinpath("dbstatus.ui"))
Ui_FileView, Ui_FileViewBase = uic.loadUiType(
        UI_FILE_DIR.joinpath("fileview.ui"))

palette = Palette()


class DBStatusPanel(Ui_DBStatusBase):

    connectionStatusChanged = QtCore.pyqtSignal(bool)

    logger = get_logger()

    def __init__(self, database, parent=None):
        super().__init__(parent)
        self.ui = Ui_DBStatus()
        self.ui.setupUi(self)

        self._database = database
        self._connected = False

    @property
    def connected(self):
        is_alive = self._database.is_alive
        if self._connected != is_alive:
            self._connected = is_alive
            self.connectionStatusChanged.emit(self._connected)
        # self.logger.debug(
            # f"database connection: {'OK' if self._connected else 'ERROR'}")
        self.ui.led_status.set_status(self._connected)
        return self._connected


class AnimatedItemDelegate(QtWidgets.QStyledItemDelegate):

    def __init__(self, role=Qt.UserRole, parent=None):
        super().__init__(parent)
        self._role = role

    def paint(self, painter, option, index):

        model = index.model()
        args = model.data(index, role=self._role)

        if args is not None:
            (t0, dt, c0, c1) = args
            frac = (datetime.now() - t0).seconds / dt
            frac = 0 if frac < 0 else frac
            frac = 1 if frac > 1 else frac
            c = palette.blend(c0, c1, frac, fmt='irgb')
            # print(f"paint: {t0} {frac} {c}")
            painter.fillRect(option.rect, QtGui.QColor(*c))
        super().paint(painter, option, index)


def _get_datafile_info(spec, path):
    info = spec.info_from_filename(path) or {}
    info['uid'] = f'{info["obsid"]}-{info["subobsid"]}-{info["scanid"]}'
    if 'source' not in info:
        info['source'] = path
    return info


class DataFileInfoModel(QtCore.QAbstractTableModel):

    logger = get_logger()

    item_update_role = Qt.UserRole
    item_update_key = '_time_last_updated'
    item_update_time = 1.

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self._info_keys = set()
        self._info = dict()

        def update_row(tl=None, br=None):
            i0 = 0 if tl is None else tl.row()
            i1 = self.rowCount() if br is None else br.row() + 1
            self.logger.debug(f"model row {i0} {i1} changed")
        self.dataChanged.connect(update_row)
        self.modelReset.connect(update_row)

    def set(self, runtime_info, changed_items):
        if len(changed_items) == len(runtime_info):
            self.reset(runtime_info)
        else:
            for i, (k, v) in enumerate(self._info):
                if k in changed_items:
                    self._info[i] = (k, runtime_info[k])
                    self._info[i][1][self.item_update_key] = datetime.now()
                    self.dataChanged.emit(
                        self.index(i, 0),
                        self.index(i, self.columnCount() - 1))

    def reset(self, runtime_info):
        self.beginResetModel()
        self._info = list(runtime_info.items())
        for i in range(len(self._info)):
            self._info[i][1][self.item_update_key] = datetime.now()
        self.endResetModel()

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self._info)

    # (header, {role: callback})
    _dispatch_columns = {
            0: ("Link", {
                Qt.DisplayRole: lambda k, v: k,
                }),
            1: ("Obs #", {
                Qt.DisplayRole: lambda k, v: v['uid'],
                }),
            2: ("Kind", {
                Qt.DisplayRole: lambda k, v: v['kindstr']
                }),
            3: ("Time", {
                Qt.DisplayRole: lambda k, v: str(v['ut'])
                }),
            4: ("File", {
                Qt.DisplayRole: lambda k, v: str(v['source'])
                }),
            }
    _dispatch_row = {
            # QtCore.Qt.BackgroundRole:
            # lambda k, v: QtGui.QBrush(Qt.yellow)
            item_update_role: lambda k, v: (
                v[DataFileInfoModel.item_update_key],
                DataFileInfoModel.item_update_time,
                "yellow", "white")
            }

    def columnCount(self, parent=QtCore.QModelIndex()):
        return max(
                i for i in self._dispatch_columns.keys()) + 1

    def headerData(self, col, orientation, role):
        if orientation == QtCore.Qt.Horizontal and \
                role == QtCore.Qt.DisplayRole:
            if col in self._dispatch_columns:
                return self._dispatch_columns[col][0]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            i = index.row()
            j = index.column()
            k, v = self._info[i]
            if role in self._dispatch_row:
                return self._dispatch_row[role](k, v)
            if j in self._dispatch_columns:
                dispatch = self._dispatch_columns[j][1]
                if role in dispatch:
                    return dispatch[role](k, v)


class DataFileGroupModelFilter(QtCore.QSortFilterProxyModel):

    def __init__(self, key_column, parent=None):
        super().__init__(parent=parent)
        self._key_column = key_column

    def filterAcceptsRow(self, source_row, source_parent):

        source = self.sourceModel()
        source_index = source.index(
                source_row, self._key_column, source_parent)
        p = Path(source.filePath(source_index))
        if p.is_dir():
            # has_data = False
            # for i in range(source.rowCount(source_index)):
            #     print(i)
            #     idx = source.index(i, self._key_column, source_index)
            #     if source.data(idx) is not None:
            #         has_data = True
            #         break
            # return has_data
            # return super().filterAcceptsRow(source_row, source_parent)
            return True
        else:
            return source.data(source_index) is not None


class DataFileGroupModel(QtWidgets.QFileSystemModel):

    def __init__(self, spec, parent=None):
        super().__init__(parent=parent)
        self._spec = spec
        self.sortfilter_model = DataFileGroupModelFilter(1)  # Interface
        self.sortfilter_model.setSourceModel(self)

    _dispatch_columns = {
            # 0: ("Dir", {
            #     Qt.DisplayRole: lambda v: v
            #     }),
            0: ("Obs #", {
                Qt.DisplayRole: lambda v: v.get('uid', None)
                }),
            1: ("Interface", {
                Qt.DisplayRole: lambda v: v.get('interface', None)
                }),
            2: ("Kind", {
                Qt.DisplayRole: lambda v: v.get('kindstr', None)
                }),
            3: ("Time", {
                Qt.DisplayRole: lambda v: str(v.get('ut', None))
                }),
            4: ("File", {
                Qt.DisplayRole: lambda v: str(v['source'])
                }),
            }

    def columnCount(self, parent=QtCore.QModelIndex()):
        return max(
                i for i in self._dispatch_columns.keys()) + 1

    def headerData(self, col, orientation, role):
        if orientation == QtCore.Qt.Horizontal and \
                role == QtCore.Qt.DisplayRole:
            if col in self._dispatch_columns:
                return self._dispatch_columns[col][0]

    def data(self, index, role=Qt.DisplayRole):
        path = Path(self.filePath(index))
        if path.is_dir():
            if index.column() == 0:
                return super().data(index, role)
            return None
        info = self._get_info(path)
        if index.isValid():
            j = index.column()
            if j in self._dispatch_columns:
                dispatch = self._dispatch_columns[j][1]
                if role in dispatch:
                    return dispatch[role](info)
        # return super().data(index, role)

    def _get_info(self, path):
        return _get_datafile_info(self._spec, path)

    def sortfilter_model_index(self, p):
        return self.sortfilter_model.mapFromSource(self.index(p))


class DataFilesView(Ui_FileViewBase):

    rootpathChanged = QtCore.pyqtSignal(str)
    runtimeInfoChanged = QtCore.pyqtSignal(dict, list)

    logger = get_logger()

    def __init__(self, datafiles, parent=None):
        super().__init__(parent)
        self.ui = Ui_FileView()
        self.ui.setupUi(self)

        self._datafiles = datafiles
        self._rootpath_init = self.rootpath

        self.datafilegroup_model = DataFileGroupModel(
                spec=datafiles.spec)

        self.ui.tv_files.setModel(self.datafilegroup_model.sortfilter_model)
        self.ui.le_rootpath.textChanged.connect(self._validate_rootpath)

        def update_fileview():
            p = self.rootpath.as_posix()
            self.datafilegroup_model.setRootPath(p)
            self.ui.tv_files.setRootIndex(
                    self.datafilegroup_model.sortfilter_model_index(p))
            self.ui.le_rootpath.setText(p)

        def reset_rootpath():
            self.rootpath = self._rootpath_init

        self.rootpathChanged.connect(update_fileview)
        self.rootpathChanged.connect(lambda x: self.runtime_info)
        self.ui.btn_reset_rootpath.clicked.connect(reset_rootpath)

        self.runtime_info_model = DataFileInfoModel()
        self.ui.tv_runtime_info.setSelectionBehavior(
                QtWidgets.QTableView.SelectRows)
        self.ui.tv_runtime_info.setModel(self.runtime_info_model)
        self.ui.tv_runtime_info.setItemDelegate(AnimatedItemDelegate(
            role=self.runtime_info_model.item_update_role))
        self.ui.tv_runtime_info.horizontalHeader().setSectionResizeMode(
                QtWidgets.QHeaderView.ResizeToContents)
        self.runtimeInfoChanged.connect(self.runtime_info_model.set)
        self.runtime_update_repaint_timer = QtCore.QTimer(self)
        self.runtime_update_repaint_timer.setInterval(100.)
        self.runtime_update_repaint_timer.timeout.connect(
                self.ui.tv_runtime_info.viewport().repaint)

        def update_repaint():
            self.runtime_update_repaint_timer.start()

            def on_stop():
                self.runtime_update_repaint_timer.stop()
                self.ui.tv_runtime_info.viewport().repaint()
                QtWidgets.QApplication.processEvents()
            QtCore.QTimer.singleShot(
                    self.runtime_info_model.item_update_time * 1.5e3,
                    on_stop)
        self.runtimeInfoChanged.connect(update_repaint)

        self.rootpathChanged.emit(self.rootpath.as_posix())

    @property
    def rootpath(self):
        return self._datafiles.rootpath

    @rootpath.setter
    def rootpath(self, path):
        old = self._datafiles.rootpath
        self._datafiles.rootpath = path
        new = self._datafiles.rootpath
        if old != new:
            self.logger.debug(f"change root path: {old} -> {new}")
            self.rootpathChanged.emit(str(new))

    def _validate_rootpath(self, path):
        p = Path(path)
        if p.is_dir():
            self.rootpath = p
            color = palette.hex('black')
        else:
            color = palette.hex('red')
        sender = self.sender()
        sender.setStyleSheet(f'color: {color};')

    @staticmethod
    def _runtime_info_changed(i1, i2):
        logger = get_logger()
        i1keys = set(i1.keys()) if i1 is not None else set()
        i2keys = set(i2.keys()) if i2 is not None else set()
        if i1keys != i2keys:
            logger.debug(f"runtime info changed: {i1keys} -> {i2keys}")
            return True, list(i2keys)
        changed = []
        for k in i1keys:
            v1 = i1[k]
            v2 = i2[k]
            for vk in ('nwid', 'obsid', 'subobsid', 'scanid', 'ut'):
                if v1[vk] != v2[vk]:
                    logger.debug(
                            f"runtime info {k}.{vk} changed: "
                            f"{v1[vk]} -> {v2[vk]}")
                    changed.append(k)
                    break
        return len(changed) > 0, changed

    @property
    def runtime_info(self):
        old = getattr(self, '_runtime_info', None)
        self._update_runtime_info()
        new = self._runtime_info
        changed, changed_items = self._runtime_info_changed(old, new)
        if changed:
            self.runtimeInfoChanged.emit(new, changed_items)
        return new

    def _update_runtime_info(self):
        _info = {}

        def link_key(info):
            return f"{info['interface']}-{info['master']}"

        for link in self._datafiles.runtime_datafile_links():
            info = _get_datafile_info(self._datafiles.spec, link)
            info['link'] = link
            _info[link_key(info)] = info

        self._runtime_info = OrderedDict(
                sorted(_info.items(), key=lambda i: i[1]['nwid']))


class GuiRuntime(QtCore.QObject):
    """This class manages the runtime resources."""

    logger = get_logger()

    def __init__(self, config, parent=None):
        super().__init__(parent=parent)
        self.config = config
        self.logger.debug(f"runtime config: {pformat_yaml(self.config)}")

    def init_db_monitors(self, gui, parent):
        if 'db' not in self.config:
            return parent.hide()
        db_config = self.config['db']
        self.databases = timeit(get_databases)(db_config)
        self.logger.debug(f"databases: {self.databases}")
        self._db_monitors = {}
        for k, v in self.databases.items():
            w = DBStatusPanel(v, parent=parent)
            parent.layout().addRow(
                    QtWidgets.QLabel(v._config.get('label', k)), w)
            self._db_monitors[k] = w
            gui.run_with_interval(2000, lambda w=w: w.connected)

    def _get_datafile_rootpath(self):
        return self.config['fs']['local']['rootpath']

    def init_df_view(self, gui, parent):
        if 'fs' not in self.config:
            return parent.hide()
        self.datafiles = ToltecDataFileStore(
                rootpath=self._get_datafile_rootpath())
        w = DataFilesView(self.datafiles, parent=parent)
        parent.layout().addWidget(w)

        gui.run_with_interval(500, lambda: w.runtime_info)

    def init_sp_view(self, gui, parent):

        class MutualExclusiveGroup(pTypes.GroupParameter):
            def __init__(self, **opts):
                pTypes.GroupParameter.__init__(self, **opts)

                def set_enabled(enabled, value):
                    if not value:
                        return
                    for c in self.children():
                        e = c.param('enabled')
                        e.setValue(
                                enabled is e,
                                blockSignal=set_enabled)
                for c in self.children():
                    c.param('enabled').sigValueChanged.connect(set_enabled)

        params = [
            {
                'name': 'Interface',
                'type': 'group',
                'children': [{
                    'name': 'Interface',
                    'type': 'list',
                    'values': {"one": 1, "two": "twosies"}, 'value': 'one'},
                    ],
                },
            MutualExclusiveGroup(
                name='Detector selection',
                children=[
                    {
                        'name': 'fancy indexing',
                        'type': 'group',
                        'children': [
                            {
                                'name': 'enabled',
                                'type': 'bool',
                                'value': True,
                                },
                            {
                                'name': 'value',
                                'type': 'str',
                                'value': '-1',
                                },
                            ],
                        },
                    {
                        'name': 'query',
                        'type': 'group',
                        'children': [
                            {
                                'name': 'enabled',
                                'type': 'bool',
                                'value': False,
                                },
                            {
                                'name': 'value',
                                'type': 'str',
                                'value': 'Qr>15000',
                                },
                            ],
                        },
                    ]
                ),
            MutualExclusiveGroup(
                name='File selection',
                children=[
                    {
                        'name': 'fancy indexing',
                        'type': 'group',
                        'children': [
                            {
                                'name': 'enabled',
                                'type': 'bool',
                                'value': True,
                                },
                            {
                                'name': '',
                                'type': 'str',
                                'value': '-1',
                                },
                            ],
                        },
                    {
                        'name': 'query',
                        'type': 'group',
                        'children': [
                            {
                                'name': 'enabled',
                                'type': 'bool',
                                'value': False,
                                },
                            {
                                'name': '',
                                'type': 'str',
                                'value': 'obsid>9000',
                                },
                            ],
                        },
                    ]
                ),
            ]
        params = Parameter.create(
                name='Scope Params', type='group', children=params)

        logger = get_logger()

        def on_params_change(params, changes):
            msg = ["tree changes:"]
            for param, change, data in changes:
                path = params.childPath(param)
                if path is not None:
                    childName = '.'.join(path)
                else:
                    childName = param.name()
                msg.append(f'- parameter: {childName}')
                msg.append(f'     change: {change}')
                msg.append(f'       data: {data}')
            logger.debug('\n'.join(msg))
        params.sigTreeStateChanged.connect(on_params_change)

        def on_value_changing(param, value):
            logger.debug(f"value changing {param}, {value}")

        def _connect_tree(p, sig, cb):
            getattr(p, sig).connect(cb)
            for c in p.children():
                _connect_tree(c, sig, cb)

        _connect_tree(params, 'sigValueChanging', on_value_changing)

        w = ParameterTree(showHeader=False)
        w.setParameters(params, showTop=True)
        parent.layout().addWidget(w)


class Gui(Ui_MainWindowBase):

    _title = f"TolTECA v{version}"
    _threads = None
    logger = get_logger()

    def __init__(self, parent=None):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle(self._title)
        self.ui.actionAbout.triggered.connect(self.about)

        self._psutil_process = psutil.Process()
        self.run_with_interval(1000, self._report_usage)

    def init_runtime(self, config):
        self.runtime = GuiRuntime(config, parent=self)
        self.runtime.init_db_monitors(self, self.ui.gb_dbstatus)
        self.runtime.init_df_view(self, self.ui.gb_datafiles)
        self.runtime.init_sp_view(self, self.ui.gb_scopeparams)

    def run_in_thread(self, target, start=True):
        thread = QtCore.QThread(self)
        target.moveToThread(thread)
        target.finished.connect(thread.quit)
        thread.started.connect(target.start)
        if start:
            thread.start()

        if self._threads is None:
            self._threads = []
        self._threads.append((thread, target))

    def run_with_interval(self, *args, start=True, **kwargs):
        target = IntervalTarget(*args, **kwargs)
        return self.run_in_thread(target, start=start)

    def about(self):
        return QtWidgets.QMessageBox.about(self, "About", "Surprise!")

    def _report_usage(self):
        p = self._psutil_process
        # log.debug("report usage")
        with p.oneshot():
            message = \
                    '    pid: {0}    cpu: {1} %    memory: {2:.1f} MB' \
                    '    page: {3:.1f} MB    num_threads: {4}'.format(
                        p.pid, p.cpu_percent(),
                        p.memory_info().rss / 2. ** 20,
                        p.memory_info().vms / 2. ** 20,
                        p.num_threads()
                        )
            self.ui.statusbar.showMessage(message)

    @staticmethod
    def _close_thread(thread):
        logger = get_logger()
        with logit(logger.debug, f'close thread {thread}'):
            thread.requestInterruption()
            thread.quit()
            thread.wait()

    def closeEvent(self, event):
        with logit(self.logger.debug, 'close window'):
            # close all threads
            with concurrent.futures.ThreadPoolExecutor(len(self._threads)) as \
                    executor:
                fs = [
                        executor.submit(self._close_thread, thread)
                        for thread, _ in self._threads]
                concurrent.futures.wait(fs, timeout=0.1)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
            "-c", "--config",
            nargs='+',
            help="The path to the config file(s). "
                 "Multiple config files are merged.",
            default=[get_pkg_data_path().joinpath("tolteca.yaml")])
    parser.add_argument(
            "-q", "--quiet",
            help="Suppress debug logs.",
            action='store_true')

    args, unparsed_args = parser.parse_known_args()

    if args.quiet:
        loglevel = 'INFO'
    else:
        loglevel = 'DEBUG'
    init_log(level=loglevel)

    logger = get_logger()
    logger.debug(f'load config from {args.config}')

    conf = None
    for c in args.config:
        with open(c, 'r') as fo:
            if conf is None:
                conf = yaml.safe_load(fo)
            else:
                rupdate(conf, yaml.safe_load(fo))
    app = qt5app(unparsed_args)

    gui = Gui()
    gui.init_runtime(conf or dict())
    gui.show()
    sys.exit(app.exec_())
