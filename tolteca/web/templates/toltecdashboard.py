#! /usr/bin/env python


from dasha.web.templates import ComponentTemplate, ComponentRoot
import dash_bootstrap_components as dbc
import dash_html_components as html
from ..tasks.ocs3 import get_ocs3_info_store
from ..tasks.ocs3 import get_ocs3_api
from dash_table import DataTable
import dash_core_components as dcc
from dash.dependencies import Input, Output
from tollan.utils.log import timeit, get_logger, disable_logger
from plotly.subplots import make_subplots
from tollan.utils import odict_from_list
import numpy as np
import pandas as pd
from dasha.web.templates.collapsecontent import CollapseContent
from dasha.web.templates.valueview import ValueView
from dasha.web.templates.ipcinfo import ReJsonIPCInfo
from tollan.utils import mapsum
from tollan.utils.fmt import pformat_yaml, pformat_bar
import dash
import cachetools.func
import functools
import re
from scipy.interpolate import interp1d
# import dash_defer_js_import as dji


class ToltecDashboard(ComponentTemplate):
    _component_cls = dbc.Container
    fluid = True
    logger = get_logger()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ocs3_api = get_ocs3_api()

    def setup_layout(self, app):

        container = self
        header = container.child(dbc.Row)
        title_container = header.child(dbc.Col)
        title_container.child(html.H1, 'TolTEC Dashboard')
        title_container.child(
                ReJsonIPCInfo(get_ocs3_info_store, timeout_thresh=3))
        title_container.child(html.Hr)
        body = self.child(dbc.Row)
        for section_name, kwargs in [
                ('roach', dict(
                    xl=dict(size=8, order=1),
                    xs=dict(size=12, order=1)
                    )),
                ('therm', dict(
                    xl=dict(size=4, order=1),
                    xs=dict(size=12, order=3)
                    )),
                ('kids', dict(
                    xl=dict(size=12, order=1),
                    xs=dict(size=12, order=2)
                    )),
                ]:
            getattr(self, f'_setup_section_{section_name}')(
                    app, body.child(
                        dbc.Col, className='mb-4',
                        style={
                            'min-width': '375px'
                            },
                        **kwargs))

        # https://github.com/yueyericardo/dash_latex/blob/master/Example1/free_particle.py
        # container.child(dji.Import, src='https://codepen.io/yueyericardo/pen/pojyvgZ.js')  # noqa: E501
        # container.child(dji.Import, src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js?config=TeX-AMS-MML_SVG')   # noqa: E501
        super().setup_layout(app)

    def _get_ocs3_attrs(self, obj_name, filter_):
        return [
                attr for attr in self._ocs3_api[obj_name]['attrs'].values()
                if filter_(attr)
                ]

    def _setup_live_update_header(self, app, container, title, interval):
        header = container.child(dbc.Row, className='mb-2').child(
                dbc.Col, className='d-flex align-items-center')
        header.child(html.H3, title, className='mr-4 my-0')
        timer = header.child(dcc.Interval, interval=interval)
        loading = header.child(dcc.Loading)
        error = container.child(dbc.Row).child(dbc.Col)
        return timer, loading, error

    @staticmethod
    def _data_not_available():
        return dbc.Alert('Unable to get data', color='danger')

    def _setup_section_roach(self, app, container):
        timer, loading, error = self._setup_live_update_header(
                app, container, 'ROACH', 2000)
        body = container.child(dbc.Row).child(dbc.Col)
        info_container = body.child(
                dbc.Container, fluid=True, className='mx-0')
        info_meta_container = body.child(
                dbc.Container, fluid=True, className='mx-0')
        details_container = body.child(
                CollapseContent(button_text='Details ...')).content

        obj_name = 'ToltecBackend'
        attrs = self._get_ocs3_attrs(
                obj_name,
                lambda a: a.get('dims', None) == ['TOLTEC_NUM_ROACHES', ])
        # this store the toltec backend global attrs
        attrs_meta = self._get_ocs3_attrs(
                obj_name,
                lambda a: a.get('dims', None) is None)

        @timeit
        @cachetools.func.ttl_cache(maxsize=1, ttl=1)
        def query_attrs():
            # logger = get_logger()
            store = get_ocs3_info_store()
            if store.is_null():
                return None, None
            result = dict()
            result_meta = dict()
            with disable_logger('rejson query'):
                with store.pipeline as p:
                    for attr in attrs:
                        store.get(
                            f'{obj_name}.attrs.{attr["name"]}')
                    for attr in attrs_meta:
                        store.get(
                            f'{obj_name}.attrs.{attr["name"]}')
                    response = p.try_execute()
                    if response is None:
                        return None, None
            n_attrs = len(attrs)
            for i, (attr, data) in enumerate(
                    zip(attrs + attrs_meta, response)):
                # logger.debug(f"attr: {attr}\ndata: {data}")
                if i < n_attrs:
                    result[attr['name']] = data
                else:
                    result_meta[attr['name']] = data
            # turn the data to data frame
            result = pd.DataFrame(result)
            # do some post processing
            result = result.drop(
                    ['ClockCount', 'ClockTime', 'StatusReg'], axis=1)
            result['SampleFreq'] = result['SampleFreq'].apply(
                    lambda x: f'{float(x):.2f}')
            result['ActionPercent'] = result['ActionPercent'].apply(
                    lambda v: pformat_bar(
                        v / 100, width=7, border=False,
                        fill='·', reverse=True) + f'{v:.0f}%')
            # parse bitwise state
            is_streaming = ((1 << 7) & result['BitwiseState']) > 0
            result['Streaming'] = ['🟢' if s else '🔴' for s in is_streaming]
            is_selected = [
                    ((1 << i) & int(result_meta['SelectedMask'], 16))
                    for i in result.index]
            result['Selected'] = ['🟢' if s else '🔴' for s in is_selected]
            result = result.reindex(columns=[
                'Selected',
                'Streaming',
                'PpsCount', 'PacketCount', 'NumKids',
                'AttenInput', 'AttenOutput',
                'SampleFreq', 'LoFreq',
                'BitwiseState',
                'CommandName',
                'ActionQuantity', 'ActionProgress', 'ActionPercent',
                ])
            return result, result_meta

        # setup info view
        tbl_info = info_container.child(
                DataTable,
                style_table={'overflowX': 'scroll'},
                style_data_conditional=[
                    {
                        # 'if': {
                        #     'filter_query': 'Roach = "ActionPercent"'
                        #     },
                        # 'if': {
                        #    'row_index': 13  # TODO avoid hardcoding this by making the above work  # noqa: E501
                        #    },
                        # 'backgroundColor': '#FF4136',
                        # 'color': 'white',
                        # 'textAlign': 'left',
                        # 'whiteSpace': 'pre-line',
                        },
                    ]
                )

        @app.callback(
                [
                    Output(tbl_info.id, 'columns'),
                    Output(tbl_info.id, 'data'),
                    Output(tbl_info.id, 'style_cell'),
                    Output(loading.id, 'children'),
                    Output(error.id, 'children'),
                    ],
                [
                    Input(timer.id, 'n_intervals')
                    ]
                )
        def update_tbl_info(n_intervals):
            info, info_meta = query_attrs()
            if info is None:
                return (
                        dash.no_update, dash.no_update,
                        dash.no_update, dash.no_update,
                        self._data_not_available())
            # transpose for better looking
            info = info.T
            info.insert(0, 'Roach', info.index)

            # m_selected = int(info_meta['SelectedMask'], 16)

            def make_name_indicator(i):
                if i == 13:
                    n = 'HWP'
                else:
                    n = f'{i}'
                return n
            columns = [
                    {"name": make_name_indicator(i), "id": i}
                    for i in info.columns]
            data = info.to_dict('records')
            style_cell = {
                'width': '3rem',
                }
            return columns, data, style_cell, "", ""

        @app.callback(
                [
                    Output(details_container.id, 'children'),
                    Output(info_meta_container.id, 'children'),
                    ],
                [
                    Input(timer.id, 'n_intervals')
                    ],
                )
        def update_details(n_intervals):
            info, info_meta = query_attrs()
            if info is None:
                return ("Error getting info.", "")
            return (
                    html.Pre(f'{pformat_yaml(info_meta)}\n{info}'),
                    html.Pre(
                        f'Heartbeat: {info_meta["Heartbeat"]} '
                        f'ObsNum: {info_meta["ObsNum"]}'
                        )
                    )

    def _setup_section_kids(self, app, container):
        timer, loading, error = self._setup_live_update_header(
                app, container, 'KIDs', 4000)
        body = container.child(dbc.Row, className='mt-n2').child(dbc.Col)
        kids_info_graph = body.child(
                dcc.Graph,
                )
        detector_values_path = 'ToltecDetectors.attrs.Values'

        array_props = odict_from_list([
                {
                    'name': 'a1100',
                    'name_long': 'TolTEC 1.1mm',
                    'roaches': slice(0, 7),
                    },
                {
                    'name': 'a1400',
                    'name_long': 'TolTEC 1.4mm',
                    'roaches': slice(7, 11),
                    },
                {
                    'name': 'a2000',
                    'name_long': 'TolTEC 2.0mm',
                    'roaches': slice(11, 13),
                    },
                ], key='name')
        n_arrays = len(array_props)

        @timeit
        def query_detector_values():
            logger = get_logger()
            store = get_ocs3_info_store()
            if store.is_null():
                return None
            with disable_logger('rejson query'):
                with store.pipeline as p:
                    store.get(detector_values_path)
                    response = p.try_execute(result_index=-1)
                    if response is None:
                        return None
            data = response
            logger.debug(f"data: {len(data)}")
            return data

        @app.callback(
                [
                    Output(kids_info_graph.id, 'figure'),
                    Output(loading.id, 'children'),
                    Output(error.id, 'children'),
                    ],
                [
                    Input(timer.id, 'n_intervals')
                    ]
                )
        def update_kids_info(n_intervals):
            data = query_detector_values()
            if data is None:
                return (
                        dash.no_update,
                        dash.no_update,
                        self._data_not_available(),)
            # import numpy as np
            # data = np.random.randn(len(data), len(data[0]))

            fig = make_subplots(
                    rows=n_arrays, cols=1, start_cell="top-left",
                    shared_xaxes=True,
                    x_title='Channel Index',
                    y_title='ROACH Index',
                    subplot_titles=[
                        a['name_long'] for a in array_props.values()],
                    row_heights=[
                        a['roaches'].stop - a['roaches'].start
                        for a in array_props.values()],
                    vertical_spacing=0.07,
                    )
            fig.update_layout(
                    uirevision=True,
                    coloraxis=dict(
                        colorscale='Viridis',
                        cmin=4, cmax=8,
                        colorbar=dict(
                            title={
                                'text': 'Log10 (I^2 + Q^2)',
                                'side': 'right',
                                },
                            )
                        ),
                    showlegend=False,
                    margin=dict(t=60),
                    )
            for i, array_prop in enumerate(array_props.values()):
                roach_indices = list(range(
                        *array_prop['roaches'].indices(len(data))))
                # mask out the invalid values
                z = np.asarray(data[array_prop['roaches']]).astype('d')
                z[z <= 0] = np.nan
                trace = {
                        'name': array_prop['name_long'],
                        'type': 'heatmap',
                        'z': np.log10(z),
                        'y': roach_indices,
                        'coloraxis': 'coloraxis1',
                        }
                fig.add_trace(trace, i + 1, 1)
                # fig.update_xaxes(
                #         row=i + 1, col=1,
                #         title=array_prop['name_long'])
                fig.update_yaxes(
                        row=i + 1, col=1,
                        tickvals=roach_indices, ticktext=roach_indices,
                        range=[
                            roach_indices[-1] + 0.5,
                            roach_indices[0] - 0.5,
                            ],
                        )

            return fig, "", ""

    def _setup_section_therm(self, app, container):
        timer, loading, error = self._setup_live_update_header(
                app, container, 'Cryostat', 5000)
        body = container.child(dbc.Row).child(dbc.Col)
        info_container = body.child(
                dbc.Container, fluid=True, className='mx-0')
        details_container = body.child(
                CollapseContent(button_text='Details ...')).content

        # this defines the items to query.
        attrs_map = {
                objname: self._get_ocs3_attrs(objname, filter_)
                for objname, filter_ in [
                    (
                        'ToltecThermetry',
                        lambda a: a.get('dims', None) == [16, ]),
                    (
                        'ToltecDilutionFridge',
                        lambda a: re.match(r'^StsDev.+', a['name'])),
                    (
                        'ToltecCryocmp', lambda a: True
                        )
                    ]
                }

        @timeit
        @cachetools.func.ttl_cache(maxsize=1, ttl=1)
        def query_attrs():
            # logger = get_logger()
            store = get_ocs3_info_store()
            if store.is_null():
                return None
            result = {}
            attrs_all = mapsum(
                    lambda i: [
                        dict(a, _objname=i[0])
                        for a in i[1]
                        ],
                    attrs_map.items()
                    )
            with disable_logger('rejson query', 'query_obj'):
                with store.pipeline as p:
                    for attr in attrs_all:
                        store.get(
                            f'{attr["_objname"]}.attrs.{attr["name"]}')
                    response = p.try_execute()
                    if response is None:
                        return None
            for attr, data in zip(attrs_all, response):
                # logger.debug(f"attr: {attr}\ndata: {data}")
                key = attr['_objname']
                if key not in result:
                    result[key] = dict()
                result[key][attr['name']] = data
            # turn the term data to data frame
            result['ToltecThermetry'] = pd.DataFrame(
                    result['ToltecThermetry'])
            # logger.debug(f"result:\n{pformat_yaml(result)}")

            # fix the dilfrg temp unit
            def C2F(c):
                return c * 1.8 + 32
            dltfrg = result['ToltecDilutionFridge']
            dltfrg['StsDevC1PtcSigWit'] = C2F(dltfrg['StsDevC1PtcSigWit'])
            dltfrg['StsDevC1PtcSigWot'] = C2F(dltfrg['StsDevC1PtcSigWot'])
            dltfrg['StsDevC1PtcSigOilt'] = C2F(dltfrg['StsDevC1PtcSigOilt'])
            return result

        def get_view_kwargs(key_attr, **kwargs):
            # this processes the info spec to prepare kwargs to pass to value
            # view.
            format_view_text = kwargs.pop('format_view_text', None)
            label = kwargs.pop('label', None)

            bar_lims = [0, 0.4, 0.7, 1]
            bar_ranges = {
                    "#92e0d3": bar_lims[0:2],
                    "#f4d44d ": bar_lims[1:3],
                    "#f45060": bar_lims[2:4],
                    }
            value_lims = kwargs.pop('lims', None)
            if value_lims is None:
                value_lims = bar_lims
            # here we rescale the ranges such that it is piece wise linear
            bar_interp = interp1d(
                    value_lims, bar_lims,
                    fill_value=(bar_lims[0], bar_lims[1]),
                    bounds_error=False)
            bar_kwargs = {
                            'min': bar_lims[0],
                            'max': bar_lims[-1],
                            'step': (bar_lims[-1] - bar_lims[0]) / 50.,
                            'size': 100,
                            'color': {
                                'ranges': bar_ranges
                                }
                            }

            # define the helpers
            def get_therm_temp_text(info, i):
                d = info['ToltecThermetry']
                if d['ChanStatus'][i] > 0:
                    return html.Span(
                                f"Error[{d['ChanStatus'][i]}]",
                                className='text-muted')
                if format_view_text is not None:
                    return format_view_text(d['Temperature'][i])
                return '{:.2f} K'.format(d['Temperature'][i])

            def get_therm_temp_bar(info, i):
                d = info['ToltecThermetry']
                if d['ChanStatus'][i] > 0:
                    return 0
                return bar_interp(d['Temperature'][i])

            def get_formatted_value_text(info, key, attr, fmt):
                if format_view_text is not None:
                    return format_view_text(info[key][attr])
                return fmt.format(info[key][attr])

            def get_formatted_value_bar(info, key, attr):
                return bar_interp(info[key][attr])

            def get_attr_temp_fmt(attr):
                if 'TempSigTemp' in attr:
                    return "{:.2f} K"
                if re.match(r'(Cool.+Temp|.+PtcSigW[io]t)', attr):
                    return "{:.2f} ℉"
                return "{:.2f} ℉"

            label_container_kwargs = {
                'className': 'd-flex flex-fill',
                'style': {
                    'font-size': 14
                    }
                }

            key, attr = key_attr.split('.')
            result = dict()
            if key == 'ToltecThermetry':
                i = int(attr)
                # as of 20200525
                labels = [
                        '2.0LNA_Stand', '2mm_array_front',
                        '1.4mm_0.1K_high', '1.4mm_1k_low',
                        'busbar_stand_top', 'OB_CERNOX',
                        'n/c', 'LS_aperture', '1.4LNA_Stand',
                        '4K_AuxPTC_Busbar', '4K_DF_bar',
                        'n/c', '2mm_1k_low', '1K_high',
                        '2mm_0.1K_high', '0.1K_high'
                        ]
                result.update({
                    'key': key_attr,
                    'label_container': label_container_kwargs,
                    'label': labels[i] if label is None else label,
                    'text': {
                        'func': functools.partial(get_therm_temp_text, i=i)
                        },
                    'bar': dict({
                        'func': functools.partial(get_therm_temp_bar, i=i)
                        }, **bar_kwargs),
                    })
            else:
                # key value pairs
                result.update({
                    'key': key_attr,
                    'label_container': label_container_kwargs,
                    'label': attr if label is None else label,
                    'text': {
                        'func': functools.partial(
                            get_formatted_value_text,
                            key=key, attr=attr,
                            fmt=get_attr_temp_fmt(attr)
                            )
                        },
                    'bar': dict({
                        'func': functools.partial(
                            get_formatted_value_bar,
                            key=key, attr=attr)
                        }, **bar_kwargs),
                    })
            result.update(kwargs)
            return result

        # arrange the view items to groups
        info_bar_lims = {
                '0.1K': [0.1,  0.175, 0.2, 0.25],
                '1.0K': [0.9, 1.25, 1.75, 2.0],
                '4.0K': [4.0, 5.5, 6.0, 300],
                'PT1 Head and AuxPTC1': [30, 40, 50, 300],
                'CryoCmpIn and DltFrg In': [40, 50, 60, 70],
                'CryoCmpOut and DltFrg Out': [85, 90, 100, 110],
                }
        info_view_text_formatters = {
                '0.1K': (lambda v: f'{v * 1e3:.0f} mK'),
                }
        info_groups = [
            {
                'name': '0.1 K',
                'attrs': [
                    # key, label (None to use default), bar lims (None to disable), view_kwargs  # noqa: E501
                    ('ToltecThermetry.15', dict(label='1.1mm_0.1K', lims=info_bar_lims['0.1K'], format_view_text=info_view_text_formatters['0.1K'])),  # 0.1K_high         # noqa: E501
                    ('ToltecThermetry.2', dict(label='1.4mm_0.1K', lims=info_bar_lims['0.1K'], format_view_text=info_view_text_formatters['0.1K'])),  # 1.4mm_0.1K_high    # noqa: E501
                    ('ToltecThermetry.14', dict(label='2.0mm_0.1K', lims=info_bar_lims['0.1K'], format_view_text=info_view_text_formatters['0.1K'])),  # 2mm_0.1K_high     # noqa: E501
                    ('ToltecDilutionFridge.StsDevT12TempSigTemp', dict(label='MC', lims=info_bar_lims['0.1K'], format_view_text=info_view_text_formatters['0.1K'])),       # noqa: E501
                    ]
                },
            {
                'name': '1 K',
                'attrs': [
                    ('ToltecThermetry.13', dict(lims=info_bar_lims['1.0K'])),  # '1K_high'       # noqa: E501
                    ('ToltecThermetry.3', dict(lims=info_bar_lims['1.0K'])),  # '1.4mm_1k_low'   # noqa: E501
                    ('ToltecThermetry.12', dict(lims=info_bar_lims['1.0K'])),  # '2mm_1k_low'    # noqa: E501
                    ('ToltecDilutionFridge.StsDevT11TempSigTemp', dict(label='Still')),          # noqa: E501
                    ]
                },
            {
                'name': '',
                'attrs': [
                    ('ToltecThermetry.5', dict(lims=info_bar_lims['4.0K'])),  # OB_CERNOX                                                 # noqa: E501
                    ('ToltecThermetry.9', dict(label='4K AuxPTC', lims=info_bar_lims['4.0K'])),  # 4K_AuxPTC_Busbar                       # noqa: E501
                    ('ToltecThermetry.10', dict(label='4K DltFrg', lims=info_bar_lims['4.0K'])),  # 4K_DF_bar                             # noqa: E501
                    ('ToltecDilutionFridge.StsDevT1TempSigTemp', dict(label='PT2 Head', lims=info_bar_lims['4.0K'])),                     # noqa: E501
                    ('ToltecDilutionFridge.StsDevT6TempSigTemp', dict(label='PT1 Head', lims=info_bar_lims['PT1 Head and AuxPTC1'])),     # noqa: E501
                    ('ToltecDilutionFridge.StsDevT16TempSigTemp', dict(label='AuxPTC4', lims=info_bar_lims['4.0K'])),                     # noqa: E501
                    ('ToltecDilutionFridge.StsDevT15TempSigTemp', dict(label='AuxPTC3', lims=info_bar_lims['4.0K'])),                     # noqa: E501
                    ('ToltecDilutionFridge.StsDevT14TempSigTemp', dict(label='AuxPTC2', lims=info_bar_lims['4.0K'])),                     # noqa: E501
                    ('ToltecDilutionFridge.StsDevT13TempSigTemp', dict(label='AuxPTC1', lims=info_bar_lims['PT1 Head and AuxPTC1'])),     # noqa: E501
                    ]
                },
            {
                'name': 'Water',
                'attrs': [
                    ('ToltecCryocmp.OilTemp', dict(label='CryoCmp Oil')),                                                                         # noqa: E501
                    ('ToltecCryocmp.CoolInTemp', dict(label='CryoCmp In', lims=info_bar_lims['CryoCmpIn and DltFrg In'])),      # noqa: E501
                    ('ToltecCryocmp.CoolOutTemp', dict(label='CryoCmp Out', lims=info_bar_lims['CryoCmpOut and DltFrg Out'])),      # noqa: E501
                    ('ToltecDilutionFridge.StsDevC1PtcSigOilt', dict(label='DltFrg Oil')),      # noqa: E501
                    ('ToltecDilutionFridge.StsDevC1PtcSigWit', dict(label='DltFrg In', lims=info_bar_lims['CryoCmpIn and DltFrg In'])),      # noqa: E501
                    ('ToltecDilutionFridge.StsDevC1PtcSigWot', dict(label='DltFrg Out', lims=info_bar_lims['CryoCmpOut and DltFrg Out'])),      # noqa: E501
                    ]
                }
            ]

        @app.callback(
                [
                    Output(info_container.id, 'children'),
                    Output(loading.id, 'children'),
                    Output(error.id, 'children'),
                    ],
                [
                    Input(timer.id, 'n_intervals')
                    ],
                )
        def update_info(n_intervals):
            # fill the view values
            info = query_attrs()
            if info is None:
                return (
                        dash.no_update, dash.no_update,
                        self._data_not_available())
            # build the views
            content_root = ComponentRoot(id=info_container.id)
            info_views = []
            for i, g in enumerate(info_groups):
                info_container_row = content_root.child(dbc.Row)
                if i > 0:
                    info_container_row.child(
                            dbc.Col, width=12).child(html.Hr, className='my-2')
                for k, kw in g['attrs']:
                    s = get_view_kwargs(k, **kw)
                    self.logger.debug(
                            f"view kwargs:\n{pformat_yaml(s)}")
                    info_views.append(
                        info_container_row.child(
                            dbc.Col, xl=12, xs=12).child(
                            ValueView(**s)))

            # fill view value
            for view in info_views:
                for co, c in zip(view.outputs, view.output_components):
                    setattr(c, co.component_property, c.func(info))
            return content_root.layout, "", ""

        @app.callback(
                Output(details_container.id, 'children'),
                [
                    Input(timer.id, 'n_intervals')
                    ],
                )
        def update_details(n_intervals):
            info = query_attrs()
            return html.Pre(pformat_yaml(info))
