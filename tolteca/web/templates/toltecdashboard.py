#! /usr/bin/env python


from dasha.web.templates import ComponentTemplate, ComponentRoot
import dash_bootstrap_components as dbc
import dash_html_components as html
from ..tasks.ocs3 import get_ocs3_info_store
from ..tasks.ocs3 import get_ocs3_api
from dash_table import DataTable
import dash_core_components as dcc
from dash.dependencies import Input, Output
from tollan.utils.log import timeit, get_logger
from plotly.subplots import make_subplots
from tollan.utils import odict_from_list
import pandas as pd
from dasha.web.templates.collapsecontent import CollapseContent
from dasha.web.templates.valueview import ValueView
from tollan.utils import mapsum
from tollan.utils.fmt import pformat_yaml
import dash
import cachetools.func
import functools
import re


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
                app, container, 'ROACH', 1000)
        body = container.child(dbc.Row).child(dbc.Col)
        info_container = body.child(
                dbc.Container, fluid=True, className='mx-0')
        details_container = body.child(
                CollapseContent(button_text='Details ...')).content

        obj_name = 'ToltecBackend'
        attrs = self._get_ocs3_attrs(
                obj_name,
                lambda a: a.get('dims', None) == ['TOLTEC_NUM_ROACHES', ])

        @timeit
        @cachetools.func.ttl_cache(maxsize=1, ttl=1)
        def query_attrs():
            # logger = get_logger()
            store = get_ocs3_info_store()

            result = dict()
            with store.pipeline as p:
                for attr in attrs:
                    store.get(
                        f'{obj_name}.attrs.{attr["name"]}')
                response = p.try_execute()
                if response is None:
                    return None
            for attr, data in zip(attrs, response):
                # logger.debug(f"attr: {attr}\ndata: {data}")
                result[attr['name']] = data
            # turn the data to data frame
            return pd.DataFrame(result)

        # setup info view
        tbl_info = info_container.child(
                DataTable,
                style_table={'overflowX': 'scroll'},
                )

        @app.callback(
                [
                    Output(tbl_info.id, 'columns'),
                    Output(tbl_info.id, 'data'),
                    Output(loading.id, 'children'),
                    Output(error.id, 'children'),
                    ],
                [
                    Input(timer.id, 'n_intervals')
                    ]
                )
        def update_tbl_info(n_intervals):
            info = query_attrs()
            if info is None:
                return (
                        dash.no_update, dash.no_update, dash.no_update,
                        self._data_not_available())
            info = info.T
            info.insert(0, 'Roach', info.index)
            columns = [
                    {"name": i, "id": i} for i in info.columns]
            data = info.to_dict('records')
            return columns, data, "", ""

        @app.callback(
                Output(details_container.id, 'children'),
                [
                    Input(timer.id, 'n_intervals')
                    ],
                )
        def update_details(n_intervals):
            info = query_attrs()
            return html.Pre(pformat_yaml(info))

    def _setup_section_kids(self, app, container):
        timer, loading, error = self._setup_live_update_header(
                app, container, 'KIDs', 2000)
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
            with store.pipeline as p:
                store.get(detector_values_path)
                response = p.try_execute()
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
            import numpy as np
            data = np.random.randn(len(data), len(data[0]))

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
                    coloraxis=dict(colorscale='RdBu'), showlegend=False,
                    margin=dict(t=60),
                    )
            for i, array_prop in enumerate(array_props.values()):
                roach_indices = list(range(
                        *array_prop['roaches'].indices(len(data))))
                trace = {
                        'name': array_prop['name_long'],
                        'type': 'heatmap',
                        'z': data[array_prop['roaches']],
                        'y': roach_indices,
                        'coloraxis': 'coloraxis1'
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
            result = {}
            attrs_all = mapsum(
                    lambda i: [
                        dict(a, _objname=i[0])
                        for a in i[1]
                        ],
                    attrs_map.items()
                    )

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
            return result

        def get_view_kwargs(key_attr, **kwargs):

            # define the helpers
            def get_therm_temp_text(info, i):
                d = info['ToltecThermetry']
                if d['ChanStatus'][i] > 0:
                    return html.Span(
                                f"Error[{d['ChanStatus'][i]}]",
                                className='text-muted')
                return '{:.2f} K'.format(d['Temperature'][i])

            def get_therm_temp_bar(info, i):
                d = info['ToltecThermetry']
                if d['ChanStatus'][i] > 0:
                    return 0
                return 1

            def get_formatted_value_text(info, key, attr, fmt):
                return fmt.format(info[key][attr])

            def get_formatted_value_bar(info, key, attr):
                return 0.5

            def get_attr_temp_fmt(attr):
                if 'TempSigTemp' in attr:
                    return "{:.2f} K"
                if re.match(r'(Cool.+Temp|.+PtcSigW[io]t)', attr):
                    return "{:.2f} ℉"

            bar_kwargs = {
                            'max': 1.,
                            'step': 0.02,
                            'size': 100,
                            }
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
                    'label': labels[i],
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
                    'label': attr,
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
        info_groups = [
            {
                'name': '0.1 K',
                'attrs': [
                    ('ToltecThermetry.15', '1.1mm_0.1K'),  # 0.1K_high
                    ('ToltecThermetry.2', '1.4mm_0.1K'),  # 1.4mm_0.1K_high
                    ('ToltecThermetry.14', '2.0mm_0.1K'),  # 2mm_0.1K_high
                    # tuple is used to overwrite the label
                    ('ToltecDilutionFridge.StsDevT12TempSigTemp', 'MC')
                    ]
                },
            {
                'name': '1 K',
                'attrs': [
                    'ToltecThermetry.13',  # 1K_high
                    'ToltecThermetry.3',   # 1.4mm_1k_low
                    'ToltecThermetry.12',  # 2mm_1k_low
                    ('ToltecDilutionFridge.StsDevT11TempSigTemp', 'Still')
                    ]
                },
            {
                'name': '',
                'attrs': [
                    'ToltecThermetry.5',   # OB_CERNOX
                    ('ToltecThermetry.9', '4K AuxPTC'),  # 4K_AuxPTC_Busbar
                    ('ToltecThermetry.10', '4K DltFrg'),  # 4K_DF_bar
                    ('ToltecDilutionFridge.StsDevT1TempSigTemp', 'PT2 Head'),
                    ('ToltecDilutionFridge.StsDevT6TempSigTemp', 'PT1 Head'),
                    ('ToltecDilutionFridge.StsDevT16TempSigTemp', 'AuxPTC4'),
                    ('ToltecDilutionFridge.StsDevT15TempSigTemp', 'AuxPTC3'),
                    ('ToltecDilutionFridge.StsDevT14TempSigTemp', 'AuxPTC2'),
                    ('ToltecDilutionFridge.StsDevT13TempSigTemp', 'AuxPTC1'),
                    ]
                },
            {
                'name': 'Water',
                'attrs': [
                    ('ToltecCryocmp.CoolInTemp', 'CryoCmp In'),
                    ('ToltecCryocmp.CoolOutTemp', 'CryoCmp Out'),
                    ('ToltecDilutionFridge.StsDevC1PtcSigWit', 'DltFrg In'),
                    ('ToltecDilutionFridge.StsDevC1PtcSigWot', 'DltFrg Out'),
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
                for k in g['attrs']:
                    if isinstance(k, tuple):
                        s = get_view_kwargs(k[0], label=k[1])
                    else:
                        s = get_view_kwargs(k)
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
