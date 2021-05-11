#! /usr/bin/env python


from dasha.web.templates import ComponentTemplate
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
from tollan.utils.log import get_logger
from plotly.subplots import make_subplots as _make_subplots
from tollan.utils import odict_from_list
import numpy as np
from dasha.web.templates.collapsecontent import CollapseContent
from tollan.utils.fmt import pformat_yaml
import cachetools.func
import functools
# import dash_defer_js_import as dji
from .. import fs_toltec_hk_rootpath
import astropy.units as u
from tollan.utils.nc import NcNodeMapper
from pathlib import Path


def make_subplots(nrows, ncols, fig_layout=None, **kwargs):
    _fig_layout = {
            'uirevision': True,
            'xaxis_autorange': True,
            'yaxis_autorange': True,
            'showlegend': True,
            }
    if fig_layout is not None:
        _fig_layout.update(fig_layout)
    fig = _make_subplots(nrows, ncols, **kwargs)
    fig.update_layout(**_fig_layout)
    return fig


def make_labeled_drp(form, label, **kwargs):
    igrp = form.child(dbc.InputGroup, size='sm', className='pr-2')
    igrp.child(dbc.InputGroupAddon(label, addon_type='prepend'))
    return igrp.child(dbc.Select, **kwargs)


class HkDataViewer(ComponentTemplate):
    _component_cls = dbc.Container

    fluid = True
    logger = get_logger()

    hkdata_spec = odict_from_list([
            {
                'key': 'cryocmp',
                'name_long': 'Compressor',
                'filename_stem': 'cryocmp',
                },
            {
                'key': 'dltfrg',
                'name_long': 'Dilution Fridge',
                'filename_stem': 'dilutionFridge',
                },
            {
                'key': 'therm',
                'name_long': 'Thermometry',
                'filename_stem': 'thermetry',
                },
            ], key='key')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup_layout(self, app):

        container = self
        header = container.child(dbc.Row)
        title_container = header.child(dbc.Col)
        title_container.child(html.H1, 'TolTEC Housekeeping Data')

        body = self.child(dbc.Row)
        for section_name, kwargs in [
                ('temp', dict(
                    )),
                # ('pres', dict(
                #     )),
                # ('stat', dict(
                #     )),
                ]:
            getattr(self, f'_setup_section_{section_name}')(
                    app, body.child(
                        dbc.Col, className='mb-4',
                        style={
                            'min-width': '375px'
                            },
                        **kwargs))
        super().setup_layout(app)

    def _setup_live_update_header(self, app, container, title, interval):
        header = container.child(dbc.Row, className='mb-2').child(
                dbc.Col, className='d-flex align-items-center')
        header.child(html.H3, title, className='mr-4 my-0')
        timer = header.child(dcc.Interval, interval=interval)
        loading = header.child(dcc.Loading)
        error = container.child(dbc.Row).child(dbc.Col)
        return timer, loading, error

    def _setup_section_temp(self, app, container):
        timer, loading, error = self._setup_live_update_header(
                app, container, 'Temperatures', 3000)
        body = container.child(dbc.Row).child(dbc.Col)
        details_container = body.child(
                CollapseContent(button_text='Details ...')).content

        controls_container, graph_container = container.child(
                dbc.Row).child(dbc.Col).grid(2, 1)

        controls_form = controls_container.child(dbc.Form, inline=True)

        datalen_drp = make_labeled_drp(
                controls_form, 'Show data of latest',
                options=[
                    {
                        'label': f'{n}',
                        'value': n,
                        }
                    for n in ['15 min', '30 min', '1 hr', '12 hr', '1 d']],
                value='15 min',
                )

        def get_therm_channel_labels(nm):
            strlen = nm.getdim(
                    'Header.ToltecThermetry.ChanLabel_slen')
            return list(map(
                lambda x: x.decode().strip(), nm.getvar(
                    'Header.ToltecThermetry.ChanLabel')[:].view(
                    f'S{strlen}').ravel()))

        def get_hkdata_filepath(hkdata_key):
            n = self.hkdata_spec[hkdata_key]['filename_stem']
            r = Path(fs_toltec_hk_rootpath).expanduser()
            for p in [
                    r.joinpath(n).joinpath(f'{n}.nc'),
                    r.joinpath(f'{n}.nc'),
                    ]:
                if p.exists():
                    return p
            else:
                return None

        @functools.lru_cache(maxsize=32)
        def _get_hkdata(filepath):
            return NcNodeMapper(source=filepath)

        @cachetools.func.ttl_cache(maxsize=1, ttl=1)
        def get_hkdata(hkdata_key):
            p = get_hkdata_filepath(hkdata_key)
            if p is None:
                return None
            nc = _get_hkdata(p.resolve().as_posix())
            nc.sync()
            return nc

        graph = graph_container.child(dcc.Graph)

        @app.callback(
                [
                    Output(graph.id, 'figure'),
                    Output(loading.id, 'children'),
                    Output(error.id, 'children'),
                    ],
                [
                    Input(timer.id, 'n_intervals'),
                    Input(datalen_drp.id, 'value'),
                    ]
                )
        def update_graph(n_intervals, datalen_value):

            def make_data(key, datalen_value):
                nc = get_hkdata(key)
                if nc is None:
                    return None
                # figure out sample rate
                time_var = {
                        'cryocmp': 'Data.ToltecCryocmp.Time',
                        'dltfrg': 'Data.ToltecDilutionFridge.SampleTime',
                        'therm': 'Data.ToltecThermetry.Time1',
                        }[key]
                dt = nc.getvar(time_var)[-2:]
                if len(dt) == 2:
                    dt = (np.diff(dt)[0]) << u.s
                else:
                    dt = 5 << u.s
                fsmp = 1 / dt
                if np.isinf(fsmp):
                    fsmp = 0.2 << u.Hz
                # calc the slice from datalen
                datalen_value, datalen_unit = datalen_value.split()
                datalen = datalen_value << u.Unit(datalen_unit)
                n_samples = int(
                        (datalen * fsmp).to_value(u.dimensionless_unscaled))
                if n_samples < 1:
                    n_samples = 1

                def F2K(t):
                    return (t - 32.) * 5. / 9. + 273.15

                def C2K(t):
                    return t + 273.15

                trans = {
                        "cryocmp": lambda x, y: (x, F2K(y)),
                        "dltfrg": lambda x, y: (x, C2K(y)),
                        "therm": None,
                        }[key]
                return {
                        'nc': nc,
                        'dt': dt,
                        'slice': slice(-n_samples, None),
                        'trans': trans
                        }
            data = {k: make_data(k, datalen_value)
                    for k in self.hkdata_spec.keys()}

            fig = make_subplots(1, 1)

            fig.update_xaxes(row=1, col=1, title=f'Time (UT)')
            fig.update_yaxes(
                    row=1, col=1, title='Temperature (K)',
                    )

            def make_trace_kwargs(d, x, y, name=None):
                if name is None:
                    name = y.rsplit('.', 1)[-1]

                kwargs = {
                        'type': 'scattergl',
                        'mode': 'lines+markers',
                        'name': name
                        }
                # read data
                nc = d['nc']
                slice_ = d['slice']
                x = nc.getvar(x)[slice_]
                y = nc.getvar(y)[slice_]
                trans = d['trans']
                if trans is not None:
                    x, y = trans(x, y)
                # x is in time
                x = np.asarray(x, dtype='d')
                y = np.asarray(y, dtype='d')
                m = x > 0
                x = x[m]
                y = y[m]
                x = np.asarray(x, dtype='datetime64[s]')
                kwargs.update({
                    'x': x,
                    'y': y
                    })
                return kwargs

            errors = []
            # add traces for all temp vars in cryocmp and dltfrg
            if data['cryocmp'] is not None:
                for dd in [
                        {
                            'd': data['cryocmp'],
                            'x': 'Data.ToltecCryocmp.Time',
                            'y': y,
                            }
                        for y in [
                           "Data.ToltecCryocmp.CoolInTemp",
                           "Data.ToltecCryocmp.CoolOutTemp",
                           "Data.ToltecCryocmp.OilTemp",
                           "Data.ToltecCryocmp.HeliumTemp",
                           ]
                        ]:
                    trace = make_trace_kwargs(**dd)
                    fig.append_trace(trace, row=1, col=1)
            else:
                errors.append(
                        dbc.Alert(
                            'cryocmp data not available', color='danger'))
            if data['dltfrg'] is not None:
                for dd in [
                        {
                            'd': data['dltfrg'],
                            'x': 'Data.ToltecDilutionFridge.SampleTime',
                            'y': y,
                            }
                        for y in [
                           "Data.ToltecDilutionFridge.StsDevC1PtcSigWit",
                           "Data.ToltecDilutionFridge.StsDevC1PtcSigWot",
                           "Data.ToltecDilutionFridge.StsDevC1PtcSigOilt",
                           "Data.ToltecDilutionFridge.StsDevC1PtcSigHt",
                           ]
                        ]:
                    trace = make_trace_kwargs(**dd)
                    fig.append_trace(trace, row=1, col=1)
            else:
                errors.append(
                        dbc.Alert(
                            'dltfrg data not available', color='danger'))

            if data['therm'] is not None:
                # get all therm labels
                for i, name in enumerate(
                        get_therm_channel_labels(data['therm']['nc'])):
                    trace = make_trace_kwargs(
                            d=data['therm'],
                            x=f"Data.ToltecThermetry.Time{i + 1}",
                            y=f"Data.ToltecThermetry.Temperature{i + 1}",
                            name=name
                            )
                    fig.append_trace(trace, row=1, col=1)
            else:
                errors.append(
                        dbc.Alert(
                            'therm data not available', color='danger'))
            return fig, "", errors

        @app.callback(
                Output(details_container.id, 'children'),
                [
                    Input(timer.id, 'n_intervals')
                    ],
                )
        def update_details(n_intervals):
            ncs = [get_hkdata(key) for key in self.hkdata_spec.keys()]
            d = {
                    'ncs': ncs
                    }
            return html.Pre(pformat_yaml(d))
