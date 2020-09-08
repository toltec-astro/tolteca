#! /usr/bin/env python


from tollan.utils.log import get_logger, timeit
from tollan.utils.fmt import pformat_yaml
from dasha.web.templates import ComponentTemplate
from dasha.web.templates.collapsecontent import CollapseContent
from dasha.web.templates.pager import ButtonListPager
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from .common.basic_obs_select_view import BasicObsSelectView
from dash.dependencies import Output, Input
import dash
from tolteca.datamodels.toltec import BasicObsData
import functools
import astropy.units as u
from plotly.subplots import make_subplots as _make_subplots
from dasha.web.templates.utils import to_dependency
from kidsproc.kidsdata import TimeStream, MultiSweep
import numpy as np
import json
from tollan.utils import odict_from_list
import itertools


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


@functools.lru_cache(maxsize=None)
@timeit
def get_kidsdata(file_loc):
    bod = BasicObsData(file_loc)
    kd = bod.read()
    kd.meta['time_obs'] = bod.meta['ut']
    return kd


def make_obs_label(meta):
    label = (
            f'{meta["obsnum"]}-{meta["subobsnum"]}-'
            f'{meta["scannum"]}')
    return label


class KidsExplorer(ComponentTemplate):
    _component_cls = html.Div

    logger = get_logger()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup_layout(self, app):

        container = self
        control_container, view_container = container.grid(2, 1)

        ctx = self._setup_control(app, control_container, {})
        ctx = self._setup_view(app, view_container, ctx)
        ctx = self._setup_graph(app, ctx['graph_container'], ctx)

    def _setup_control(self, app, container, ctx):
        bods_select_container = container.grid(1, 1)
        bods_select = bods_select_container.child(BasicObsSelectView())
        container.setup_layout(app)
        # we install a pager into the the bods select
        bods_control_container = bods_select.ctx['control_container']
        # add a row to bods control as a pager
        pager = bods_control_container.child(dbc.Row).child(
                dbc.Col, className='mt-2').child(ButtonListPager(
                    title_text='Select tones ...',
                    n_items_per_page_options=[20, 50, 100, 'All']
                    ))
        # lets move the pages btns into a dbc card
        # btns_container_parent = pager.btns_container.parent
        # pager.btns_container.parent = btns_container_parent.child(
        #         dbc.Card, className='my-2'
        #         ).child(dbc.CardBody, className='px-2 py-2')

        def update_n_items(dataitem_value, network_value):
            bods = bods_select.bods_from_select_inputs(
                    dataitem_value, network_value)
            return bods.bod_list[0].meta["n_tones"]

        pager.register_n_items_callback(
                inputs=bods_select.select_inputs,
                callback=update_n_items
                )

        # this has to be called manually because
        # the pager is added after the parent's setup_layout
        pager.setup_layout(app)

        def prepare_selected_data(
                dataitem_value, network_value, pager_data):
            if (
                    dataitem_value is None
                    or network_value is None or pager_data is None):
                return None
            bods = bods_select.bods_from_select_inputs(
                    dataitem_value, network_value)
            n_tones = bods.bod_list[0].meta["n_tones"]
            # if pager_data is None:
            #     return
            assert pager_data['stop'] <= n_tones
            tone_slice = slice(
                    pager_data['start'],
                    pager_data['stop'])
            return {'bods': bods,
                    'dataitem_value': dataitem_value,
                    'network_value': network_value,
                    'n_tones': n_tones,
                    'tone_slice': tone_slice,
                    'page_id': pager_data['page_id'],
                    'n_items': pager_data['n_items'],
                    'n_pages': pager_data['n_pages'],
                    'n_items_per_page': pager_data['n_items_per_page'],
                    }

        def prepare_selected_data_objs(*args):
            d = prepare_selected_data(*args)
            if d is None:
                raise dash.exceptions.PreventUpdate
            bods = d['bods']
            bods.sort(['obsnum', 'subobsnum', 'scannum'])
            data_objs = [get_kidsdata(bod.file_loc) for bod in bods]
            bods['data_obj'] = data_objs
            self.logger.debug(f"get {len(data_objs)} data objs.")
            tone_ids = range(*d['tone_slice'].indices(d['n_tones']))
            self.logger.debug(f"get {len(tone_ids)} tone_ids")
            d.update({
                'data_objs': data_objs,
                'tone_ids': tone_ids
                })
            return d

        select_inputs = bods_select.select_inputs + pager.page_data_inputs

        ctx.update({
            'select_inputs': select_inputs,
            'prepare_selected_data': prepare_selected_data,
            'prepare_selected_data_objs': prepare_selected_data_objs,
            })

        return ctx

    def _setup_view(self, app, container, ctx):

        graph_controls_container, graph_container = container.grid(2, 1)

        # details_container = graph_container.child(dbc.Row).child(dbc.Col)
        details_container = graph_controls_container.child(
                CollapseContent(button_text='Details ...')).content

        @app.callback(
                Output(details_container.id, 'children'),
                ctx['select_inputs'],
                )
        def update_details(*args):
            d = ctx['prepare_selected_data'](*args)
            if d is None:
                raise dash.exceptions.PreventUpdate
            bods = d.pop('bods')
            d['n_bods'] = len(bods)
            return html.Pre(pformat_yaml(d))

        container.setup_layout(app)
        ctx.update({
            'graph_controls_container': graph_controls_container,
            'graph_container': graph_container,
            })
        return ctx

    def _setup_graph(self, app, container, ctx):
        control_container, graph_container = container.grid(2, 1)
        g0, g1 = map(lambda c: c.child(dcc.Graph), graph_container.grid(2, 1))
        control_form = control_container.child(dbc.Form, inline=True)

        def make_labeled_drp(form, label, **kwargs):
            igrp = form.child(dbc.InputGroup, size='sm', className='pr-2')
            igrp.child(dbc.InputGroupAddon(label, addon_type='prepend'))
            return igrp.child(dbc.Select, **kwargs)

        f_unit = u.MHz

        select_mode_drp = make_labeled_drp(
                control_form, 'Click mode',
                options=[
                    {'label': 'collate tone', 'value': 'per_tone'},
                    {'label': 'individual', 'value': 'per_item'},
                    ],
                value='per_tone'
                )

        data_axis_items = odict_from_list([
                {
                    'key': 'kds',
                    'trace_name':
                    lambda d, idx: make_obs_label(d['kds'][idx].meta),
                    'trace_customdata': lambda *a: dict(),
                    # lambda d, idx: {'meta': d['kds'][idx].meta}
                    },
                {
                    'key': 'tone_ids',
                    'trace_name':
                    lambda d, idx: f'tone {d["tone_ids"][idx]}',
                    'trace_customdata':
                    lambda d, idx: {'tone_id': d["tone_ids"][idx]},
                    },
                ], key='key')

        axis_select_options = odict_from_list([
            {
                'key': 'obsnum',
                'label': 'obsnum',
                'value': None,  # this will use an implicit range
                'tick_value':
                lambda kd: make_obs_label(kd.meta),
                'map_data_axis_items': ['kds', ],
                'pass_data_items': ['kds', ],
                },
            {
                'key': 'ut',
                'label': 'obs. time (UT)',
                'value':
                lambda kd: np.datetime64(kd.meta['time_obs'].isot),
                'kwargs': {
                    'tickformat': '%Y-%m-%d %H:%M:%S'
                    },
                'map_data_axis_items': ['kds', ],
                'pass_data_items': ['kds', ],
                },
            {
                'key': 'f_center',
                'label': f'f_center ({f_unit})',
                'value':
                    lambda kd, ti:
                    kd.meta['tone_axis_data']['f_center'][ti].to_value(f_unit),
                'map_data_axis_items': ['kds', 'tone_ids'],
                'pass_data_items': ['kds', 'tone_ids'],
                },
            {
                'key': 'Qr',
                'label': f'Qr',
                'value':
                    lambda rkd, ti:
                    None if rkd is None else rkd.table['Qr'][ti],
                'map_data_axis_items': ['kds', 'tone_ids'],
                'pass_data_items': ['rkds', 'tone_ids'],
                },
            {
                'key': 'fr',
                'label': f'fr ({f_unit})',
                'value':
                    lambda rkd, ti:
                    None if rkd is None else
                    rkd.model.fr.quantity[ti].to_value(f_unit),
                'map_data_axis_items': ['kds', 'tone_ids'],
                'pass_data_items': ['rkds', 'tone_ids'],
                },

            ], key='key')

        axis_select_items = odict_from_list([
                {
                    'key': 'x',
                    'label': 'X-Axis',
                    'value': 'f_center',
                    },
                {
                    'key': 'y',
                    'label': 'Y-Axis',
                    'value': 'obsnum',
                    },
                {
                    'key': 'z',
                    'label': 'Z-Axis',
                    'value': None,
                    },
                ], key='key')

        axis_select_drps = dict()
        # create all the axis selection drpdowns
        for axis in axis_select_items.values():
            axis_select_drps[axis['key']] = make_labeled_drp(
                control_form,
                axis['label'],
                options=[
                    {
                        'label': d['label'],
                        'value': d['key'],
                        }
                    for d in axis_select_options.values()],
                value=axis['value'],
                )

        details_container = control_form.child(
                CollapseContent(button_text='Details ...')).content

        container.setup_layout(app)

        @timeit
        def get_data_objs(args, mask=None):
            d = ctx['prepare_selected_data_objs'](*args)
            if mask is None:
                mask = slice(None, None)
            elif isinstance(mask, int):
                mask = slice(mask, mask + 1)
            tone_ids = d['tone_ids']
            bods = d['bods'][mask]
            kds = d['data_objs'][mask]
            rbods = d['bods']['reduced_bod'][mask]
            rkds = [
                    None if rbod is None else get_kidsdata(rbod.file_loc)
                    for rbod in rbods
                    ]
            sbods = d['bods']['sweep_bod'][mask]
            skds = [
                    None if sbod is None else get_kidsdata(sbod.file_loc)
                    for sbod in sbods]
            rsbods = d['bods']['reduced_sweep_bod'][mask]
            rskds = [
                    None if rsbod is None else get_kidsdata(rsbod.file_loc)
                    for rsbod in rsbods]
            return locals()

        @app.callback(
                [
                    Output(g0.id, 'figure'),
                    Output(details_container.id, 'children'),
                    ],
                [
                    Input(drp.id, 'value')
                    for drp in axis_select_drps.values()
                    ]
                + ctx['select_inputs'],
                )
        @timeit
        def update_g0(*args):

            axis_select_values = args[:len(axis_select_drps)]
            d = get_data_objs(args[-len(axis_select_drps):])

            fig = make_subplots(
                    1, 1,
                    fig_layout={
                        'clickmode': 'event+select',
                        }
                    )

            # this maps the data items to get the value
            def map_func(func, items):
                shape = tuple(len(d[item]) for item in items)
                return np.asanyarray(list(itertools.starmap(
                    func, itertools.product(*(d[item] for item in items)))
                    )).reshape(shape)

            # this maps the data_axis_items to get trace kwargs
            def make_trace_kwargs(item, idx, values):
                n_pts = len(values['x'])

                kwargs = {
                        'type': 'scattergl',
                        'mode': 'lines+markers',
                        }
                kwargs.update({
                    'name': data_axis_items[item]['trace_name'](d, idx),
                    'x': values['x'],
                    'y': values['y'],
                    })
                customdata = \
                    data_axis_items[item]['trace_customdata'](d, idx)
                kwargs['customdata'] = [
                        dict(index=i, **customdata)
                        for i in range(n_pts)
                        ]
                return kwargs

            def prepare_axis_data(axis_select_value):
                if axis_select_value is None:
                    return None
                option = axis_select_options[axis_select_value]

                axis_kwargs = {
                        'title': option['label'],
                        }
                if 'kwargs' in option:
                    axis_kwargs.update(**option['kwargs'])

                value = option.get('value', None)
                data_items = option['pass_data_items']

                if value is not None:
                    value = map_func(value, data_items)
                else:
                    if len(data_items) > 1:
                        raise ValueError("implicit value can only be 1-d")
                    # implicit data item will use the index as the vlaue
                    # the ticks will be the data item
                    data_item = data_items[0]
                    value = np.arange(len(d[data_item]), dtype='i')
                if 'tick_value' in option:
                    if len(data_items) > 1:
                        raise ValueError("tick value can only be 1-d")
                    axis_kwargs.update({
                        'tickmode': 'array',
                        'tickvals': value,
                        'ticktext': map_func(option['tick_value'], data_items)
                        })
                return {
                        'value': value,
                        'axis_layout': axis_kwargs,
                        'option': option
                        }

            axis_data = dict()
            for i, k in enumerate(axis_select_items.keys()):
                axis_data[k] = prepare_axis_data(axis_select_values[i])

            # update axis layouts
            fig.update_xaxes(row=1, col=1, **axis_data['x']['axis_layout'])
            fig.update_yaxes(row=1, col=1, **axis_data['y']['axis_layout'])

            # we collate trace on tone_ids
            trace_data_select_items = ['tone_ids', ]
            for indices in itertools.product(
                    *(
                        range(len(d[item]))
                        for item in trace_data_select_items)):
                # match the items with the var name
                for trace_item, idx in zip(trace_data_select_items, indices):
                    self.logger.debug(
                            f'trace_item: {trace_item} idx: {idx}')
                    values = dict()
                    for k in axis_select_items.keys():
                        if axis_data[k] is None:
                            continue
                        map_items = axis_data[k]['option'][
                                'map_data_axis_items']
                        slice_ = [slice(None), ] * len(map_items)
                        if trace_item in map_items:
                            # this is to be sliced
                            slice_[map_items.index(trace_item)] = idx
                        # self.logger.debug(
                        #         f'map_items: {map_items} slice: {slice_}')
                        values[k] = np.ravel(
                                axis_data[k]['value'][tuple(slice_)])
                    t = make_trace_kwargs(trace_item, idx, values)
                    fig.append_trace(t, row=1, col=1)

            return fig, html.Pre(pformat_yaml({
                'axis_data': axis_data,
                'data_objs': d}))

        @app.callback(
                Output(g1.id, 'figure'),
                [
                    Input(g0.id, 'clickData'),
                    Input(select_mode_drp.id, 'value')
                    ],
                [
                    to_dependency('state', i)
                    for i in ctx['select_inputs']
                    ]
                )
        @timeit
        def update_g1(selected_data, select_mode_value, *args):
            if selected_data is None:
                raise dash.exceptions.PreventUpdate
            # get the hovered item
            ti = selected_data['points'][0]['customdata']['tone_id']
            if select_mode_value == 'per_tone':
                i = None
            elif select_mode_value == 'per_item':
                i = selected_data['points'][0]['customdata']['index']
            else:
                i = selected_data['points'][0]['customdata']['index']
            return _update_g1(ti, i, json.dumps(args))

        @functools.lru_cache(maxsize=64)
        def _update_g1(ti, i, args):
            args = json.loads(args)
            # get the data objs
            if i is None:
                dm = slice(None, None)
            else:
                dm = slice(i, i + 1)
            d = ctx['prepare_selected_data_objs'](*args)
            kds = d['data_objs'][dm]
            rbods = d['bods']['reduced_bod'][dm]
            rkds = [
                    None if rbod is None else get_kidsdata(rbod.file_loc)
                    for rbod in rbods
                    ]
            sbods = d['bods']['sweep_bod'][dm]
            skds = [
                    None if sbod is None else get_kidsdata(sbod.file_loc)
                    for sbod in sbods]
            rsbods = d['bods']['reduced_sweep_bod'][dm]
            rskds = [
                    None if rsbod is None else get_kidsdata(rsbod.file_loc)
                    for rsbod in rsbods]
            return make_fig(kds, rkds, skds, rskds, ti)

        def make_fig(kds, rkds, skds, rskds, ti):
            f_unit = u.MHz
            d21_f_unit = u.Hz
            # I-Q, S21-f, D21-f
            fig = make_subplots(
                    2, 2, shared_xaxes=True,
                    vertical_spacing=0.05,
                    specs=[
                        [{"rowspan": 2}, {}],
                        [None, {}]
                        ],
                    fig_layout={
                        'legend': dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="left",
                            x=0
                            ),
                        'title_text': f'tone {ti}',
                        'title_x': 0.5,
                        'height': 600,
                        }
                    )
            fig.update_xaxes(row=1, col=1, title=f'I')
            fig.update_xaxes(row=2, col=2, title=f'f_center ({f_unit})')
            fig.update_yaxes(
                    row=1, col=1, title='Q',
                    scaleanchor='x',
                    scaleratio=1.
                    )
            fig.update_yaxes(row=1, col=2, title='Log |S21|', side='right')
            fig.update_yaxes(
                    row=2, col=2,
                    title=f'|dS21/df| (1/{d21_f_unit})', side='right')
            for kd, rkd, skd, rskd in zip(kds, rkds, skds, rskds):
                if isinstance(kd, MultiSweep):
                    make_sweep_trace(fig, kd, rkd, ti)
                elif isinstance(kd, TimeStream):
                    if skd in kds:
                        # the kd plot will be done as a separate kd data:
                        make_timestream_trace(fig, kd, rkd, None, None, ti)
                    else:
                        make_timestream_trace(fig, kd, rkd, skd, rskd, ti)
                else:
                    continue
            return fig

        def make_sweep_trace(fig, kd, rkd, ti):
            f_unit = u.MHz
            d21_f_unit = u.Hz
            self.logger.debug(f'make sweep trace kd={kd} rkd={rkd} ti={ti}')
            label = make_obs_label(kd.meta)
            trace_kwargs = {
                    'type': 'scattergl',
                    'mode': 'lines+markers',
                    'name': label,
                    'marker_size': 5,
                    }
            frequency = kd.frequency[ti].to_value(f_unit)
            f_center = kd.meta['tone_axis_data']['f_center'][ti].to_value(
                    f_unit)
            S21 = kd.S21.to_value(u.adu)[ti]
            D21 = np.abs(np.gradient(
                S21, kd.frequency.to_value(d21_f_unit)[ti]))
            I = S21.real  # noqa: E741
            Q = S21.imag
            fig.append_trace(
                dict(
                    trace_kwargs, **{
                        'x': I,
                        'y': Q,
                        }),
                row=1, col=1)
            for y, row in [
                    (np.log10(np.abs(S21)), 1),
                    (D21, 2)
                    ]:
                fig.append_trace(
                    dict(
                        trace_kwargs, **{
                            'x': frequency,
                            'y': y,
                            'showlegend': False
                            }),
                    row=row, col=2)
            shapes = fig.layout.shapes
            fig.update_layout(
                    shapes=list(shapes) + [
                        {
                            'type': "line",
                            'x0': f_center,
                            'x1': f_center,
                            'y0': 0,
                            'y1': 1,
                            'xref': 'x2',
                            'yref': "paper",
                            'line_color': '#cccccc'
                            },
                        ])
            # make reduced sweep
            if rkd is None:
                return fig
            mdl = rkd.get_model(ti)
            S21_mdl = mdl(kd.frequency[ti])
            I_mdl = S21_mdl.real
            Q_mdl = S21_mdl.imag
            D21_mdl = np.abs(np.gradient(
                S21_mdl, kd.frequency.to_value(d21_f_unit)[ti]))
            f_res = mdl.fr.quantity.to_value(f_unit)

            trace_kwargs = {
                    'type': 'scattergl',
                    'mode': 'lines',
                    'name': label + ' model',
                    }
            fig.append_trace(
                dict(
                    trace_kwargs, **{
                        'x': I_mdl,
                        'y': Q_mdl,
                        }),
                row=1, col=1)

            for y, row in [
                    (np.log10(np.abs(S21_mdl)), 1),
                    (D21_mdl, 2)
                    ]:
                fig.append_trace(
                    dict(
                        trace_kwargs, **{
                            'x': frequency,
                            'y': y,
                            'showlegend': False
                            }),
                    row=row, col=2)
            shapes = fig.layout.shapes
            fig.update_layout(
                    shapes=list(shapes) + [
                        {
                            'type': "line",
                            'x0': f_res,
                            'x1': f_res,
                            'y0': 0,
                            'y1': 1,
                            'xref': 'x2',
                            'yref': "paper",
                            'line_color': '#ff0000'
                            },
                        ])
            return fig

        def make_timestream_trace(fig, kd, rkd, skd, rskd, ti):
            self.logger.debug(
                    f'make timestream fig kd={kd} rkd={rkd}'
                    f' skd={skd} rskd={rskd} ti={ti}')
            if skd is not None:
                make_sweep_trace(fig, skd, rskd, ti)

            label = make_obs_label(kd.meta)

            I = kd.I.to_value(u.adu)[ti]   # noqa: E741
            Q = kd.Q.to_value(u.adu)[ti]
            # I-Q, S21-f, D21-f

            fig.append_trace(
                    {
                        'x': I,
                        'y': Q,
                        'type': 'scattergl',
                        'mode': 'markers',
                        'name': label,
                        'marker_size': 1,
                        'marker_color': '#ff0033',
                        },
                    row=1, col=1)
            return fig
