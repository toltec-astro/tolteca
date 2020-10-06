#! /usr/bin/env python

from tolteca.cal import ToltecCalib

from dasha.web.templates import ComponentTemplate
from dasha.web.templates.common import LabeledDropdown, LabeledInput
from dasha.web.templates.utils import make_subplots
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input
import dash_daq as daq
import dash
import plotly.express as px

# from tollan.utils.fmt import pformat_yaml
from tollan.utils.log import get_logger
from tollan.utils import odict_from_list

from astroquery.utils import parse_coordinates
import astropy.units as u
from astropy.time import Time
# from astropy.coordinates import SkyCoord

import numpy as np
from pathlib import Path
import functools

from .. import ArrayProjModel, SkyProjModel


def get_calobj_list():
    """Return a list of calobjs to be used as input."""
    logger = get_logger()

    # TODO This shall be updated to connect to the dpdb to retrieve
    # the available property tables.
    # for now we just bundle one such table here.
    datadir = Path(__file__).with_name('data')

    result = []

    @functools.lru_cache(maxsize=None)
    def load_calobj(index_file):
        return ToltecCalib.from_indexfile(i)

    for p in datadir.glob('*'):
        i = p.joinpath('index.yaml')
        if i.exists():
            k = p.name
            logger.debug(f"load {k}: {i}")
            d = {
                'key': k,
                'index_file': i,
                'calobj': load_calobj(i)
                }
            result.append(d)
    return result


class ComponentTemplate(ComponentTemplate):
    _component_cls = dbc.Container
    fluid = True

    def setup_layout(self, app):
        container = self
        header, body = container.grid(2, 1)
        header.className = 'mt-4'
        header.children = [
                html.H2("TolTEC Simulator"),
                html.Hr()
                ]

        proj_container, mapping_container = body.grid(2, 1)

        self._setup_proj_container(app, proj_container)
        self._setup_mapping_container(app, mapping_container)

        super().setup_layout(app)

    def _setup_proj_container(self, app, container):
        header, body = container.grid(2, 1)
        header.className = 'mt-2'
        header.children = [
                html.H4("Array Projection"),
                html.Hr()
                ]

        controls_container, view_container = body.grid(2, 1)
        controls_form = controls_container.child(dbc.Form, inline=True)

        calobjs = odict_from_list(get_calobj_list(), key='key')
        calobj_drp_options = [
                {
                    'label': v['key'],
                    'value': v['key'],
                    }
                for v in calobjs.values()
                ]

        calobj_drp = controls_form.child(LabeledDropdown(
            label_text='array_prop_table',
            dropdown_props={
                'options': calobj_drp_options,
                'value': calobj_drp_options[0]['value']
                },
            className='mr-2',
            )).dropdown

        time_obs_input = controls_form.child(LabeledInput(
            label_text='time_obs',
            input_props={
                'placeholder': "e.g.: 2020-01-01 00:00:00, J2020.0",
                'value': '2020-01-01 00:00:00'
                },
            className='mr-2',
            )).input

        coord_obs_input = controls_form.child(LabeledInput(
            label_text='coord_obs',
            input_props={
                'placeholder': "e.g.: M51, 180d 0d, 1h12m43.2s +1d12m43s",
                'value': '180d 0d'
                },
            className='mr-2',
            )).input

        outline_only_toggle_grp = controls_form.child(
                dbc.FormGroup, className='mx-2')
        outline_only_toggle_grp.child(dbc.Label('Outline only'))
        outline_only_toggle = outline_only_toggle_grp.child(
                daq.BooleanSwitch, on=False)

        array_graph, projected_array_graph, sky_array_graph = map(
                lambda c: c.child(dcc.Graph), view_container.grid(1, 3))

        def make_projected_traces(tbl, m_proj, unit_out):
            array_names = tbl.meta['array_names']
            x_a = tbl['x'].quantity.to(u.cm)
            y_a = tbl['y'].quantity.to(u.cm)
            x_p, y_p = m_proj(tbl['array_name'], x_a, y_a)

            def _make_traces(array_name):
                m = tbl['array_name'] == array_name
                mtbl = tbl[m]
                mtbl.meta = tbl.meta[array_name]
                # per array pos
                mx_a = x_a[m]
                my_a = y_a[m]
                mx_p = x_p[m]
                my_p = y_p[m]
                iv = mtbl.meta['edge_indices']
                vx_a = mx_a[iv]
                vy_a = my_a[iv]
                vx_p = mx_p[iv]
                vy_p = my_p[iv]

                def get_symbol(r):
                    s = ['line-ew', 'line-ns', 'line-ne', 'line-nw']
                    return s[r['ori'] + r['pg'] * 2]

                def make_closed(v):
                    v = v.tolist()
                    v.append(v[0])
                    return v

                kwargs = {
                        'type': 'scattergl',
                        'mode': 'markers',
                        'name': array_name,
                        'customdata': [{'nw': r['nw']} for r in mtbl],
                        'hovertemplate': 'nw: %{customdata.nw}<br>x: %{x:.3f}<br>y: %{y:.3f}',  # noqa: E501
                        'marker': {
                            # 'symbol': [get_symbol(r) for r in mtbl],
                            'symbol': 'circle-open',
                            'color': mtbl['nw'],
                            'colorscale': px.colors.qualitative.Dark24,
                            'cmin': 0,
                            'cmax': 24,
                            }
                        }
                return {
                        'input': dict(kwargs, **{
                            'x': mx_a.to_value(u.cm),
                            'y': my_a.to_value(u.cm),
                            }),
                        'output': dict(kwargs, **{
                            'x': mx_p.to_value(unit_out),
                            'y': my_p.to_value(unit_out),
                            }),
                        'input_outline': dict(kwargs, **{
                            'x': make_closed(vx_a.to_value(u.cm)),
                            'y': make_closed(vy_a.to_value(u.cm)),
                            'mode': 'lines',
                            }),
                        'output_outline': dict(kwargs, **{
                            'x': make_closed(vx_p.to_value(unit_out)),
                            'y': make_closed(vy_p.to_value(unit_out)),
                            'mode': 'lines',
                            }),
                        }
            return {
                    array_name: _make_traces(array_name)
                    for array_name in array_names
                    }

        @functools.lru_cache(maxsize=None)
        def make_array_traces(calobj_drp_value):
            tbl = calobjs[calobj_drp_value]['calobj'].get_array_prop_table()
            m_proj = ArrayProjModel()
            return tbl, make_projected_traces(tbl, m_proj, u.arcmin)

        def make_array_plot_figure(
                n_arrays, title_text, subplot_titles, x_title, y_title):
            fig = make_subplots(
                    n_arrays,
                    1,
                    fig_layout={
                        # 'clickmode': 'event+select',
                        # 'legend': dict(
                        #     orientation="h",
                        #     yanchor="bottom",
                        #     y=1.02,
                        #     xanchor="left",
                        #     x=0
                        #     ),
                        'showlegend': False,
                        'title_text': title_text,
                        'title_x': 0.5,
                        'height': 1200,
                        },
                    subplot_titles=subplot_titles,
                    )
            fig.update_xaxes(
                    row=n_arrays, col=1,
                    title=x_title,
                    )
            fig.update_yaxes(
                    row=n_arrays, col=1,
                    title=y_title,
                    )
            for i in range(n_arrays):
                fig.update_xaxes(
                        row=i + 1,
                        col=1,
                        scaleanchor=f'y{i + 1}',
                        scaleratio=1.,
                        )
            return fig

        @app.callback(
                Output(array_graph.id, 'figure'),
                [
                    Input(calobj_drp.id, 'value'),
                    Input(outline_only_toggle.id, 'on')
                    ]
                )
        def update_array_plot(calobj_drp_value, outline_only):
            tbl, traces = make_array_traces(calobj_drp_value)
            array_names = tbl.meta['array_names']
            fig = make_array_plot_figure(
                    len(traces),
                    f'array_frame',
                    array_names,
                    'x (cm)',
                    'y (cm)'
                    )
            if outline_only:
                key = 'input_outline'
            else:
                key = 'input'
            for i, (array_name, tt) in enumerate(traces.items()):
                fig.append_trace(tt[key], row=i + 1, col=1)
            return fig

        @app.callback(
                Output(projected_array_graph.id, 'figure'),
                [
                    Input(calobj_drp.id, 'value'),
                    Input(outline_only_toggle.id, 'on')
                    ]
                )
        def update_projected_array_plot(calobj_drp_value, outline_only):
            tbl, traces = make_array_traces(calobj_drp_value)
            array_names = tbl.meta['array_names']
            fig = make_array_plot_figure(
                    len(traces),
                    f'toltec_frame',
                    array_names,
                    'Az (arcmin)',
                    'Alt (arcmin)'
                    )
            if outline_only:
                key = 'output_outline'
            else:
                key = 'output'
            for i, (array_name, tt) in enumerate(traces.items()):
                fig.append_trace(tt[key], row=i + 1, col=1)
            return fig

        @app.callback(
                [
                    Output(sky_array_graph.id, 'figure'),
                    Output(time_obs_input.id, 'valid'),
                    Output(time_obs_input.id, 'invalid'),
                    Output(coord_obs_input.id, 'valid'),
                    Output(coord_obs_input.id, 'invalid'),
                    ],
                [
                    Input(calobj_drp.id, 'value'),
                    Input(time_obs_input.id, 'value'),
                    Input(coord_obs_input.id, 'value'),
                    Input(outline_only_toggle.id, 'on'),
                    ]
                )
        def update_sky_array_plot(
                calobj_drp_value,
                time_obs_value, coord_obs_value, outline_only):
            try:
                time_obs = Time(time_obs_value)
            except Exception:
                return (
                    dash.no_update,
                    False, True,
                    dash.no_update, dash.no_update)
            try:
                coord_obs = parse_coordinates(coord_obs_value)
                # coord_obs = SkyCoord(coord_obs_value, frame='icrs')
            except Exception:
                return (
                    dash.no_update,
                    dash.no_update, dash.no_update,
                    False, True)
            tbl = calobjs[calobj_drp_value]['calobj'].get_array_prop_table()
            n_arrays = len(tbl.meta['array_names'])
            m_proj = ArrayProjModel() | SkyProjModel(
                    ref_coord=coord_obs,
                    time_obs=time_obs,
                    evaluate_frame='icrs',
                    )
            traces = make_projected_traces(tbl, m_proj, u.deg)
            fig = make_array_plot_figure(
                    n_arrays,
                    f'sky_frame',
                    tbl.meta['array_names'],
                    'RA (deg)',
                    'Dec (deg)'
                    )
            # update teh scaleratio to match dec
            cos_dec = np.cos(np.deg2rad(coord_obs.dec.degree))
            for i in range(n_arrays):
                fig.update_xaxes(
                        row=i + 1,
                        col=1,
                        scaleanchor=f'y{i + 1}',
                        scaleratio=cos_dec,
                        )
            if outline_only:
                key = 'output_outline'
            else:
                key = 'output'
            for i, (array_name, tt) in enumerate(traces.items()):
                # fig.append_trace(
                #         make_2mass_image_trace(coord_obs_value),
                #         row=i + 1,
                #         col=1,
                #         )
                fig.append_trace(tt[key], row=i + 1, col=1)
            return fig, True, False, True, False

    def _setup_mapping_container(self, app, container):
        header, body = container.grid(2, 1)
        header.className = 'mt-2'
        header.children = [
                html.H4("Mapping Pattern"),
                html.Hr()
                ]

        controls_container, view_container = body.grid(2, 1)
        controls_form = controls_container.child(dbc.Form, inline=True)


@functools.lru_cache(maxsize=None)
def make_2mass_image_trace(coord_obs_value):

    ref_coord = parse_coordinates(coord_obs_value)

    from astropy.wcs import WCS
    from astroquery.skyview import SkyView
    # from astropy.visualization import ZScaleInterval, ImageNormalize
    from astropy.visualization import make_lupton_rgb
    from astropy.wcs.utils import proj_plane_pixel_scales

    print(ref_coord)
    hdulists = SkyView.get_images(
            ref_coord,
            # survey=['WISE 12', 'WISE 4.6', 'WISE 3.4'],
            survey=['2MASS-K', '2MASS-H', '2MASS-J'],
            )
    print(hdulists)
    # scales = [0.3, 0.8, 1.0]
    scales = [1.5, 1.0, 1.0]  # white balance

    def _bkg_subtracted_data(hdu, scale=1.):
        ni, nj = hdu.data.shape
        mask = np.ones_like(hdu.data, dtype=bool)
        frac = 5
        mask[
                ni // frac:(frac - 1) * ni // 4,
                nj // frac:(frac - 1) * nj // 4] = False
        data_bkg = hdu.data[mask]
        bkg = 3 * np.nanmedian(data_bkg) - 2 * np.nanmean(data_bkg)
        return (hdu.data - bkg) * scale

    image = make_lupton_rgb(
            *(_bkg_subtracted_data(
                hl[0], scale=scale)
                for hl, scale in zip(hdulists, scales)),
            Q=10, stretch=50)
    w = WCS(hdulists[0][0].header)
    dx, dy = proj_plane_pixel_scales(w)
    x0, y0 = w.all_pix2world(image.shape[0], 0, 0)
    cos_dec = np.cos(np.deg2rad(ref_coord.dec.degree))
    trace = {
            'type': 'image',
            'name': f"{ref_coord.to_string('hmsdms')}",
            'z': image,
            'x0': x0.item(),
            'y0': y0.item(),
            'dx': np.abs(dx) / cos_dec,
            'dy': np.abs(dy),
            }
    print(trace)
    return trace
