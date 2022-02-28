#!/usr/bin/env python

from dash_component_template import ComponentTemplate, NullComponent
from dash import html, dcc, Input, Output, State
from dash.exceptions import PreventUpdate
import dash
import dash_bootstrap_components as dbc

from dasha.web.templates.common import (
        CollapseContent,
        LabeledChecklist,
        LabeledInput,
        )
from dasha.web.templates.utils import PatternMatchingId, make_subplots
import astropy.units as u
# from astropy.coordinates import get_icrs_coordinates
from astroquery.utils import parse_coordinates
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astroplan import FixedTarget
from astroplan import (AltitudeConstraint, AtNightConstraint)
from astroplan import observability_table

from dataclasses import dataclass, field
import numpy as np
import functools
import bisect
from typing import Union
from io import StringIO
from schema import Or


from tollan.utils.log import get_logger, timeit
from tollan.utils.fmt import pformat_yaml
from tollan.utils import rupdate
from tollan.utils.dataclass_schema import add_schema

from ....utils import yaml_load, yaml_dump
from ....simu.toltec.lmt import lmt_info
from ....simu.toltec.toltec_info import toltec_info
from ....simu import mapping_registry, SimulatorRuntime, ObsParamsConfig
from ....simu.mapping.utils import resolve_sky_coords_frame

from .preset import PresetsConfig


def _add_from_name_factory(cls):
    """A helper decorator to add ``from_name`` factory method to class."""
    cls._subclasses = dict()

    def _init_subclass(cls, name):
        cls._subclasses[name] = cls
        cls.name = name

    cls.__init_subclass__ = classmethod(_init_subclass)

    def from_name(cls, name):
        """Return the site instance for `name`."""
        if name not in cls._subclasses:
            raise ValueError(f"invalid obs site name {name}")
        subcls = cls._subclasses[name]
        return subcls()

    cls.from_name = classmethod(from_name)

    return cls


@_add_from_name_factory
class ObsSite(object):
    """A class provides info of observing site."""

    @classmethod
    def get_observer(cls, name):
        return cls._subclasses[name].observer


@_add_from_name_factory
class ObsInstru(object):
    """A class provides info of observing instrument."""
    pass


class Lmt(ObsSite, name='lmt'):
    """An `ObsSite` for LMT."""

    info = lmt_info
    display_name = info['name_long']
    observer = info['observer']

    def make_site_info_display(self, container):
        info_container = container.child(
                CollapseContent(button_text='Site Info ...')
            ).content
        info_panel = info_container.child(self.InfoPanel(site=self))
        return info_panel

    def make_controls(self, container):
        return container.child(self.ControlPanel(site=self))

    class ControlPanel(ComponentTemplate):
        class Meta:
            component_cls = dbc.Form

        _atm_q_values = [25, 50, 75]
        _atm_q_default = 25
        _tel_surface_rms_default = 76 << u.um

        def __init__(self, site, **kwargs):
            super().__init__(**kwargs)
            self._site = site
            container = self
            self._info_store = container.child(dcc.Store, data={
                'name': site.name
                })

        @property
        def info_store(self):
            return self._info_store

        def setup_layout(self, app):
            container = self.child(dbc.Row, className='gy-2')
            atm_select = container.child(
                    LabeledChecklist(
                        label_text='Atm. Quantile',
                        className='w-auto',
                        size='sm',
                        # set to true to allow multiple check
                        multi=False
                        )).checklist
            atm_select.options = [
                    {
                        'label': f"{q} %",
                        'value': q,
                        }
                    for q in self._atm_q_values]
            atm_select.value = self._atm_q_default

            tel_surface_input = container.child(
                    LabeledInput(
                        label_text='Tel. Surface RMS',
                        className='w-auto',
                        size='sm',
                        input_props={
                            # these are the dbc.Input kwargs
                            'type': 'number',
                            'min': 0,
                            'max': 200,
                            'placeholder': '0-200',
                            'style': {
                                'flex': '0 1 5rem'
                                },
                            },
                        suffix_text='μm'
                        )).input
            tel_surface_input.value = self._tel_surface_rms_default.to_value(
                u.um)
            super().setup_layout(app)

            # collect inputs to store
            app.clientside_callback(
                """
                function(atm_select_value, tel_surface_value, data_init) {
                    data = {...data_init}
                    data['atm_model_name'] = 'am_q' + atm_select_value
                    data['tel_surface_rms'] = tel_surface_value + ' um'
                    return data
                }
                """,
                Output(self.info_store.id, 'data'),
                [
                    Input(atm_select.id, 'value'),
                    Input(tel_surface_input.id, 'value'),
                    State(self.info_store.id, 'data')
                    ]
                )

    class InfoPanel(ComponentTemplate):
        class Meta:
            component_cls = dbc.Container

        def __init__(self, site, **kwargs):
            kwargs.setdefault('fluid', True)
            super().__init__(**kwargs)
            self._site = site

        def setup_layout(self, app):
            container = self
            info = self._site.info
            location = info['location']
            timezone_local = info['timezone_local']
            info_store = container.child(
                dcc.Store, data={
                    'name': self._site.name,
                    'display_name': self._site.display_name,
                    'lon': location.lon.degree,
                    'lat': location.lat.degree,
                    'height_m': location.height.to_value(u.m),
                    'timezone_local': timezone_local.zone,
                    })
            pre_kwargs = {
                'className': 'mb-0',
                'style': {
                    'font-size': '80%'
                    }
                }
            loc_display = container.child(
                html.Pre,
                f'Location: {location.lon.to_string()} '
                f'{location.lat.to_string()}',
                # f' {location.height:.0f}'
                **pre_kwargs)
            loc_display.className += ' mb-0'
            # loc_display = container.child(
            #     LabeledInput(
            #         label_text='Location',
            #         className='w-auto',
            #         size='sm',
            #         input_props={
            #             # these are the dbc.Input kwargs
            #             'type': 'text',
            #             'plaintext': True,
            #             },
            #         )).input
            # loc_display.value = (
            #     f'Location: {location.lon.to_string()} '
            #     f'{location.lat.to_string()}'
            #     )
            time_display = container.child(
                html.Pre, **pre_kwargs)
            timer = container.child(dcc.Interval, interval=500)
            super().setup_layout(app)
            app.clientside_callback(
                """
function(t_interval, info) {
    // console.log(info)
    // current datetime
    var dt = new Date()
    let ut_fmt = new Intl.DateTimeFormat(
        'sv-SE', {
            timeZone: 'UTC',
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
            hour12: false,
            timeZoneName: 'short'})
    result = 'Time: ' + ut_fmt.format(dt)
    let lmt_lt_fmt = new Intl.DateTimeFormat(
        'sv-SE', {
            timeZone: info["timezone_local"],
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
            hour12: false,
            timeZoneName: 'short'})
    result = result + '\\nLocal Time: ' + lmt_lt_fmt.format(dt)
    // LST
    var au = window.dash_clientside.astro_utils
    var lst = au.getLST(dt, info["lon"])
    // convert to hh:mm::ss
    var n = new Date(0,0)
    n.setSeconds(+lst * 60 * 60)
    lmt_lst = n.toTimeString().slice(0, 8)
    result = result + '\\nLST: ' + lmt_lst
    return result
}
""",
                Output(time_display.id, 'children'),
                Input(timer.id, 'n_intervals'),
                Input(info_store.id, 'data'),
                prevent_initial_call=True
                )


class Toltec(ObsInstru, name='toltec'):
    """An `ObsInstru` for TolTEC."""

    info = toltec_info
    display_name = info['name_long']

    def make_controls(self, container):
        return container.child(self.ControlPanel(instru=self))

    class ControlPanel(ComponentTemplate):
        class Meta:
            component_cls = dbc.Form

        def __init__(self, instru, **kwargs):
            super().__init__(**kwargs)
            self._instru = instru
            container = self
            self._info_store = container.child(dcc.Store, data={
                'name': instru.name
                })

        @property
        def info_store(self):
            return self._info_store

        def setup_layout(self, app):
            toltec_info = self._instru.info
            container = self.child(dbc.Row, className='gy-2')
            band_select = container.child(
                    LabeledChecklist(
                        label_text='TolTEC band',
                        className='w-auto',
                        size='sm',
                        # set to true to allow multiple check
                        multi=False,
                        input_props={
                            'style': {
                                'text-transform': 'none'
                                }
                            }
                        )).checklist
            band_select.options = [
                    {
                        'label': str(toltec_info[a]['wl_center']),
                        'value': a,
                        }
                    for a in toltec_info['array_names']
                    ]
            band_select.value = toltec_info['array_names'][0]

            covtype_select = container.child(
                    LabeledChecklist(
                        label_text='Coverage Unit',
                        className='w-auto',
                        size='sm',
                        # set to true to allow multiple check
                        multi=False
                        )).checklist
            covtype_select.options = [
                    {
                        'label': 'mJy/beam',
                        'value': 'depth',
                        },
                    {
                        'label': 's/pixel',
                        'value': 'time',
                        },
                    ]
            covtype_select.value = 'depth'
            super().setup_layout(app)

            # collect inputs to store
            app.clientside_callback(
                """
                function(band_select_value, covtype_select_value, data_init) {
                    data = {...data_init}
                    data['array_name'] = band_select_value
                    data['coverage_map_type'] = covtype_select_value
                    return data
                }
                """,
                Output(self.info_store.id, 'data'),
                [
                    Input(band_select.id, 'value'),
                    Input(covtype_select.id, 'value'),
                    State(self.info_store.id, 'data')
                    ]
                )


class ObsPlanner(ComponentTemplate):
    """An observation Planner."""
    # TODO we should be able to generate the __init__ function

    class Meta:
        component_cls = dbc.Container

    logger = get_logger()

    def __init__(
            self,
            raster_model_length_max=3 << u.deg,
            lissajous_model_length_max=20 << u.deg,
            t_exp_max=1 << u.hour,
            site_name='lmt',
            instru_name='toltec',
            pointing_catalog_path=None,
            presets_config_path=None,
            title_text='Obs Planner',
            **kwargs):
        kwargs.setdefault('fluid', True)
        super().__init__(**kwargs)
        self._raster_model_length_max = raster_model_length_max,
        self._lissajous_model_length_max = lissajous_model_length_max,
        self._t_exp_max = t_exp_max,
        self._site = ObsSite.from_name(site_name)
        self._instru = ObsInstru.from_name(instru_name)
        self._pointing_catalog_path = pointing_catalog_path
        self._presets_config_path = presets_config_path
        self._title_text = title_text

    def setup_layout(self, app):
        container = self
        header, body = container.grid(2, 1)
        # Header
        title_container = header.child(
            html.Div, className='d-flex align-items-baseline')
        title_container.child(html.H1(self._title_text, className='display-3'))
        app_details = title_container.child(
                CollapseContent(button_text='Details ...', className='ms-4')
            ).content
        app_details.child(html.Pre(pformat_yaml(self.__dict__)))
        header.child(html.Hr(className='mt-0 mb-3'))
        # Body
        controls_panel, plots_panel = body.colgrid(1, 2, width_ratios=[1, 3])
        controls_panel.style = {
                    'width': '375px'
                    }
        # make the plotting area auto fill the available space.
        plots_panel.style = {
            'flex-grow': '1'
            }
        # Left panel, these are containers for the input controls
        target_container, mapping_container, \
            obssite_container, obsinstru_container = \
            controls_panel.colgrid(4, 1, gy=3)

        target_container = target_container.child(self.Card(
            title_text='Target')).body_container
        target_info_store = target_container.child(
            ObsPlannerTargetSelect(site=self._site, className='px-0')
            ).target_info_store

        mapping_container = mapping_container.child(self.Card(
            title_text='Mapping')).body_container
        mapping_info_store = mapping_container.child(
            ObsPlannerMappingPresetsSelect(
                presets_config_path=self._presets_config_path,
                className='px-0')
            ).mapping_info_store

        # mapping execution
        exec_button_container = mapping_container.child(
            html.Div,
            className=(
                'd-flex justify-content-between align-items-start mt-2'
                )
            )
        exec_details = exec_button_container.child(
                CollapseContent(button_text='Details ...')
            ).content.child(
                html.Pre,
                'N/A',
                className='mb-0',
                style={
                    'font-size': '80%'
                    })

        # mapping execute button and execute result data store
        exec_button_disabled_color = 'primary'
        exec_button_enabled_color = 'danger'
        # this is to save some configs to clientside for enable/disable
        # the exec button.
        exec_button_config_store = exec_button_container.child(
            dcc.Store, data={
                'disabled_color': exec_button_disabled_color,
                'enabled_color': exec_button_enabled_color
                })
        exec_button = exec_button_container.child(
            dbc.Button,
            "Execute",
            size='sm',
            color=exec_button_disabled_color, disabled=True)
        exec_info_store = exec_button_container.child(dcc.Store)

        # site config panel
        obssite_title = f'Site: {self._site.display_name}'
        obssite_container = obssite_container.child(self.Card(
            title_text=obssite_title)).body_container
        site_info_store = self._site.make_controls(
            obssite_container).info_store

        # instru config panel
        obsinstru_title = f'Instrument: {self._instru.display_name}'
        obsinstru_container = obsinstru_container.child(self.Card(
            title_text=obsinstru_title)).body_container
        instru_info_store = self._instru.make_controls(
            obsinstru_container).info_store

        # Right panel, for plotting
        mapping_plot_container, instru_plot_container = \
            plots_panel.colgrid(2, 1, gy=3)

        mapping_plotter_loading = mapping_plot_container.child(
            dbc.Spinner,
            show_initially=False, color='primary',
            spinner_style={"width": "5rem", "height": "5rem"}
            )
        mapping_plotter = mapping_plotter_loading.child(
            ObsPlannerMappingPlotter(
                site=self._site,
            ))

        super().setup_layout(app)

        # this toggles the exec button enabled only when
        # user has provided valid mapping data and target data through
        # the forms
        app.clientside_callback(
            """
            function (mapping_data, target_data, btn_cfg) {
                var disabled_color = btn_cfg['disabled_color']
                var enabled_color = btn_cfg['enabled_color']
                if ((mapping_data === null) || (target_data === null)){
                    return [disabled_color, true]
                }
                return [enabled_color, false]
            }
            """,
            [
                Output(exec_button.id, 'color'),
                Output(exec_button.id, 'disabled'),
                ],
            [
                Input(mapping_info_store.id, 'data'),
                Input(target_info_store.id, 'data'),
                State(exec_button_config_store.id, 'data')
                ]
            )

        # this callback collects all data from info stores and
        # create the exec config. The exec config is then used
        # to make the traj data, which is consumed by the plotter
        # to create figure. The returned data is stored on
        # clientside and the graphs are updated with the figs

        @app.callback(
            [
                Output(exec_info_store.id, 'data'),
                # this is to trigger the loading state
                Output(mapping_plotter_loading.id, 'color'),
                ],
            [
                Input(exec_button.id, 'n_clicks'),
                State(mapping_info_store.id, 'data'),
                State(target_info_store.id, 'data'),
                State(site_info_store.id, 'data'),
                State(instru_info_store.id, 'data'),
                State(mapping_plotter_loading.id, 'color'),
                ]
            )
        def make_exec_info_data(
                n_clicks, mapping_data, target_data, site_data, instru_data,
                loading):
            """Collect data for the planned observation."""
            if mapping_data is None or target_data is None:
                return (None, loading)
            exec_config = ObsPlannerExecConfig.from_data(
                mapping_data=mapping_data,
                target_data=target_data,
                site_data=site_data,
                instru_data=instru_data)
            # generate the traj. Note that this is cached for performance.
            traj_data = _make_traj_data_cached(exec_config)

            # create figures to be displayed
            mapping_figs = mapping_plotter.make_figures(
                exec_config=exec_config,
                traj_data=traj_data)
            # stores the data to client sidestore
            return (
                {
                    # the traj data needs to be transformed for json
                    # serialization
                    'mapping_traj_data': {
                        'time_obs_mjd': traj_data['time_obs'].mjd,
                        'ra_deg': traj_data['ra'].degree,
                        'dec_deg': traj_data['dec'].degree,
                        'az_deg': traj_data['az'].degree,
                        'alt_deg': traj_data['alt'].degree,
                        'target_az_deg': traj_data['target_az'].degree,
                        'target_alt_deg': traj_data['target_alt'].degree,
                        },
                    'mapping_figs': {
                        name: fig.to_dict()
                        for name, fig in mapping_figs.items()
                        },
                    # need the juggling so all stuff become serializable
                    'exec_config': yaml_load(
                        StringIO(yaml_dump(exec_config.to_dict())))
                    }, loading)

        app.clientside_callback(
            '''
            function(exec_info) {
                return JSON.stringify(exec_info)
            }
            ''',
            Output(exec_details.id, 'children'),
            [
                Input(exec_info_store.id, 'data'),
                ]
            )

        # connect exec info with plotter
        mapping_plotter.make_mapping_plot_callbacks(
            app, exec_info_store_id=exec_info_store.id)

        # @app.callback(
        #     Output(mapping_plot_container.id, 'children'),
        #     [
        #         Input(mapping_plot_button.id, 'n_clicks'),
        #         State(mapping_info_store.id, 'data'),
        #         State(target_info_store.id, 'data'),
        #         ]
        #     )
        # def make_mapping_plots(n_clicks, mapping_data, target_data):
        #     if mapping_data is None or target_data is None:
        #         raise PreventUpdate
        #     mapping_config = _make_mapping_config(
        #         mapping_data=mapping_data, target_data=target_data)
        #     return mapping_plotter.make_figures(mapping_config)

        # @app.callback(
        #     [
        #         Output(mapping_config_details.id, 'children'),
        #         Output(mapping_plot_button.id, 'color'),
        #         Output(mapping_plot_button.id, 'disabled'),
        #         ],
        #     [
        #         Input(mapping_info_store.id, 'data'),
        #         Input(target_info_store.id, 'data'),
        #         ]
        #     )
        # def make_mapping_config_details(mapping_data, target_data):
        #     if mapping_data is None and target_data is None:
        #         return ("Invalid data.", plot_button_disabled_color, True)
        #     if mapping_data is None:
        #         return (
        #             "Invalid mapping data.",
        #             plot_button_disabled_color, True)
        #     if target_data is None:
        #         return (
        #             "Invalid target data.", plot_button_disabled_color, True)
        #     mapping_config = _make_mapping_config(
        #         mapping_data=mapping_data, target_data=target_data)
        #     mapping_model = mapping_config.get_model(
        #         observer=self._site.observer)
        #     result = {
        #         'mapping_model_name': mapping_model.name,
        #         'mapping_model_pattern_time':
        #         f'{mapping_model.t_pattern:.2f}',
        #         'config_dict': {
        #             'mapping': asdict(mapping_config)
        #             }
        #         }
        #     details_text = f'{pformat_yaml(result)}'.strip()
        #     return (details_text, plot_button_enabled_color, False)

    class Card(ComponentTemplate):
        class Meta:
            component_cls = dbc.Card

        def __init__(self, title_text, **kwargs):
            # kwargs.setdefault(
            #     'style', {
            #         # this is for width 1280 col-3
            #         'max-width': '293.75px'
            #         })
            super().__init__(**kwargs)
            container = self
            container.child(html.H6(title_text, className='card-header'))
            self.body_container = container.child(dbc.CardBody)


def _collect_mapping_config_tooltips():
    # this function generates a dict for tooltip string used for mapping
    # pattern config fields.
    logger = get_logger()
    result = dict()
    for key, info in mapping_registry._register_info.items():
        # get the underlying entry key and value schema
        s = info['dispatcher_schema']
        tooltips = dict()
        mapping_type = key
        for key_schema, value_schema in s._schema.items():
            item_key = key_schema._schema
            desc = key_schema.description
            tooltips[item_key] = desc
        result[mapping_type] = tooltips
    logger.debug(
        f"collected tooltips for mapping configs:\n{pformat_yaml(result)}")
    result['common'] = {
        't_exp': 'Exposure time of the observation.'
        }
    return result


@add_schema
@dataclass
class ObsPlannerExecConfig(object):
    """A class for obs planner execution config.
    """
    # this class looks like the SimuConfig but it is actually independent
    # from it. The idea is that the obs planner template will define
    # components that fill in this object. And all actual planning functions
    # happens by consuming this object.
    mapping: dict = field(
        metadata={
            'description': "The simulator mapping trajectory config.",
            'schema': mapping_registry.schema,
            'pformat_schema_type': f'<{mapping_registry.name}>'
            }
        )
    obs_params: ObsParamsConfig = field(
        metadata={
            'description': 'The dict contains the observation parameters.',
            })
    # TODO since we now only support LMT/TolTEC, we do not have a
    # good measure of the schema to use for the generic site and instru
    # we just save the raw data dict here.
    site_data: Union[None, dict] = field(
        default=None,
        metadata={
            'description': "The data dict for the site.",
            'schema': Or(dict, None)
            }
        )
    instru_data: Union[None, dict] = field(
        default=None,
        metadata={
            'description': "The data dict for the instrument.",
            'schema': Or(dict, None)
            }
        )

    @classmethod
    def from_data(
            cls, mapping_data, target_data, site_data=None, instru_data=None):
        mapping_dict = cls._make_mapping_config_dict(mapping_data, target_data)
        obs_params_dict = {
            'f_smp_mapping': '10 Hz',
            'f_smp_probing': '100 Hz',
            't_exp': mapping_data.pop('t_exp', None)
            }
        return cls.from_dict({
            'mapping': mapping_dict,
            'obs_params': obs_params_dict,
            'site_data': site_data,
            'instru_data': instru_data,
            })

    @staticmethod
    def _make_mapping_config_dict(mapping_data, target_data):
        '''Return mapping config dict from mapping and target data store.'''
        cfg = dict(**mapping_data)
        cfg['t0'] = f"{target_data['date']} {target_data['time']}"
        cfg['target'] = target_data['name']
        # for mapping config, we discard the t_exp.
        cfg.pop("t_exp", None)
        return cfg

    def get_simulator_runtime(self):
        """Return a simulator runtime object from this config."""
        site_data = self.site_data
        instru_data = self.instru_data
        # dispatch site/instru to create parts of the simu config dict
        if site_data['name'] == 'lmt' and instru_data['name'] == 'toltec':
            simu_cfg = self._make_lmt_toltec_simu_config_dict(
                lmt_data=site_data, toltec_data=instru_data)
        else:
            raise ValueError("unsupported site/instruments for simu.")
        # add to simu cfg the mapping and obs_params
        exec_cfg = self.to_dict()
        rupdate(simu_cfg, {
            'simu': {
                'obs_params': exec_cfg['obs_params'],
                'mapping': exec_cfg['mapping'],
                }
            })
        return SimulatorRuntime(simu_cfg)

    @staticmethod
    def _make_lmt_toltec_simu_config_dict(lmt_data, toltec_data):
        """Return simu config dict segment for LMT TolTEC."""
        atm_q = lmt_data['atm_q']
        atm_model_name = f'am_q{atm_q}'
        return {
            'simu': {
                'obs_params': {
                    'f_smp_probing': '122 Hz',
                    'f_smp_mapping': '20 Hz'
                    },
                'instrument': {
                    'name': 'toltec',
                    },
                'sources': [
                    {
                        'type': 'toltec_power_loading',
                        'atm_model_name': atm_model_name
                        },
                    ],
                }
            }

    def __hash__(self):
        # allow this object to be hashed so we can cache it to avoid
        # repeated calculation
        exec_cfg_yaml = yaml_dump(self.to_dict())
        return exec_cfg_yaml.__hash__()

    @timeit
    def make_traj_data(self):
        logger = get_logger()
        logger.info(f'make traj data for {pformat_yaml(self.to_dict())}')
        if self.site_data is None:
            logger.warning("no site data found for generating trajectory.")
            return None
        # get observer from site name
        observer = ObsSite.get_observer(self.site_data['name'])
        logger.debug(f"observer: {observer}")

        mapping_model = self.mapping.get_model(observer=observer)

        t_exp = self.obs_params.t_exp or mapping_model.t_pattern
        dt_smp_s = (
            1. / self.obs_params.f_smp_mapping).to_value(u.s)
        t = np.arange(0, t_exp.to_value(u.s) + dt_smp_s, dt_smp_s) << u.s
        n_pts = t.size
        # ensure we at least have 100 points
        if n_pts < 100:
            n_pts = 100
            t = np.linspace(0, t_exp.to_value(u.s), n_pts) << u.s
        logger.debug(f"create {n_pts} sampling points for t_exp={t_exp}")

        # this is to show the mapping in offset unit
        offset_model = mapping_model.offset_mapping_model
        dlon, dlat = offset_model(t)
        time_obs = mapping_model.t0 + t

        # then evaluate the mapping model in mapping.ref_frame
        # to get the bore sight coords
        bs_coords = mapping_model.evaluate_coords(t)

        # and we can convert the bs_coords to other frames if needed
        bs_coords_icrs = bs_coords.transform_to('icrs')

        # and we can convert the bs_coords to other frames if needed
        altaz_frame = resolve_sky_coords_frame(
            'altaz', observer=observer, time_obs=time_obs
            )

        bs_coords_altaz = bs_coords.transform_to(altaz_frame)
        # also for the target coord
        target_coord = self.mapping.target_coord
        target_coords_altaz = target_coord.transform_to(altaz_frame)

        return {
            'time_obs': time_obs,
            'dlon': dlon,
            'dlat': dlat,
            'ra': bs_coords_icrs.ra,
            'dec': bs_coords_icrs.dec,
            'az': bs_coords_altaz.az,
            'alt': bs_coords_altaz.alt,
            'target_az': target_coords_altaz.az,
            'target_alt': target_coords_altaz.alt,
            }


@functools.lru_cache(maxsize=1)
def _make_traj_data_cached(exec_config):
    return exec_config.make_traj_data()


class ObsPlannerMappingPresetsSelect(ComponentTemplate):

    class Meta:
        component_cls = dbc.Container

    logger = get_logger()

    # we inspect the simu mapping patterns to collect default
    # tooltip strings for mapping presets
    _mapping_config_tooltips = _collect_mapping_config_tooltips()

    def __init__(self, presets_config_path=None, t_exp_max=1 << u.h, **kwargs):
        kwargs.setdefault('fluid', True)
        super().__init__(**kwargs)
        with open(presets_config_path, 'r') as fo:
            self._presets = PresetsConfig.from_dict(yaml_load(fo))
        self._t_exp_max = t_exp_max
        container = self
        self._mapping_info_store = container.child(dcc.Store)

    @property
    def mapping_info_store(self):
        return self._mapping_info_store

    @property
    def presets(self):
        return self._presets

    def make_mapping_preset_options(self):
        result = []
        for preset in self.presets:
            result.append({
                'label': preset.label,
                'value': preset.key
                })
        return result

    def setup_layout(self, app):
        container = self
        controls_form = container.child(dbc.Form)
        controls_form_container = controls_form.child(
            dbc.Row, className='gx-2 gy-2')
        # mapping_preset_select = controls_form_container.child(
        #     LabeledDropdown(
        #         label_text='Mapping Pattern',
        #         # className='w-auto',
        #         size='sm',
        #         placeholder='Select a mapping pattern template ...'
        #         )).dropdown
        mapping_preset_select = controls_form_container.child(
            dbc.Select,
            size='sm',
            placeholder='Choose a mapping pattern template to edit ...',
            )
        mapping_preset_feedback = controls_form_container.child(
            dbc.FormFeedback)
        mapping_preset_select.options = self.make_mapping_preset_options()
        mapping_preset_tooltip = controls_form_container.child(
            html.Div)
        mapping_preset_form_container = controls_form_container.child(
            html.Div)
        mapping_preset_data_store = controls_form_container.child(dcc.Store)
        mapping_ref_frame_select = controls_form_container.child(
            LabeledChecklist(
                label_text='Mapping Reference Frame',
                className='w-auto',
                size='sm',
                # set to true to allow multiple check
                multi=False,
                checklist_props={
                    'options': [
                        {
                            'label': "AZ/Alt",
                            'value': "altaz",
                            },
                        {
                            'label': "RA/Dec",
                            'value': "icrs",
                            }
                        ],
                    'value': 'altaz'
                    }
                )).checklist
        super().setup_layout(app)
        self.make_mapping_preset_form_callbacks(
            app, parent_id=mapping_preset_form_container.id,
            preset_select_id=mapping_preset_select.id,
            datastore_id=mapping_preset_data_store.id
            )

        @app.callback(
            [
                Output(mapping_preset_select.id, 'valid'),
                Output(mapping_preset_select.id, 'invalid'),
                Output(mapping_preset_feedback.id, 'children'),
                Output(mapping_preset_feedback.id, 'type'),
                ],
            [
                Input(self.mapping_info_store.id, 'data')
                ]
            )
        def validate_mapping(mapping_data):
            if mapping_data is None:
                return [False, True, 'Invalid mapping settings.', 'invalid']
            # we use some fake data to initialize the mapping config object
            target_data = {
                'name': '180d 0d',
                'date': '2022-02-02',
                'time': '06:00:00'
                }
            # this should never fail since both mapping_data and the
            # target_data should be always correct at this point
            exec_config = ObsPlannerExecConfig.from_data(
                mapping_data, target_data
                )
            mapping_config = exec_config.mapping
            t_pattern = mapping_config.get_offset_model().t_pattern
            # for those have t_exp, we report the total exposure time
            if 't_exp' in mapping_data:
                t_exp = mapping_data['t_exp']
                feedback_content = (
                    f' Total exposure time: {t_exp}. '
                    f'Time to finish one pass: {t_pattern:.2f}.')
                return [True, False, feedback_content, 'valid']
            # raster like patterns, we check the exp time to make sure
            # it is ok
            t_exp_max = self._t_exp_max
            if t_pattern > self._t_exp_max:
                feedback_content = (
                    f'The pattern takes {t_pattern:.2f} to finish, which '
                    f'exceeds the required maximum value of {t_exp_max}.'
                    )
                return [
                    False, True, feedback_content, 'invalid'
                    ]
            # then we just report the exposure time
            feedback_content = (
                f'Time to finish the pattern: {t_pattern:.2f}.')
            return [True, False, feedback_content, 'valid']

        app.clientside_callback(
            """
            function(ref_frame_value, preset_data) {
                if (preset_data === null) {
                    return null
                }
                data = {...preset_data}
                data['ref_frame'] = ref_frame_value
                return data
            }
            """,
            output=Output(self.mapping_info_store.id, 'data'),
            inputs=[
                Input(mapping_ref_frame_select.id, 'value'),
                Input(mapping_preset_data_store.id, 'data')
                ])

        @app.callback(
            Output(mapping_preset_tooltip.id, 'children'),
            [
                Input(mapping_preset_select.id, 'value')
                ]
            )
        def make_preset_tooltip(preset_name):
            if preset_name is None:
                raise PreventUpdate
            preset = self._presets.get(type='mapping', key=preset_name)
            header_text = preset.description
            if header_text is None:
                type = preset.get_data_item('type').value
                header_text = f'mapping type: {type}'
            content_text = preset.description_long
            if content_text is not None:
                content_text = html.P(content_text)
            return dbc.Popover([
                    dbc.PopoverBody(header_text),
                    dbc.PopoverBody(content_text)
                ],
                target=mapping_preset_select.id,
                trigger='hover'
                )

    def make_mapping_preset_form_callbacks(
            self, app,
            parent_id, preset_select_id, datastore_id):
        # here we create dynamic layout for given selected preset
        # and define pattern matching layout to collect values
        # we turn off auto_index because we'll use the unique key
        # to identify each field
        pmid = PatternMatchingId(
            container_id=parent_id, preset_name='', key='', auto_index=False)

        # the callback to make preset form
        @app.callback(
            Output(parent_id, 'children'),
            [
                Input(preset_select_id, 'value'),
                ],
            # prevent_initial_call=True
            )
        def make_mapping_preset_layout(preset_name):
            if preset_name is None:
                return None
            self.logger.debug(
                f"generate form for mapping preset {preset_name}")
            preset = self._presets.get(type='mapping', key=preset_name)
            container = NullComponent(id=parent_id)
            form_container = container.child(dbc.Form).child(
                        dbc.Row, className='gx-2 gy-2')
            # get default tooltip for mapping preset
            mapping_type = preset.get_data_item('type').value
            tooltips = self._mapping_config_tooltips[mapping_type]
            tooltips.update(self._mapping_config_tooltips['common'])
            for entry in preset.data:
                # make pattern matching id for all fields
                vmin = entry.value_min
                if entry.component_type == 'number':
                    if vmin is None:
                        vmin = '-∞'
                    vmax = entry.value_max
                    if vmax is None:
                        vmax = '+∞'
                    placeholder = f'{entry.value} [{vmin}, {vmax}]'
                else:
                    placeholder = f'{entry.value}'
                input_props = {
                    'type': entry.component_type,
                    'id': pmid(preset_name=preset_name, key=entry.key),
                    'value': entry.value,
                    'min': entry.value_min,
                    'max': entry.value_max,
                    'placeholder': placeholder,
                    # 'style': {
                    #     'max-width': '5rem'
                    #     }
                    }
                component_kw = {
                    'label_text': entry.label or entry.key,
                    'suffix_text': None if not entry.unit else entry.unit,
                    'size': 'sm',
                    'className': 'w-100',
                    'input_props': input_props,
                    }
                rupdate(component_kw, entry.component_kw)
                # TODO use custom BS5 to set more sensible breakpoints.
                entry_input = form_container.child(
                    dbc.Col, xxl=12, width=12).child(
                        LabeledInput(**component_kw)).input
                # tooltip
                tooltip_text = entry.description
                if not tooltip_text:
                    tooltip_text = tooltips[entry.key]
                tooltip_text += f' Preset default: {placeholder}'
                if entry.unit is not None:
                    tooltip_text += f' {entry.unit}'
                form_container.child(
                    dbc.Tooltip, tooltip_text,
                    target=entry_input.id)

            return container.layout

        # the callback to collect form data
        # TODO may be make this as clientside
        @app.callback(
            Output(datastore_id, 'data'),
            [
                Input(pmid(
                    container_id=parent_id,
                    preset_name=dash.ALL,
                    key=dash.ALL), 'value'),
                State(pmid(
                    container_id=parent_id,
                    preset_name=dash.ALL,
                    key=dash.ALL), 'invalid'),
                ]
            )
        def collect_data(input_values, invalid_values):
            logger = get_logger()
            logger.debug(f'input_values: {input_values}')
            logger.debug(f'invalid_values: {invalid_values}')
            if not input_values or any((i is None for i in input_values)) \
                    or any(invalid_values):
                # invalid form
                return None
            # get keys from callback context
            result = dict()
            inputs = dash.callback_context.inputs_list[0]
            # get the preset data and the fields dict
            preset_name = inputs[0]['id']['preset_name']
            preset = self._presets.get('mapping', preset_name)

            # here we extract the mapping config dict from preset data
            for input_ in inputs:
                key = input_['id']['key']
                item = preset.get_data_item(key)
                value = input_['value']
                value_text = f'{value} {item.unit or ""}'.strip()
                result[key] = value_text
            return result


class ObsPlannerTargetSelect(ComponentTemplate):

    class Meta:
        component_cls = dbc.Container

    logger = get_logger()

    def __init__(self, site, target_alt_min=20 << u.deg, **kwargs):
        kwargs.setdefault('fluid', True)
        super().__init__(**kwargs)
        self._site = site
        self._target_alt_min = target_alt_min
        container = self
        self._target_info_store = container.child(dcc.Store)

    @property
    def target_info_store(self):
        return self._target_info_store

    def setup_layout(self, app):
        container = self
        controls_form_container = container.child(dbc.Form).child(
                    dbc.Row, className='gx-2 gy-2')
        target_name_container = controls_form_container.child(html.Div)
        target_name_input = target_name_container.child(
            dbc.Input,
            size='sm',
            type='text',
            placeholder=(
                        'M1, NGC 1277, 17.7d -2.1d, 10h09m08s +20d19m18s'),
            value='M1',
            # autofocus=True,
            debounce=True,
            )
        target_name_feedback = target_name_container.child(
            dbc.FormFeedback, type='valid')

        target_resolved_store = controls_form_container.child(dcc.Store)

        # date picker
        date_picker_container = controls_form_container.child(html.Div)
        date_picker_group = date_picker_container.child(
            LabeledInput(
                label_text='Obs Date',
                className='w-auto',
                size='sm',
                input_props={
                    # these are the dbc.Input kwargs
                    'type': 'text',
                    'min': '1990-01-01',
                    'max': '2099-12-31',
                    'placeholder': 'yyyy-mm-dd',
                    # 'style': {
                    #     'flex': '0 1 7rem'
                    #     },
                    'debounce': True,
                    'value': '2022-01-01',
                    'pattern': r'[1-2]\d\d\d-[0-1]\d-[0-3]\d',
                    },
                ))
        date_picker_input = date_picker_group.input
        date_picker_container.child(
            dbc.Tooltip,
            f"""
The date of observation. The visibility report is done by checking if
the target elevation is > {self._target_alt_min} during the night time.
            """,
            target=date_picker_input.id)
        date_picker_feedback = date_picker_group.feedback

        # time picker
        time_picker_container = controls_form_container.child(html.Div)
        time_picker_group = time_picker_container.child(
            LabeledInput(
                label_text='Obs Start Time (UT)',
                className='w-auto',
                size='sm',
                input_props={
                    # these are the dbc.Input kwargs
                    'type': 'text',
                    'placeholder': 'HH:MM:SS',
                    # 'style': {
                    #     'flex': '0 1 7rem'
                    #     },
                    'debounce': True,
                    'value': '06:00:00',
                    'pattern': r'[0-1]\d:[0-5]\d:[0-5]\d',
                    },
                ))
        time_picker_input = time_picker_group.input
        time_picker_container.child(
            dbc.Tooltip,
            f"""
The time of observation. The target has to be at
elevation > {self._target_alt_min} during the night.
            """,
            target=time_picker_input.id)
        time_picker_feedback = time_picker_group.feedback

        self._site.make_site_info_display(container)

        super().setup_layout(app)

        @app.callback(
            [
                Output(target_resolved_store.id, 'data'),
                Output(target_name_input.id, 'valid'),
                Output(target_name_input.id, 'invalid'),
                Output(target_name_feedback.id, 'children'),
                Output(target_name_feedback.id, 'type'),
                ],
            [
                Input(target_name_input.id, 'value'),
             ]
            )
        def resolve_target(name):
            logger = get_logger()
            logger.info(f"resolve target name {name}")
            if not name:
                return (
                    None,
                    False, True,
                    'Enter the name or coordinate of target.',
                    'invalid'
                    )
            try:
                coord = parse_coordinates(name)
                coord_text = (
                    f'{coord.ra.degree}d '
                    f'{coord.dec.degree}d (J2000)'
                    )
                return (
                    {
                        'ra_deg': coord.ra.degree,
                        'dec_deg': coord.dec.degree,
                        'name': name,
                     },
                    True, False,
                    f'Target coordinate resolved: {coord_text}.',
                    'valid'
                    )
            except Exception as e:
                return (
                    None,
                    False, True,
                    f'Unable to resolve target: {e}',
                    'invalid'
                    )

        obs_constraints = [
            AltitudeConstraint(self._target_alt_min, 91*u.deg),
            AtNightConstraint()
            ]
        observer = self._site.observer

        @app.callback(
            [
                Output(date_picker_input.id, 'valid'),
                Output(date_picker_input.id, 'invalid'),
                Output(date_picker_feedback.id, 'children'),
                Output(date_picker_feedback.id, 'type'),
                ],
            [
                Input(date_picker_input.id, 'value'),
                Input(target_resolved_store.id, 'data')
             ]
            )
        def validate_date(date_value, data):
            logger = get_logger()
            if data is None:
                return [False, True, "", 'invalid']
            try:
                t0 = Time(date_value)
            except ValueError:
                return (
                    False, True, 'Invalid Date. Use yyyy-mm-dd.', 'invalid')
            target_coord = SkyCoord(
                ra=data['ra_deg'] << u.deg, dec=data['dec_deg'] << u.deg)
            target = FixedTarget(name=data['name'], coord=target_coord)
            time_grid = t0 + (np.arange(0, 24, 0.5) << u.h)
            summary = observability_table(
                    obs_constraints, observer, [target], times=time_grid)
            logger.info(f'Visibility of targets on day of {t0}\n{summary}')
            ever_observable = summary['ever observable'][0]
            if ever_observable:
                target_uptime = (
                    summary['fraction of time observable'][0] * 24) << u.h
                t_mt = observer.target_meridian_transit_time(
                    t0, target, n_grid_points=48)
                feedback_content = (
                    f"Total up-time: {target_uptime:.1f}. "
                    f"Highest at {t_mt.datetime.strftime('UT %H:%M:%S')}.")
                return (
                    ever_observable,
                    not ever_observable,
                    feedback_content, 'valid'
                    )
            feedback_content = (
                'Target is not up at night. Pick another date.')
            return (
                ever_observable,
                not ever_observable,
                feedback_content,
                'invalid'
                )

        @app.callback(
            [
                Output(time_picker_input.id, 'valid'),
                Output(time_picker_input.id, 'invalid'),
                Output(time_picker_feedback.id, 'children'),
                Output(time_picker_feedback.id, 'type'),
                ],
            [
                Input(time_picker_input.id, 'value'),
                Input(date_picker_input.id, 'value'),
                Input(target_resolved_store.id, 'data'),
                ]
            )
        def validate_time(time_value, date_value, data):
            if data is None:
                return (False, True, '', 'invalid')
            # verify time value only.
            try:
                _ = Time(f'2000-01-01 {time_value}')
            except ValueError:
                return (
                    False, True, 'Invalid time. Use HH:MM:SS.', 'invalid')
            # verify target availability
            t0 = Time(f'{date_value} {time_value}')
            if not observer.is_night(t0):
                sunrise_time_str = observer.sun_rise_time(
                    t0, which="previous").iso
                sunset_time_str = observer.sun_set_time(
                    t0, which="next").iso
                feedback_content = (
                    f'The time entered is not at night. Sunrise: '
                    f'{sunrise_time_str}. '
                    f'Sunset: {sunset_time_str}'
                    )
                return (False, True, feedback_content, 'invalid')
            target_coord_icrs = SkyCoord(
                ra=data['ra_deg'] << u.deg, dec=data['dec_deg'] << u.deg
                )
            altaz_frame = observer.altaz(time=t0)
            target_coord_altaz = target_coord_icrs.transform_to(altaz_frame)
            target_az = target_coord_altaz.az
            target_alt = target_coord_altaz.alt
            alt_min = self._target_alt_min
            if target_alt < self._target_alt_min:
                feedback_content = (
                    f'Target at Az = {target_az.degree:.4f}d '
                    f'Alt ={target_alt.degree:.4f}d '
                    f'is too low (< {alt_min}) to observer. Pick another time.'
                    )
                return (False, True, feedback_content, 'invalid')
            feedback_content = (
                f'Target Az = {target_az.degree:.4f}d '
                f'Alt = {target_alt.degree:.4f}d.'
                )
            return (
                True, False,
                feedback_content,
                'valid'
                )

        app.clientside_callback(
            """
            function(
                date_value, date_valid,
                time_value, time_valid,
                target_data, target_valid) {
                if (date_valid && time_valid && target_valid) {
                    data = {...target_data}
                    data['date'] = date_value
                    data['time'] = time_value
                    return data
                }
                return null
            }
            """,
            output=Output(self.target_info_store.id, 'data'),
            inputs=[
                Input(date_picker_input.id, 'value'),
                Input(date_picker_input.id, 'valid'),
                Input(time_picker_input.id, 'value'),
                Input(time_picker_input.id, 'valid'),
                Input(target_resolved_store.id, 'data'),
                Input(target_name_input.id, 'valid'),
                ])


class ObsPlannerMappingPlotter(ComponentTemplate):

    class Meta:
        component_cls = dbc.Container

    logger = get_logger()

    def __init__(self, site, target_alt_min=20 << u.deg, **kwargs):
        kwargs.setdefault('fluid', True)
        super().__init__(**kwargs)
        self._site = site
        self._target_alt_min = target_alt_min

        container = self
        self._graphs = [
            c.child(dcc.Graph, figure=self._make_empty_figure())
            for c in container.colgrid(2, 2,).ravel()
            ]

    def make_mapping_plot_callbacks(self, app, exec_info_store_id):
        # update graph with figures in exec_info_store
        app.clientside_callback(
            """
            function(exec_data) {
                if (exec_data === null) {
                    return [None, None, None, None]
                }
                figs = exec_data['mapping_figs']
                return [
                    figs["visibility"],
                    figs["offset"],
                    figs["altaz"],
                    figs["icrs"],
                    ]
            }
            """,
            [
                Output(graph.id, 'figure')
                for graph in self._graphs
                ],
            [
                Input(exec_info_store_id, 'data')
                ]
            )

    @timeit
    def _make_figures(self, mapping_config, obs_params, traj_data):

        visibility_fig = self._plot_visibility(mapping_config)

        mapping_figs = self._plot_mapping_pattern(mapping_config)

        # create the container for all the figures
        container = NullComponent(self.id)
        visibility_plot_container, \
            offset_plot_container, \
            altaz_traj_plot_container, \
            icrs_traj_plot_container = container.colgrid(2, 2).ravel()
        visibility_plot_container.child(
            dcc.Graph, figure=visibility_fig)
        offset_plot_container.child(
            dcc.Graph, figure=mapping_figs['offset'])
        altaz_traj_plot_container.child(
            dcc.Graph, figure=mapping_figs['altaz'])
        icrs_traj_plot_container.child(
            dcc.Graph, figure=mapping_figs['icrs'])
        return container.layout

    @timeit
    def make_figures(self, exec_config, traj_data):

        visibility_fig = self._plot_visibility(exec_config)
        mapping_figs = self._plot_mapping_pattern(
            exec_config, traj_data)
        figs = dict(**mapping_figs, visibility=visibility_fig)
        return figs

    @staticmethod
    def _make_day_grid(day_start):
        day_grid = day_start + (np.arange(0, 24 * 60 + 1) << u.min)
        return day_grid

    @functools.lru_cache
    def _get_target_coords_altaz_for_day(self, target_coord_str, day_start):
        day_grid = self._make_day_grid(day_start)
        observer = self._site.observer
        return SkyCoord(target_coord_str).transform_to(
            observer.altaz(time=day_grid))

    fig_layout_default = {
        'xaxis': dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor='black',
            linewidth=4,
            ticks='outside',
            ),
        'yaxis': dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor='black',
            linewidth=4,
            ticks='outside',
            ),
        'plot_bgcolor': 'white',
        'margin': dict(
            autoexpand=True,
            l=0,
            r=10,
            b=0,
            t=10,
            ),
        'modebar': {
            'orientation': 'v',
            },
        }

    def _make_empty_figure(self):
        return {
            "layout": {
                "xaxis": {
                    "visible": False
                    },
                "yaxis": {
                    "visible": False
                    },
                # "annotations": [
                #     {
                #         "text": "No matching data found",
                #         "xref": "paper",
                #         "yref": "paper",
                #         "showarrow": False,
                #         "font": {
                #             "size": 28
                #             }
                #         }
                #     ]
                }
            }

    @timeit
    def _plot_visibility(self, exec_config):
        logger = get_logger()
        mapping_config = exec_config.mapping
        t_pattern = mapping_config.get_offset_model().t_pattern
        obs_params = exec_config.obs_params
        # highlight the obstime and t_exp
        t_exp = obs_params.t_exp or t_pattern

        # make a plot of the uptimes for given mapping_config
        observer = self._site.observer
        t0 = mapping_config.t0
        t1 = t0 + t_exp
        day_start = Time(int(t0.mjd), format='mjd')
        day_grid = self._make_day_grid(day_start)
        # day_end = day_start + (24 << u.h)

        target_coord = mapping_config.target_coord
        target_coords_altaz_for_day = self._get_target_coords_altaz_for_day(
            target_coord.to_string('hmsdms'), day_start)
        # target_name = mapping_config.target
        t_sun_rise = observer.sun_rise_time(day_start, which='next')
        t_sun_set = observer.sun_set_time(day_start, which='next')

        # since day_grid is sorted we can use bisect to locate index
        # in the time grid.
        # this is index right before sunrise
        i_sun_rise = bisect.bisect_left(day_grid, t_sun_rise) - 1
        # this is index right after sunset
        i_sun_set = bisect.bisect_right(day_grid, t_sun_set)
        # index for exposure time period
        i_t0 = bisect.bisect_left(day_grid, t0) - 1
        i_t1 = bisect.bisect_right(day_grid, t1)

        logger.debug(
            f'sun_rise: {t_sun_rise.iso}\n'
            f'sun_set: {t_sun_set.iso}\n'
            f'{i_sun_rise=} {i_sun_set=} '
            f'{i_t0=} {i_t1=}')
        # split the data into three segments at break points b0 and b1
        if i_sun_rise < i_sun_set:
            # middle is daytime
            b0 = i_sun_rise
            b1 = i_sun_set
            seg_is_daytime = [False, True, False]
        else:
            # middle is nighttime
            b0 = i_sun_set
            b1 = i_sun_rise
            seg_is_daytime = [True, False, True]
        seg_slices = [
            slice(None, b0),
            slice(b0, b1 + 1),
            slice(b1 + 1, None)
            ]

        trace_kw_daytime = {
            'line': {
                'color': 'orange'
                },
            'name': 'Daytime',
            'legendgroup': 'daytime'
            }
        trace_kw_nighttime = {
            'line': {
                'color': 'blue'
                },
            'name': 'Night',
            'legendgroup': 'nighttime'
            }

        fig = make_subplots(1, 1, fig_layout=self.fig_layout_default)

        trace_kw = {
                'type': 'scattergl',
                'mode': 'lines',
                }
        seg_showlegend = [True, True, False]
        # sometimes the first segment is empty so we put the legend on the
        # third one
        if len(day_grid[seg_slices[0]]) == 0:
            seg_showlegend = [False, True, True]
        # make seg_trace kwargs and create trace for each segment
        for s, is_daytime, showlegend in zip(
                seg_slices, seg_is_daytime, seg_showlegend):
            if is_daytime:
                trace_kw_s = dict(trace_kw, **trace_kw_daytime)
            else:
                trace_kw_s = dict(trace_kw, **trace_kw_nighttime)
            # create and add trace
            fig.add_trace(
                dict(trace_kw_s, **{
                    'x': day_grid[s].to_datetime(),
                    'y': target_coords_altaz_for_day[s].alt.degree,
                    'showlegend': showlegend
                    }))
        # obs period
        fig.add_trace(
            dict(trace_kw, **{
                'x': day_grid[i_t0:i_t1].to_datetime(),
                'y': target_coords_altaz_for_day[i_t0:i_t1].alt.degree,
                'mode': 'markers',
                'marker': {
                    'color': 'red',
                    'size': 8
                    },
                'name': "Target"
                })
            )
        # shaded region for too low elevation
        fig.add_hrect(
            y0=-90,
            y1=self._target_alt_min.to_value(u.deg),
            line_width=1, fillcolor="gray", opacity=0.2)

        # update some layout
        fig.update_xaxes(
            title_text="Time [UT]",
            automargin=True)
        fig.update_yaxes(
            title_text="Target Altitude [deg]",
            automargin=True,
            range=[-10, 90])
        fig.update_layout(
            yaxis_autorange=False,
            legend=dict(
                orientation="h",
                x=1,
                y=1.02,
                yanchor="bottom",
                xanchor="right",
                bgcolor="#dfdfdf",
                )
            )
        return fig

    @timeit
    def _plot_mapping_pattern(self, exec_config, traj_data):
        figs = [
            make_subplots(1, 1, fig_layout=self.fig_layout_default)
            for _ in range(3)]

        trace_kw = {
                'type': 'scattergl',
                'mode': 'lines',
                'line': {
                    'color': 'red'
                    },
                'showlegend': False,
                # 'marker': {
                #     'color': 'black',
                #     'size': 2
                #     }
                }

        # offset
        fig = offset_fig = figs[0]
        fig.add_trace(dict(trace_kw, **{
            'x': traj_data['dlon'].to_value(u.arcmin),
            'y': traj_data['dlat'].to_value(u.arcmin),
            }))
        ref_frame_name = exec_config.mapping.ref_frame.name

        if ref_frame_name == 'icrs':
            fig.update_xaxes(
                title_text="Delta-Source RA [arcmin]")
            fig.update_yaxes(
                title_text="Delta-Source Dec [arcmin]")
        elif ref_frame_name == 'altaz':
            fig.update_xaxes(
                title_text="Delta-Source Az [arcmin]")
            fig.update_yaxes(
                title_text="Delta-Source Alt [arcmin]")
        else:
            fig.update_xaxes(
                title_text="Delta-Source [arcmin]")
            fig.update_yaxes(
                title_text="Delta-Source [arcmin]")

        # altaz
        fig = altaz_fig = figs[1]
        fig.add_trace(dict(trace_kw, **{
            'x': traj_data['az'].to_value(u.deg),
            'y': traj_data['alt'].to_value(u.deg),
            }))
        fig.add_trace(dict(trace_kw, **{
            'x': traj_data['target_az'].to_value(u.deg),
            'y': traj_data['target_alt'].to_value(u.deg),
            'line': {
                'color': 'purple'
                }
            }))
        fig.update_xaxes(
            title_text="Azimuth [deg]")
        fig.update_yaxes(
            title_text="Altitude [deg]")

        # icrs
        fig = icrs_fig = figs[2]
        fig.add_trace(dict(trace_kw, **{
            'x': traj_data['ra'].to_value(u.deg),
            'y': traj_data['dec'].to_value(u.deg),
            }))
        fig.update_xaxes(
            title_text="RA [deg]")
        fig.update_yaxes(
            title_text="Dec [deg]")

        # all
        for fig in figs:
            fig.update_xaxes(
                automargin=True)
            fig.update_yaxes(
                automargin=True)
            fig.update_xaxes(
                    row=1,
                    col=1,
                    scaleanchor='y1',
                    scaleratio=1.,
                    )
        return {
            'offset': offset_fig,
            'altaz': altaz_fig,
            'icrs': icrs_fig
            }
