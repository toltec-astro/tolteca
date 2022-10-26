#!/usr/bin/env python

from dash_component_template import ComponentTemplate, NullComponent
from dash import html, dcc, Input, Output, State
from dash.exceptions import PreventUpdate
import dash
import dash_bootstrap_components as dbc
import dash_aladin_lite as dal

from dasha.web.templates.common import (
    CollapseContent,
    LabeledChecklist,
    LabeledInput,
)
from dasha.web.templates.utils import PatternMatchingId, make_subplots
import dash_js9 as djs9
import base64
from pathlib import Path

import astropy.units as u
from astropy.table import Table, QTable

# from astropy.coordinates import get_icrs_coordinates
from astroquery.utils import parse_coordinates
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.coordinates.erfa_astrom import erfa_astrom, ErfaAstromInterpolator
from astroplan import FixedTarget
from astroplan import AltitudeConstraint, AtNightConstraint
from astroplan import observability_table

from dataclasses import dataclass, field
import numpy as np
import functools
import bisect
from typing import ItemsView, Union
from io import StringIO
from schema import Or
import jinja2
import json

from tollan.utils.log import get_logger, timeit
from tollan.utils.fmt import pformat_yaml
from tollan.utils import rupdate
from tollan.utils.dataclass_schema import add_schema
from tollan.utils.namespace import Namespace

from ....utils import yaml_load, yaml_dump
from ....simu.utils import SkyBoundingBox
from ....simu import mapping_registry, SimulatorRuntime, ObsParamsConfig
from ....simu.mapping.utils import resolve_sky_coords_frame
from ....utils.common_schema import PhysicalTypeSchema

from .preset import PresetsConfig
from .base import ObsSite, ObsInstru


_j2env = jinja2.Environment()
"""A jinja2 environment for generating clientside callbacks."""


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
        site_name="lmt",
        instru_name=None,
        pointing_catalog_path=None,
        presets_config_path=None,
        js9_config_path=None,
        lmt_tel_surface_rms=76 << u.um,
        toltec_det_noise_factors=None,
        toltec_apt_path=None,
        title_text="Obs Planner",
        subtitle_text=None,
        **kwargs,
    ):
        kwargs.setdefault("fluid", True)
        super().__init__(**kwargs)
        self._raster_model_length_max = (raster_model_length_max,)
        self._lissajous_model_length_max = (lissajous_model_length_max,)
        self._t_exp_max = (t_exp_max,)
        self._site = ObsSite.from_name(site_name)
        self._instru = None if instru_name is None else ObsInstru.from_name(instru_name)
        self._pointing_catalog_path = pointing_catalog_path
        self._presets_config_path = presets_config_path
        self._js9_config_path = js9_config_path
        self._title_text = title_text
        self._subtitle_text = subtitle_text
        if self._instru.name == "toltec":
            # get designed apt for reporting yields
            from .toltec import _get_apt_designed

            apt_designed = _get_apt_designed()
            # propagate some settings to the instru/site
            if toltec_apt_path is None:
                apt = apt_designed
            else:
                apt = Table.read(toltec_apt_path, format="ascii.ecsv")
            self._instru._apt_path = toltec_apt_path
            self._instru._apt_designed = apt_designed
            self._instru._apt = apt
            self._instru._det_noise_factors = toltec_det_noise_factors
            self._instru._revision = subtitle_text

        if self._site.name == "lmt":
            self._site._tel_surface_rms = lmt_tel_surface_rms

    def setup_layout(self, app):

        if self._js9_config_path is not None:
            # this is required to locate the js9 helper
            # in case js9 is used
            djs9.setup_js9(app, config_path=self._js9_config_path)

        container = self
        header, body = container.grid(2, 1)
        # Header
        title_container = header.child(
            html.Div, className="d-flex align-items-baseline"
        )
        title_container.child(html.H2(self._title_text, className="my-2"))
        if self._subtitle_text is not None:
            title_container.child(
                html.P(self._subtitle_text, className="text-secondary mx-2")
            )

        if False:
            app_details = title_container.child(
                CollapseContent(button_text="Details ...", className="ms-4")
            ).content
            app_details.child(html.Pre(pformat_yaml(self.__dict__)))
        header.child(html.Hr(className="mt-0 mb-3"))
        # Body
        controls_panel, results_panel = body.colgrid(1, 2, width_ratios=[1, 3])
        controls_panel.style = {"width": "375px"}
        controls_panel.parent.style = {"flex-wrap": "nowrap"}
        # make the plotting area auto fill the available space.
        results_panel.style = {
            "flex-grow": "1",
            "flex-shrink": "1",
        }
        # Left panel, these are containers for the input controls
        # make two layers of controls, for for planning
        # one for re-producing
        controls_tabs = controls_panel.child(dbc.Tabs, className="nav-pills btn-sm")
        planning_controls_panel = controls_tabs.child(
            dbc.Tab,
            label="Plan",
            labelClassName="py-0 my-2",
            activeLabelClassName="my-2",
        )
        verifying_controls_panel = controls_tabs.child(
            dbc.Tab,
            label="Verify",
            labelClassName="py-0 my-2",
            activeLabelClassName="my-2",
        )
        (
            target_container,
            obssite_container,
            obsinstru_container,
            mapping_container,
        ) = planning_controls_panel.colgrid(4, 1, gy=3)

        target_container = target_container.child(
            self.Card(title_text="Target")
        ).body_container
        target_select_panel = target_container.child(
            ObsPlannerTargetSelect(site=self._site, className="px-0")
        )
        target_info_store = target_select_panel.info_store

        mapping_container = mapping_container.child(
            self.Card(title_text="Mapping")
        ).body_container
        mapping_preset_select = mapping_container.child(
            ObsPlannerMappingPresetsSelect(
                presets_config_path=self._presets_config_path, className="px-0"
            )
        )
        mapping_info_store = mapping_preset_select.info_store

        # mapping execution
        exec_button_container = mapping_container.child(
            html.Div,
            className=("d-flex justify-content-between align-items-start mt-2"),
        )
        exec_details = exec_button_container.child(
            CollapseContent(button_text="Details ...")
        ).content.child(html.Pre, "N/A", className="mb-0", style={"font-size": "80%"})

        # mapping execute button and execute result data store
        exec_button_disabled_color = "primary"
        exec_button_enabled_color = "danger"
        # this is to save some configs to clientside for enable/disable
        # the exec button.
        exec_button_config_store = exec_button_container.child(
            dcc.Store,
            data={
                "disabled_color": exec_button_disabled_color,
                "enabled_color": exec_button_enabled_color,
            },
        )
        exec_button = exec_button_container.child(
            dbc.Button,
            "Execute",
            size="sm",
            color=exec_button_disabled_color,
            disabled=True,
        )
        exec_info_store = exec_button_container.child(dcc.Store)

        # site config panel
        obssite_title = f"Site: {self._site.display_name}"
        obssite_container = obssite_container.child(
            self.Card(title_text=obssite_title)
        ).body_container
        site_info_store = self._site.make_controls(obssite_container).info_store

        if self._instru is not None:
            # instru config panel
            obsinstru_title = f"Instrument: {self._instru.display_name}"
            obsinstru_container = obsinstru_container.child(
                self.Card(title_text=obsinstru_title)
            ).body_container
            instru_info_store = self._instru.make_controls(
                obsinstru_container
            ).info_store
        else:
            instru_info_store = obsinstru_container.child(dcc.Store)

        # verify controls
        (
            verify_input_container,
            verify_details_container,
        ) = verifying_controls_panel.colgrid(2, 1, gy=3)

        verify_input_container = verify_input_container.child(
            self.Card(title_text="Upload Result to Verify")
        ).body_container
        verify_input_upload = verify_input_container.child(
            dcc.Upload,
            ["Drag and Drop or ", html.A("Select a File", className="color-primary")],
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
            },
        )
        upload_content_store = verify_input_container.child(dcc.Store)

        verify_details_container = verify_details_container.child(
            self.Card(title_text="Details")
        ).body_container

        # upload execution
        verify_details = verify_details_container.child(
            html.Div,
        )
        upload_exec_button_container = verify_details_container.child(
            html.Div,
            className=("d-flex justify-content-between align-items-start mt-2"),
        )
        # this is to save some configs to clientside for enable/disable
        # the exec button.
        upload_exec_button_container.child(html.Div)
        upload_exec_button = upload_exec_button_container.child(
            dbc.Button,
            "Execute",
            size="sm",
            color=exec_button_disabled_color,
            disabled=True,
        )

        # Right panel, for plotting
        # mapping_plot_container, dal_container = \
        # dal_container, mapping_plot_container, js9_container = \
        #     plots_panel.colgrid(3, 1, gy=3)
        (
            dal_container,
            results_controls_container,
            mapping_plot_container,
            instru_results_container,
        ) = results_panel.colgrid(4, 1, gy=3)

        if self._instru is not None:
            instru_results_controls = self._instru.make_results_controls(
                results_controls_container, className="px-0 d-flex"
            )
            mapping_plot_controls_container = instru_results_controls
            instru_results = self._instru.make_results_display(
                instru_results_container, className="px-0"
            )
        else:
            mapping_plot_controls_container = results_controls_container

        mapping_plot_container.className = "my-0"
        mapping_plot_collapse = mapping_plot_controls_container.child(
            CollapseContent(
                button_text="Show Trajs in Horizontal Coords...",
                button_props={
                    # 'color': 'primary',
                    "disabled": True,
                    "style": {
                        "text-transform": "none",
                    },
                },
                content=mapping_plot_container.child(
                    dbc.Collapse, is_open=self._instru is None, className="mt-3"
                ),
            ),
        )
        mapping_plot_container = mapping_plot_collapse.content

        mapping_plotter_loading = mapping_plot_container.child(
            dbc.Spinner,
            show_initially=False,
            color="primary",
            spinner_style={"width": "5rem", "height": "5rem"},
        )
        mapping_plotter = mapping_plotter_loading.child(
            ObsPlannerMappingPlotter(
                site=self._site,
            )
        )

        skyview_height = "50vh" if self._instru is None else "40vh"
        skyview = dal_container.child(
            dal.DashAladinLite,
            survey="P/DSS2/color",
            target="M1",
            fov=(10 << u.arcmin).to_value(u.deg),
            style={"width": "100%", "height": skyview_height},
            options={
                "showLayerBox": True,
                "showSimbadPointerControl": True,
                "showReticle": False,
            },
        )

        super().setup_layout(app)

        # connect the target name to the dal view
        # this need to also handle the 
        @app.callback(
            Output(skyview.id, 'target'),
            [
                Input(target_select_panel.target_info_store.id, "data"),
                Input(upload_content_store.id, "data"),
            ]
        )
        def set_dal_target(target_info, upload_data):
            triggered_id = dash.callback_context.triggered_id
            if triggered_id == target_select_panel.id:
                if target_info is None: 
                    return dash.no_update
                return "{} {}".format(target_info['ra_deg'], target_info['dec_deg'])
            if triggered_id == upload_content_store.id:
                return "{} {}".format(upload_data['ra_deg'], upload_data['dec_deg'])
            return dash.no_update

        # connect the target name to the dal view
        # app.clientside_callback(
        #     """
        #     function(target_info) {
        #         if (!target_info) {
        #             return window.dash_clientside.no_update;
        #         }
        #         var ra = target_info.ra_deg;
        #         var dec = target_info.dec_deg;
        #         return ra + ' ' + dec
        #     }
        #     """,
        #     Output(skyview.id, "target"),
        #     Input(target_select_panel.target_info_store.id, "data"),
        # )

        # this toggles the upload exec button enabled only when
        # user has provided valid upload content
        app.clientside_callback(
            """
            function (content_data, btn_cfg) {
                var disabled_color = btn_cfg['disabled_color']
                var enabled_color = btn_cfg['enabled_color']
                if (content_data === null) {
                    return [disabled_color, true]
                }
                return [enabled_color, false]
            }
            """,
            [
                Output(upload_exec_button.id, "color"),
                Output(upload_exec_button.id, "disabled"),
            ],
            [
                Input(upload_content_store.id, "data"),
                State(exec_button_config_store.id, "data"),
            ],
        )

        # this toggles the exec button enabled only when
        # user has provided valid mapping data and target data through
        # the forms
        app.clientside_callback(
            """
            function (mapping_pattern_feedback_type, mapping_data, target_data, btn_cfg) {
                var disabled_color = btn_cfg['disabled_color']
                var enabled_color = btn_cfg['enabled_color']
                if ((mapping_pattern_feedback_type === "invalid") || (mapping_data === null) || (target_data === null)){
                    return [disabled_color, true]
                }
                return [enabled_color, false]
            }
            """,
            [
                Output(exec_button.id, "color"),
                Output(exec_button.id, "disabled"),
            ],
            [
                Input(mapping_preset_select.preset_feedback.id, "type"),
                Input(mapping_info_store.id, "data"),
                Input(target_info_store.id, "data"),
                State(exec_button_config_store.id, "data"),
            ],
        )

        # this callback collects all data from info stores and
        # create the exec config. The exec config is then used
        # to make the traj data, which is consumed by the plotter
        # to create figure. The returned data is stored on
        # clientside and the graphs are updated with the figs

        if self._instru is not None:
            instru_results_loading = instru_results.loading_indicators
            # connect exec info with instru results
            instru_results.make_callbacks(app, exec_info_store_id=exec_info_store.id)
            instru_results_controls.make_callbacks(
                app, exec_info_store_id=exec_info_store.id
            )
        else:
            instru_results_loading = {"outputs": list(), "states": list()}

        @app.callback(
            [
                Output(mapping_preset_select.preset_select.id, "options"),
                Output(mapping_preset_select.preset_select.id, "value"),
                Output(mapping_preset_select.ref_frame_select.id, "options"),
                Output(mapping_preset_select.ref_frame_select.id, "value"),
            ],
            [
                Input(target_info_store.id, "data"),
                Input(site_info_store.id, "data"),
                Input(instru_info_store.id, "data"),
                State(mapping_preset_select.preset_select.id, "options"),
                State(mapping_preset_select.ref_frame_select.id, "options"),
            ],
        )
        def update_mapping_preset_select_options(
            target_data,
            site_data,
            instru_data,
            prev_preset_select_options,
            prev_ref_frame_select_options,
        ):
            target_data = target_data or dict()
            site_data = site_data or dict()
            instru_data = instru_data or dict()
            condition_dict = {
                k: v
                for k, v in {**target_data, **site_data, **instru_data}.items()
                if isinstance(v, bool)
            }
            preset_select_options = mapping_preset_select.make_mapping_preset_options(
                condition_dict=condition_dict
            )
            # print(f"{preset_select_options=}" "{prev_preset_select_options}")
            if preset_select_options == prev_preset_select_options:
                # this is to avoid invalid the mapping pattern when the condition did not change.
                preset_select_options = dash.no_update
                preset_select_value = dash.no_update
            else:
                preset_select_value = ""
            ref_frame_select_options = mapping_preset_select.make_ref_frame_options(
                condition_dict=condition_dict
            )
            ref_frame_select_value = next(
                o for o in ref_frame_select_options if not o.get("disabled", False)
            )["value"]
            if ref_frame_select_options == prev_ref_frame_select_options:
                ref_frame_select_options = dash.no_update
                ref_frame_select_value = dash.no_update
            return (
                preset_select_options,
                preset_select_value,
                ref_frame_select_options,
                ref_frame_select_value,
            )

        @app.callback(
            [
                Output(upload_content_store.id, "data"),
                Output(verify_details.id, "children"),
            ],
            [
                Input(verify_input_upload.id, "contents"),
                State(verify_input_upload.id, "filename"),
                State(verify_input_upload.id, "last_modified"),
            ],
        )
        def parse_upload_content(
            content,
            filename,
            last_modified,
        ):
            """Parse the uploaded input."""
            if content is None:
                return None, html.Pre("N/A")

            error_msgs = []

            def make_error_report():
                return html.Div([dbc.Alert(msg, color="danger") for msg in error_msgs])

            try:
                _, content_string = content.split(",")
                content = base64.b64decode(content_string).decode()
                # print(content)
                tbl = QTable.read(content, format="ascii.ecsv")
            except Exception:
                error_msgs.append("Unable to parse the uploaded file.")
                return None, make_error_report()

            m = tbl.meta.copy()
            ecfg = m["exec_config"]
            # apt needs to be checked if they match
            apt_name = Path(ecfg["instru_data"]["apt_path"]).name
            current_apt_name = self._instru._apt_path.name
            if apt_name != current_apt_name:
                error_msgs.append(
                    f"Mismatch APT version: {apt_name} != {current_apt_name}"
                )
            # updated apt path
            ecfg["instru_data"]["apt_path"] = self._instru._apt_path.parent.joinpath(
                apt_name
            ).as_posix()
            last_modified = Time(last_modified, format="unix").isot
            revision = (
                m["exec_config"]["instru_data"]["revision"].split(":")[-1].strip()
            )
            current_revision = self._instru._revision.split(":")[-1].strip()
            # compare revision:
            if float(current_revision) > float(revision):
                error_msgs.append(
                    f"The result is generated from an earlier revision: {revision} < {current_revision}"
                )
            mapping = json.dumps(ecfg["mapping"], indent=2)
            t_exp = ecfg["obs_params"]["t_exp"]
            target_coord = parse_coordinates(ecfg['mapping']['target'])
            details_content = f"""
ECSV_filename: {filename}
last_modified: {last_modified}
created_at   : {m["created_at"]}
revision     : {revision}
apt_name     : {apt_name}
instru_name  : {m['exec_config']['instru_data']['name']}
polarized    : {m['exec_config']['instru_data']['polarized']}
atm_mdl_name : {m['exec_config']['site_data']['atm_model_name']}
desired_sens : {m['exec_config']['desired_sens']}
mapping      : {mapping}
t_exp        : {t_exp}
depth_rms_coadd_actual:
    a1100    : {tbl['depth_rms_coadd_actual'][0]}
    a1400    : {tbl['depth_rms_coadd_actual'][1]}
    a2000    : {tbl['depth_rms_coadd_actual'][2]}
proj_tot_time: {tbl['proj_total_time'][0]}
            """
# - {name: depth_rms_coadd_actual, unit: mJy, datatype: float64}
# - {name: proj_science_time, unit: h, datatype: float64}
# - {name: proj_science_time_per_night, datatype: int64}
# - {name: proj_n_nights, datatype: int64}
# - {name: proj_science_overhead_time, unit: h, datatype: float64}
# - {name: proj_total_time, unit: h, datatype: float64}
# - {name: proj_overhead_time, unit: h, datatype: float64}
# - {name: proj_overhead_percent, datatype: string}

            return {"exec_config": ecfg, "ra_deg": target_coord.ra.degree, 'dec_deg': target_coord.dec.degree}, html.Div([make_error_report(), html.Pre(details_content)])

        @app.callback(
            [
                Output(exec_info_store.id, "data"),
                Output(mapping_plotter_loading.id, "color"),
            ]
            + instru_results_loading["outputs"],
            [
                Input(exec_button.id, "n_clicks"),
                Input(upload_exec_button.id, "n_clicks"),
                State(upload_content_store.id, "data"),
                State(mapping_info_store.id, "data"),
                State(target_info_store.id, "data"),
                State(site_info_store.id, "data"),
                State(instru_info_store.id, "data"),
                State(mapping_plotter_loading.id, "color"),
            ]
            + instru_results_loading["states"],
            prevent_initial_call=True,
        )
        def make_exec_info_data(
            n_clicks,
            upload_n_clicks,
            upload_data,
            mapping_data,
            target_data,
            site_data,
            instru_data,
            mapping_loading,
            *instru_loading,
        ):
            """Collect data for the planned observation."""
            triggered_id = dash.callback_context.triggered_id
            if triggered_id == exec_button.id:
                if mapping_data is None or target_data is None:
                    return (None, mapping_loading) + instru_loading
                exec_config = ObsPlannerExecConfig.from_data(
                    mapping_data=mapping_data,
                    target_data=target_data,
                    site_data=site_data,
                    instru_data=instru_data,
                )
            elif triggered_id == upload_exec_button.id:
                if upload_data is None:
                    return (None, mapping_loading) + instru_loading
                exec_config = ObsPlannerExecConfig.from_dict(upload_data["exec_config"])

            else:
                return dash.no_update
            # generate the traj. Note that this is cached for performance.
            traj_data = _make_traj_data_cached(exec_config)

            # create figures to be displayed
            mapping_figs = mapping_plotter.make_figures(
                exec_config=exec_config, traj_data=traj_data
            )
            skyview_params = mapping_plotter.make_skyview_params(
                exec_config=exec_config, traj_data=traj_data
            )

            # copy the instru traj data results
            instru_results = None
            if traj_data["instru"] is not None:
                instru_results = traj_data["instru"].get("results", None)

            # send the items to clientside for displaying
            return (
                {
                    "mapping_figs": {
                        name: fig.to_dict() for name, fig in mapping_figs.items()
                    },
                    "skyview_params": skyview_params,
                    "exec_config": exec_config.to_yaml_dict(),
                    # instru specific data,
                    # this is consumed by the instru results
                    # display
                    "instru": instru_results,
                },
                mapping_loading,
            ) + instru_loading

        app.clientside_callback(
            """
            function(exec_info) {
                if (!exec_info) {
                    return true;
                }
                return false;
            }
            """,
            Output(mapping_plot_collapse.button.id, "disabled"),
            [
                Input(exec_info_store.id, "data"),
            ],
        )

        app.clientside_callback(
            """
            function(exec_info) {
                // console.log(exec_info)
                return JSON.stringify(
                    exec_info && exec_info.exec_config,
                    null,
                    2
                    );
            }
            """,
            Output(exec_details.id, "children"),
            [
                Input(exec_info_store.id, "data"),
            ],
        )

        # update the sky map layers
        # with the traj data
        app.clientside_callback(
            """
            function(exec_info) {
                // console.log(exec_info);
                if (!exec_info) {
                    return Array(2).fill(window.dash_clientside.no_update);
                }
                var svp = exec_info.skyview_params;
                var fov = svp.fov || window.dash_clientside.no_update;
                var layers = svp.layers || window.dash_clientside.no_update;
                return [fov, layers]
            }
            """,
            [
                Output(skyview.id, "fov"),
                Output(skyview.id, "layers"),
            ],
            [
                Input(exec_info_store.id, "data"),
            ],
        )

        # connect exec info with plotter
        mapping_plotter.make_mapping_plot_callbacks(
            app, exec_info_store_id=exec_info_store.id
        )

    class Card(ComponentTemplate):
        class Meta:
            component_cls = dbc.Card

        def __init__(self, title_text, **kwargs):
            super().__init__(**kwargs)
            container = self
            container.child(html.H6(title_text, className="card-header"))
            self.body_container = container.child(dbc.CardBody)


def _collect_mapping_config_tooltips():
    # this function generates a dict for tooltip string used for mapping
    # pattern config fields.
    logger = get_logger()
    result = dict()
    for key, info in mapping_registry._register_info.items():
        # get the underlying entry key and value schema
        s = info["dispatcher_schema"]
        tooltips = dict()
        mapping_type = key
        for key_schema, value_schema in s._schema.items():
            item_key = key_schema._schema
            desc = key_schema.description
            tooltips[item_key] = desc
        result[mapping_type] = tooltips
    logger.debug(f"collected tooltips for mapping configs:\n{pformat_yaml(result)}")
    result["common"] = {"t_exp": "Exposure time of the observation."}
    return result


@add_schema
@dataclass
class ObsPlannerExecConfig(object):
    """A class for obs planner execution config."""

    # this class looks like the SimuConfig but it is actually independent
    # from it. The idea is that the obs planner template will define
    # components that fill in this object. And all actual planning functions
    # happens by consuming this object.
    mapping: dict = field(
        metadata={
            "description": "The simulator mapping trajectory config.",
            "schema": mapping_registry.schema,
            "pformat_schema_type": f"<{mapping_registry.name}>",
        }
    )
    obs_params: ObsParamsConfig = field(
        metadata={
            "description": "The dict contains the observation parameters.",
        }
    )
    desired_sens: u.Quantity = field(
        metadata={
            "description": "The desired sensitivity",
            "schema": PhysicalTypeSchema("spectral flux density"),
        }
    )
    # TODO since we now only support LMT/TolTEC, we do not have a
    # good measure of the schema to use for the generic site and instru
    # we just save the raw data dict here.
    site_data: Union[None, dict] = field(
        default=None,
        metadata={
            "description": "The data dict for the site.",
            "schema": Or(dict, None),
        },
    )
    instru_data: Union[None, dict] = field(
        default=None,
        metadata={
            "description": "The data dict for the instrument.",
            "schema": Or(dict, None),
        },
    )

    @classmethod
    def from_data(cls, mapping_data, target_data, site_data=None, instru_data=None):
        mapping_dict = cls._make_mapping_config_dict(mapping_data, target_data)
        obs_params_dict = {
            "f_smp_mapping": "10 Hz",
            "f_smp_probing": "100 Hz",
            "t_exp": mapping_data.pop("t_exp", None),
        }
        return cls.from_dict(
            {
                "mapping": mapping_dict,
                "obs_params": obs_params_dict,
                "site_data": site_data,
                "instru_data": instru_data,
                "desired_sens": mapping_data["desired_sens_mJy"] << u.mJy,
            }
        )

    @staticmethod
    def _make_mapping_config_dict(mapping_data, target_data):
        """Return mapping config dict from mapping and target data store."""
        cfg = dict(**mapping_data)
        cfg["t0"] = f"{target_data['date']} {target_data['time']}"
        cfg["target"] = target_data["name"]
        # for mapping config, we discard the t_exp and desired_sens.
        cfg.pop("t_exp", None)
        cfg.pop("desired_sens_mJy", None)
        return cfg

    def to_yaml_dict(self):
        # this differes from to_dict in that it only have basic
        # serializable types.
        return yaml_load(StringIO(yaml_dump(self.to_dict())))

    def get_simulator_runtime(self):
        """Return a simulator runtime object from this config."""
        # dispatch site/instru to create parts of the simu config dict
        if self.site_data["name"] == "lmt" and self.instru_data["name"] == "toltec":
            simu_cfg = self._make_lmt_toltec_simu_config_dict(
                lmt_data=self.site_data, toltec_data=self.instru_data
            )
        else:
            raise ValueError("unsupported site/instruments for simu.")
        # add to simu cfg the mapping and obs_params
        exec_cfg = self.to_yaml_dict()
        rupdate(
            simu_cfg,
            {
                "simu": {
                    "obs_params": exec_cfg["obs_params"],
                    "mapping": exec_cfg["mapping"],
                }
            },
        )
        return SimulatorRuntime(simu_cfg)

    @staticmethod
    def _make_lmt_toltec_simu_config_dict(lmt_data, toltec_data):
        """Return simu config dict segment for LMT TolTEC."""
        atm_model_name = lmt_data["atm_model_name"]
        tel_surface_rms = f'{lmt_data["tel_surface_rms_um"]} um'
        apt_path = toltec_data["apt_path"]
        det_noise_factor = toltec_data["det_noise_factors"][atm_model_name]
        return {
            "simu": {
                "jobkey": "obs_planner_simu",
                "obs_params": {"f_smp_probing": "122 Hz", "f_smp_mapping": "20 Hz"},
                "instrument": {
                    "name": "toltec",
                    "array_prop_table": apt_path,
                },
                "sources": [
                    {
                        "type": "toltec_power_loading",
                        "atm_model_name": atm_model_name,
                        "atm_cache_dir": None,
                        "tel_surface_rms": tel_surface_rms,
                        "det_noise_factor": det_noise_factor,
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
        logger.info(f"make traj data for {pformat_yaml(self.to_dict())}")
        if self.site_data is None:
            logger.warning("no site data found for generating trajectory.")
            return None
        # get observer from site name
        observer = ObsSite.get_observer(self.site_data["name"])
        logger.debug(f"observer: {observer}")

        mapping_model = self.mapping.get_model(observer=observer)

        t_exp = self.obs_params.t_exp or mapping_model.t_pattern
        dt_smp_s = (1.0 / self.obs_params.f_smp_mapping).to_value(u.s)
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
        if self.mapping.type == "raster":
            bs_coords_overheadmask = mapping_model.evaluate_holdflag(t) > 0
        else:
            bs_coords_overheadmask = np.zeros((len(bs_coords),), dtype=bool)

        # and we can convert the bs_coords to other frames if needed
        altaz_frame = resolve_sky_coords_frame(
            "altaz", observer=observer, time_obs=time_obs
        )
        # this interpolator can speeds things up a bit
        erfa_interp_len = 300.0 << u.s
        with erfa_astrom.set(ErfaAstromInterpolator(erfa_interp_len)):
            # and we can convert the bs_coords to other frames if needed
            bs_coords_icrs = bs_coords.transform_to("icrs")

            bs_coords_altaz = bs_coords.transform_to(altaz_frame)
            # also for the target coord
            target_coord = self.mapping.target_coord
            target_coords_altaz = target_coord.transform_to(altaz_frame)

        bs_sky_bbox_icrs = SkyBoundingBox.from_lonlat(
            bs_coords_icrs.ra,
            bs_coords_icrs.dec,
        )
        bs_traj_data = {
            "t_exp": t_exp,
            "time_obs": time_obs,
            "dlon": dlon,
            "dlat": dlat,
            "ra": bs_coords_icrs.ra,
            "dec": bs_coords_icrs.dec,
            "az": bs_coords_altaz.az,
            "alt": bs_coords_altaz.alt,
            "ra_overhead": bs_coords_icrs.ra[bs_coords_overheadmask],
            "dec_overhead": bs_coords_icrs.dec[bs_coords_overheadmask],
            "az_overhead": bs_coords_altaz.az[bs_coords_overheadmask],
            "alt_overhead": bs_coords_altaz.alt[bs_coords_overheadmask],
            "overheadmask": bs_coords_overheadmask,
            "target_az": target_coords_altaz.az,
            "target_alt": target_coords_altaz.alt,
            "sky_bbox_icrs": bs_sky_bbox_icrs,
        }
        # make instru specific traj data
        if self.instru_data:
            obsinstru = ObsInstru.from_name(self.instru_data["name"])
            instru_traj_data = obsinstru.make_traj_data(self, bs_traj_data)
        else:
            instru_traj_data = None
        return {
            "site": bs_traj_data,
            "instru": instru_traj_data,
        }


@functools.lru_cache(maxsize=8)
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
        kwargs.setdefault("fluid", True)
        super().__init__(**kwargs)
        with open(presets_config_path, "r") as fo:
            self._presets = PresetsConfig.from_dict(yaml_load(fo))
        self._t_exp_max = t_exp_max
        container = self
        self._info_store = container.child(dcc.Store)

    @property
    def info_store(self):
        return self._info_store

    @property
    def presets(self):
        return self._presets

    def make_mapping_preset_options(self, condition_dict=None):
        result = []
        for preset in self.presets:
            option = {"label": preset.label, "value": preset.key}
            if condition_dict is None:
                disabled = False
            else:
                disabled = not set(preset.conditions.items()).issubset(
                    set(condition_dict.items())
                )
            option["disabled"] = disabled
            result.append(option)
        return result

    def make_ref_frame_options(self, condition_dict=None):
        if condition_dict is None:
            polarized = None
        else:
            polarized = condition_dict.get("polarized", False)
        if polarized:
            options = [
                {
                    "label": "AZ/Alt",
                    "value": "altaz",
                    "disabled": True,
                },
                {
                    "label": "RA/Dec",
                    "value": "icrs",
                },
            ]
        else:
            options = [
                {
                    "label": "AZ/Alt",
                    "value": "altaz",
                },
                {
                    "label": "RA/Dec",
                    "value": "icrs",
                },
            ]
        return options

    def setup_layout(self, app):
        container = self
        controls_form = container.child(dbc.Form)
        controls_form_container = controls_form.child(dbc.Row, className="gx-2 gy-2")
        # mapping_preset_select = controls_form_container.child(
        #     LabeledDropdown(
        #         label_text='Mapping Pattern',
        #         # className='w-auto',
        #         size='sm',
        #         placeholder='Select a mapping pattern template ...'
        #         )).dropdown
        desired_sens_container = controls_form_container.child(html.Div)
        desired_sens_group = desired_sens_container.child(
            LabeledInput(
                label_text="Desired Map RMS (mJy/beam)",
                className="w-auto",
                size="sm",
                input_props={
                    # these are the dbc.Input kwargs
                    "type": "number",
                    "min": 0.0,
                    "placeholder": "1.0",
                    # 'style': {
                    #     'flex': '0 1 7rem'
                    #     },
                    "debounce": True,
                    "value": 1.0,
                },
            )
        )
        desired_sens_container.child(
            dbc.Tooltip,
            f"""
The desired map RMS. This determines the number of passes the mapping
pattern defined below being executed. Fractional numbers will be rounded
up, and the reporting table will list the actual map RMS with the given passes
and the overhead involved in carrying out the calculation.
Therefore it is advertised to have integer number of passes,
which can be achieved by either changing the mapping pattern time
(per-pass execution time, via tweaking the t_exp (lissajous-like) or length/space/speed (raster)),
or this value directly after getting the initial execution output for the per-pass map RMS.
            """,
            target=desired_sens_group.input.id,
        )
        mapping_preset_select = self.preset_select = controls_form_container.child(
            dbc.Select,
            size="sm",
            placeholder="Choose a mapping pattern template to edit ...",
        )
        mapping_preset_feedback = self.preset_feedback = controls_form_container.child(
            dbc.FormFeedback
        )
        mapping_preset_select.options = self.make_mapping_preset_options()
        mapping_preset_tooltip = controls_form_container.child(html.Div)
        mapping_preset_form_container = controls_form_container.child(html.Div)
        mapping_preset_data_store = controls_form_container.child(dcc.Store)
        mapping_ref_frame_select = (
            self.ref_frame_select
        ) = controls_form_container.child(
            LabeledChecklist(
                label_text="Mapping Reference Frame",
                className="w-auto",
                size="sm",
                # set to true to allow multiple check
                multi=False,
                checklist_props={
                    "options": self.make_ref_frame_options,
                    "value": "altaz",
                },
            )
        ).checklist
        super().setup_layout(app)
        self.make_mapping_preset_form_callbacks(
            app,
            parent_id=mapping_preset_form_container.id,
            preset_select_id=mapping_preset_select.id,
            datastore_id=mapping_preset_data_store.id,
        )

        @app.callback(
            [
                Output(mapping_preset_select.id, "valid"),
                Output(mapping_preset_select.id, "invalid"),
                Output(mapping_preset_feedback.id, "children"),
                Output(mapping_preset_feedback.id, "type"),
            ],
            [Input(self.info_store.id, "data")],
        )
        def validate_mapping(mapping_data):
            if mapping_data is None:
                return [False, True, "Invalid mapping settings.", "invalid"]
            # we use some fake data to initialize the mapping config object
            target_data = {"name": "180d 0d", "date": "2022-02-02", "time": "06:00:00"}
            # this should never fail since both mapping_data and the
            # target_data should be always correct at this point
            exec_config = ObsPlannerExecConfig.from_data(mapping_data, target_data)
            mapping_config = exec_config.mapping
            t_pattern = mapping_config.get_offset_model().t_pattern
            # for those have t_exp, we report the total exposure time
            if "t_exp" in mapping_data:
                t_exp = mapping_data["t_exp"]
                feedback_content = (
                    f" Total exposure time: {t_exp}. "
                    f"Time to finish one pass: {t_pattern:.2f}."
                )
                return [True, False, feedback_content, "valid"]
            # raster like patterns, we check the exp time to make sure
            # it is ok
            t_exp_max = self._t_exp_max
            if t_pattern > self._t_exp_max:
                feedback_content = (
                    f"The pattern takes {t_pattern:.2f} to finish, which "
                    f"exceeds the required maximum value of {t_exp_max}."
                )
                return [False, True, feedback_content, "invalid"]
            # then we just report the exposure time
            feedback_content = f"Time to finish the pattern: {t_pattern:.2f}."
            return [True, False, feedback_content, "valid"]

        @app.callback(
            [
                Output(desired_sens_group.input.id, "valid"),
                Output(desired_sens_group.input.id, "invalid"),
                Output(desired_sens_group.feedback.id, "children"),
                Output(desired_sens_group.feedback.id, "type"),
            ],
            [Input(desired_sens_group.input.id, "value")],
        )
        def validate_desired_sens(desired_sens_value):
            if desired_sens_value is None:
                return [False, True, "Desired map RMS is required", "invalid"]
            return [True, False, "", "valid"]

        app.clientside_callback(
            """
            function(desired_sens, ref_frame_value, preset_data) {
                if (preset_data === null) {
                    return null
                }
                data = {...preset_data}
                data['ref_frame'] = ref_frame_value
                data['desired_sens_mJy'] = desired_sens
                return data
            }
            """,
            output=Output(self.info_store.id, "data"),
            inputs=[
                Input(desired_sens_group.input.id, "value"),
                Input(mapping_ref_frame_select.id, "value"),
                Input(mapping_preset_data_store.id, "data"),
            ],
        )

        @app.callback(
            Output(mapping_preset_tooltip.id, "children"),
            [Input(mapping_preset_select.id, "value")],
        )
        def make_preset_tooltip(preset_name):
            if not preset_name:
                raise PreventUpdate
            preset = self._presets.get(type="mapping", key=preset_name)
            header_text = preset.description
            if header_text is None:
                type = preset.get_data_item("type").value
                header_text = f"mapping type: {type}"
            content_text = preset.description_long
            if content_text is not None:
                content_text = html.P(content_text)
            return dbc.Popover(
                [dbc.PopoverBody(header_text), dbc.PopoverBody(content_text)],
                target=mapping_preset_select.id,
                trigger="hover",
            )

    def make_mapping_preset_form_callbacks(
        self, app, parent_id, preset_select_id, datastore_id
    ):
        # here we create dynamic layout for given selected preset
        # and define pattern matching layout to collect values
        # we turn off auto_index because we'll use the unique key
        # to identify each field
        pmid = PatternMatchingId(
            container_id=parent_id, preset_name="", key="", auto_index=False
        )

        # the callback to make preset form
        @app.callback(
            Output(parent_id, "children"),
            [
                Input(preset_select_id, "value"),
            ],
            # prevent_initial_call=True
        )
        def make_mapping_preset_layout(preset_name):
            if not preset_name:
                return None
            self.logger.debug(f"generate form for mapping preset {preset_name}")
            preset = self._presets.get(type="mapping", key=preset_name)
            container = NullComponent(id=parent_id)
            form_container = container.child(dbc.Form).child(
                dbc.Row, className="gx-2 gy-2"
            )
            # get default tooltip for mapping preset
            mapping_type = preset.get_data_item("type").value
            tooltips = self._mapping_config_tooltips[mapping_type]
            tooltips.update(self._mapping_config_tooltips["common"])
            for entry in preset.data:
                # make pattern matching id for all fields
                vmin = entry.value_min
                if entry.component_type == "number":
                    if vmin is None:
                        vmin = "-∞"
                    vmax = entry.value_max
                    if vmax is None:
                        vmax = "+∞"
                    placeholder = f"{entry.value} [{vmin}, {vmax}]"
                else:
                    placeholder = f"{entry.value}"
                input_props = {
                    "type": entry.component_type,
                    "id": pmid(preset_name=preset_name, key=entry.key),
                    "value": entry.value,
                    "min": entry.value_min,
                    "max": entry.value_max,
                    "placeholder": placeholder,
                    # 'style': {
                    #     'max-width': '5rem'
                    #     }
                }
                component_kw = {
                    "label_text": entry.label or entry.key,
                    "suffix_text": None if not entry.unit else entry.unit,
                    "size": "sm",
                    "className": "w-100",
                    "input_props": input_props,
                }
                rupdate(component_kw, entry.component_kw)
                # TODO use custom BS5 to set more sensible breakpoints.
                entry_input = (
                    form_container.child(dbc.Col, xxl=12, width=12)
                    .child(LabeledInput(**component_kw))
                    .input
                )
                # tooltip
                tooltip_text = entry.description
                if not tooltip_text:
                    tooltip_text = tooltips[entry.key]
                tooltip_text += f" Preset default: {placeholder}"
                if entry.unit is not None:
                    tooltip_text += f" {entry.unit}"
                form_container.child(dbc.Tooltip, tooltip_text, target=entry_input.id)

            return container.layout

        # the callback to collect form data
        # TODO may be make this as clientside
        @app.callback(
            Output(datastore_id, "data"),
            [
                Input(
                    pmid(container_id=parent_id, preset_name=dash.ALL, key=dash.ALL),
                    "value",
                ),
                State(
                    pmid(container_id=parent_id, preset_name=dash.ALL, key=dash.ALL),
                    "invalid",
                ),
            ],
        )
        def collect_data(input_values, invalid_values):
            logger = get_logger()
            logger.debug(f"input_values: {input_values}")
            logger.debug(f"invalid_values: {invalid_values}")
            if (
                not input_values
                or any((i is None for i in input_values))
                or any(invalid_values)
            ):
                # invalid form
                return None
            # get keys from callback context
            result = dict()
            inputs = dash.callback_context.inputs_list[0]
            # get the preset data and the fields dict
            preset_name = inputs[0]["id"]["preset_name"]
            preset = self._presets.get("mapping", preset_name)

            # here we extract the mapping config dict from preset data
            for input_ in inputs:
                key = input_["id"]["key"]
                item = preset.get_data_item(key)
                value = input_["value"]
                value_text = f'{value} {item.unit or ""}'.strip()
                result[key] = value_text
            return result


class ObsPlannerTargetSelect(ComponentTemplate):
    class Meta:
        component_cls = dbc.Container

    logger = get_logger()

    def __init__(self, site, target_alt_min=20 << u.deg, **kwargs):
        kwargs.setdefault("fluid", True)
        super().__init__(**kwargs)
        self._site = site
        self._target_alt_min = target_alt_min
        container = self
        self._target_info_store = container.child(dcc.Store)
        self._info_store = container.child(dcc.Store)

    @property
    def target_info_store(self):
        return self._target_info_store

    @property
    def info_store(self):
        return self._info_store

    def setup_layout(self, app):
        container = self
        controls_form_container = container.child(dbc.Form).child(
            dbc.Row, className="gx-2 gy-2"
        )
        target_name_container = controls_form_container.child(html.Div)
        target_name_input = target_name_container.child(
            dbc.Input,
            size="sm",
            type="text",
            placeholder=("M1, NGC 1277, 17.7d -2.1d, 10h09m08s +20d19m18s"),
            value="M1",
            # autofocus=True,
            debounce=True,
        )
        target_name_feedback = target_name_container.child(
            dbc.FormFeedback, type="valid"
        )
        target_name_container.child(
            dbc.Tooltip,
            """
Enter source name or coordinates. Names are sent to a name lookup server
for resolving the coordinates. Coordinates should be like "17.7d -2.1d"
or "10h09m08s +20d19m18s".
            """,
            target=target_name_input.id,
        )

        # date picker
        date_picker_container = controls_form_container.child(html.Div)
        date_picker_group = date_picker_container.child(
            LabeledInput(
                label_text="Obs Date",
                className="w-auto",
                size="sm",
                input_props={
                    # these are the dbc.Input kwargs
                    "type": "text",
                    "min": "1990-01-01",
                    "max": "2099-12-31",
                    "placeholder": "yyyy-mm-dd",
                    # 'style': {
                    #     'flex': '0 1 7rem'
                    #     },
                    "debounce": True,
                    "value": "2022-01-01",
                    "pattern": r"[1-2]\d\d\d-[0-1]\d-[0-3]\d",
                },
            )
        )
        date_picker_input = date_picker_group.input
        date_picker_container.child(
            dbc.Tooltip,
            f"""
The date of observation. The visibility report is done by checking if
the target elevation is > {self._target_alt_min} during the night time.
            """,
            target=date_picker_input.id,
        )
        date_picker_feedback = date_picker_group.feedback

        # time picker
        time_picker_container = controls_form_container.child(html.Div)
        time_picker_group = time_picker_container.child(
            LabeledInput(
                label_text="Obs Start Time (UT)",
                className="w-auto",
                size="sm",
                input_props={
                    # these are the dbc.Input kwargs
                    "type": "text",
                    "placeholder": "HH:MM:SS",
                    # 'style': {
                    #     'flex': '0 1 7rem'
                    #     },
                    "debounce": True,
                    "value": "06:00:00",
                    "pattern": r"[0-1]\d:[0-5]\d:[0-5]\d",
                },
            )
        )
        time_picker_input = time_picker_group.input
        time_picker_container.child(
            dbc.Tooltip,
            f"""
The time of observation. The target has to be at
elevation > {self._target_alt_min} during the night.
            """,
            target=time_picker_input.id,
        )
        time_picker_feedback = time_picker_group.feedback
        check_button_container = container.child(
            html.Div, className=("d-flex justify-content-end mt-2")
        )
        check_button = check_button_container.child(
            dbc.Button,
            "Plot Alt. vs Time",
            color="primary",
            size="sm",
        )
        check_result_modal = check_button_container.child(
            dbc.Modal, is_open=False, centered=False
        )

        self._site.make_info_display(container)

        super().setup_layout(app)

        @app.callback(
            [
                Output(self.target_info_store.id, "data"),
                Output(target_name_input.id, "valid"),
                Output(target_name_input.id, "invalid"),
                Output(target_name_feedback.id, "children"),
                Output(target_name_feedback.id, "type"),
            ],
            [
                Input(target_name_input.id, "value"),
            ],
        )
        def resolve_target(name):
            logger = get_logger()
            logger.info(f"resolve target name {name}")
            if not name:
                return (
                    None,
                    False,
                    True,
                    "Enter the name or coordinate of target.",
                    "invalid",
                )
            try:
                coord = parse_coordinates(name)
                coord_text = f"{coord.ra.degree}d " f"{coord.dec.degree}d (J2000)"
                return (
                    {
                        "ra_deg": coord.ra.degree,
                        "dec_deg": coord.dec.degree,
                        "name": name,
                    },
                    True,
                    False,
                    f"Target coordinate resolved: {coord_text}.",
                    "valid",
                )
            except Exception as e:
                logger.debug(f"error parsing {name}", exc_info=True)
                return (None, False, True, f"Unable to resolve target: {e}", "invalid")

        obs_constraints = [
            AltitudeConstraint(self._target_alt_min, 91 * u.deg),
            AtNightConstraint(),
        ]
        observer = self._site.observer

        @app.callback(
            [
                Output(date_picker_input.id, "valid"),
                Output(date_picker_input.id, "invalid"),
                Output(date_picker_feedback.id, "children"),
                Output(date_picker_feedback.id, "type"),
            ],
            [
                Input(date_picker_input.id, "value"),
                Input(self.target_info_store.id, "data"),
            ],
        )
        def validate_date(date_value, data):
            logger = get_logger()
            if data is None:
                return [False, True, "", "invalid"]
            try:
                t0 = Time(date_value)
            except ValueError:
                return (False, True, "Invalid Date. Use yyyy-mm-dd.", "invalid")
            target_coord = SkyCoord(
                ra=data["ra_deg"] << u.deg, dec=data["dec_deg"] << u.deg
            )
            target = FixedTarget(name=data["name"], coord=target_coord)
            time_grid = t0 + (np.arange(0, 24, 0.5) << u.h)
            summary = observability_table(
                obs_constraints, observer, [target], times=time_grid
            )
            logger.info(f"Visibility of targets on day of {t0}\n{summary}")
            ever_observable = summary["ever observable"][0]
            if ever_observable:
                target_uptime = (summary["fraction of time observable"][0] * 24) << u.h
                t_mt = observer.target_meridian_transit_time(
                    t0, target, n_grid_points=48
                )
                feedback_content = (
                    f"Total up-time: {target_uptime:.1f}. "
                    f"Highest at {t_mt.datetime.strftime('UT %H:%M:%S')}."
                )
                return (ever_observable, not ever_observable, feedback_content, "valid")
            feedback_content = "Target is not up at night. Pick another date."
            return (ever_observable, not ever_observable, feedback_content, "invalid")

        @app.callback(
            [
                Output(time_picker_input.id, "valid"),
                Output(time_picker_input.id, "invalid"),
                Output(time_picker_feedback.id, "children"),
                Output(time_picker_feedback.id, "type"),
            ],
            [
                Input(time_picker_input.id, "value"),
                Input(date_picker_input.id, "value"),
                Input(self.target_info_store.id, "data"),
            ],
        )
        def validate_time(time_value, date_value, data):
            if data is None:
                return (False, True, "", "invalid")
            # verify time value only.
            try:
                _ = Time(f"2000-01-01 {time_value}")
            except ValueError:
                return (False, True, "Invalid time. Use HH:MM:SS.", "invalid")
            # verify target availability
            t0 = Time(f"{date_value} {time_value}")
            if not observer.is_night(t0):
                sunrise_time_str = observer.sun_rise_time(t0, which="previous").iso
                sunset_time_str = observer.sun_set_time(t0, which="next").iso
                feedback_content = (
                    f"The time entered is not at night. Sunrise: "
                    f"{sunrise_time_str}. "
                    f"Sunset: {sunset_time_str}"
                )
                return (False, True, feedback_content, "invalid")
            target_coord_icrs = SkyCoord(
                ra=data["ra_deg"] << u.deg, dec=data["dec_deg"] << u.deg
            )
            altaz_frame = observer.altaz(time=t0)
            target_coord_altaz = target_coord_icrs.transform_to(altaz_frame)
            target_az = target_coord_altaz.az
            target_alt = target_coord_altaz.alt
            alt_min = self._target_alt_min
            if target_alt < self._target_alt_min:
                feedback_content = (
                    f"Target at Az = {target_az.degree:.4f}d "
                    f"Alt ={target_alt.degree:.4f}d "
                    f"is too low (< {alt_min}) to observer. "
                    f"Pick another time."
                )
                return (False, True, feedback_content, "invalid")
            feedback_content = (
                f"Target Az = {target_az.degree:.4f}d "
                f"Alt = {target_alt.degree:.4f}d."
            )
            return (True, False, feedback_content, "valid")

        @app.callback(
            [
                Output(check_result_modal.id, "children"),
                Output(check_result_modal.id, "is_open"),
            ],
            [
                Input(check_button.id, "n_clicks"),
                State(date_picker_input.id, "value"),
                State(time_picker_input.id, "value"),
                State(self.target_info_store.id, "data"),
            ],
            prevent_initial_call=True,
        )
        def check_visibility(n_clicks, date_value, time_value, target_data):
            # composing a dummy exec config object so we can call
            # the plotter
            def make_output(content):
                return [dbc.ModalBody(content), True]

            if target_data is None:
                return make_output("Invalid target format.")
            t_exp = 0 << u.min
            try:
                t0 = Time(f"{date_value}")
            except ValueError:
                return make_output("Invalid date format.")
            try:
                t0 = Time(f"{date_value} {time_value}")
                # this will show the target position
                t_exp = 2 << u.min
            except ValueError:
                pass
            plotter = ObsPlannerMappingPlotter(site=self._site)
            target_coord = SkyCoord(
                f"{target_data['ra_deg']} {target_data['dec_deg']}",
                unit=u.deg,
                frame="icrs",
            )
            exec_config = Namespace(
                obs_params=Namespace(t_exp=t_exp),
                mapping=Namespace(t0=t0, target_coord=target_coord),
            )
            fig = plotter._plot_visibility(exec_config)
            return make_output(dcc.Graph(figure=fig))

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
            output=Output(self.info_store.id, "data"),
            inputs=[
                Input(date_picker_input.id, "value"),
                Input(date_picker_input.id, "valid"),
                Input(time_picker_input.id, "value"),
                Input(time_picker_input.id, "valid"),
                Input(self.target_info_store.id, "data"),
                Input(target_name_input.id, "valid"),
            ],
        )


class ObsPlannerMappingPlotter(ComponentTemplate):
    class Meta:
        component_cls = dbc.Container

    logger = get_logger()

    def __init__(self, site, target_alt_min=20 << u.deg, **kwargs):
        kwargs.setdefault("fluid", True)
        super().__init__(**kwargs)
        self._site = site
        self._target_alt_min = target_alt_min

        container = self
        self._graphs = [
            c.child(dcc.Graph, figure=self._make_empty_figure())
            for c in container.colgrid(
                1,
                4,
            ).ravel()
        ]

    def make_mapping_plot_callbacks(self, app, exec_info_store_id):
        # update graph with figures in exec_info_store
        app.clientside_callback(
            _j2env.from_string(
                """
            function(exec_data) {
                if (exec_data === null) {
                    return Array(4).fill({{empty_fig}});
                }
                figs = exec_data['mapping_figs']
                return [
                    figs["visibility"],
                    figs["offset"],
                    figs["altaz"],
                    figs["icrs"],
                    ]
            }
            """
            ).render(empty_fig=json.dumps(self._make_empty_figure())),
            [Output(graph.id, "figure") for graph in self._graphs],
            [Input(exec_info_store_id, "data")],
        )

    @timeit
    def make_figures(self, exec_config, traj_data):

        visibility_fig = self._plot_visibility(exec_config)
        mapping_figs = self._plot_mapping_pattern(exec_config, traj_data)
        figs = dict(**mapping_figs, visibility=visibility_fig)
        return figs

    @timeit
    def make_skyview_params(self, exec_config, traj_data):
        # return the dict that setup the skyview.
        # mapping_config = exec_config.mapping
        bs_traj_data = traj_data["site"]
        instru_traj_data = traj_data["instru"]
        bs_sky_bbox_icrs = SkyBoundingBox.from_lonlat(
            bs_traj_data["ra"],
            bs_traj_data["dec"],
        )
        fov_sky_bbox = bs_sky_bbox_icrs
        if instru_traj_data is not None:
            # figure out instru overlay layout bbox
            instru_sky_bbox_icrs = SkyBoundingBox.from_lonlat(
                instru_traj_data["ra"],
                instru_traj_data["dec"],
            )
            fov_sky_bbox = fov_sky_bbox.pad_with(
                instru_sky_bbox_icrs.width,
                instru_sky_bbox_icrs.height,
            )
        fov = max(fov_sky_bbox.width, fov_sky_bbox.height).to_value(u.deg)
        # set the view fov larger
        fov = fov / 0.618
        layers = list()
        # the mapping pattern layer
        layers.append(
            {
                "type": "overlay",
                "data": [
                    {
                        "data": list(
                            zip(bs_traj_data["ra"].degree, bs_traj_data["dec"].degree)
                        ),
                        "type": "polyline",
                        "color": "red",
                        "lineWidth": 1,
                    }
                ],
                "options": {
                    "name": "Mapping Trajectory",
                    "color": "red",
                    "show": True,
                },
            },
        )
        if instru_traj_data is not None:
            layers.extend(instru_traj_data["skyview_layers"])
        params = dict(
            {
                "target": exec_config.mapping.target,
                "fov": fov,
                "layers": layers,
                "options": {"showLayerBox": True},
            }
        )
        return params

    @staticmethod
    def _make_day_grid(day_start):
        day_grid = day_start + (np.arange(0, 24 * 60 + 1) << u.min)
        return day_grid

    @functools.lru_cache
    def _get_target_coords_altaz_for_day(self, target_coord_str, day_start):
        day_grid = self._make_day_grid(day_start)
        observer = self._site.observer
        return SkyCoord(target_coord_str).transform_to(observer.altaz(time=day_grid))

    fig_layout_default = {
        "xaxis": dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor="black",
            linewidth=4,
            ticks="outside",
        ),
        "yaxis": dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor="black",
            linewidth=4,
            ticks="outside",
        ),
        "plot_bgcolor": "white",
        "margin": dict(
            autoexpand=True,
            l=0,
            r=10,
            b=0,
            t=10,
        ),
        "modebar": {
            "orientation": "v",
        },
    }

    def _make_empty_figure(self):
        return {
            "layout": {
                "xaxis": {"visible": False},
                "yaxis": {"visible": False},
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
        obs_params = exec_config.obs_params
        # highlight the obstime and t_exp
        t_exp = obs_params.t_exp
        if t_exp is None:
            t_exp = mapping_config.get_offset_model().t_pattern

        # make a plot of the uptimes for given mapping_config
        observer = self._site.observer
        t0 = mapping_config.t0
        t1 = t0 + t_exp
        day_start = Time(int(t0.mjd), format="mjd")
        day_grid = self._make_day_grid(day_start)
        # day_end = day_start + (24 << u.h)

        target_coord = mapping_config.target_coord
        target_coords_altaz_for_day = self._get_target_coords_altaz_for_day(
            target_coord.to_string("hmsdms"), day_start
        )
        # target_name = mapping_config.target
        t_sun_rise = observer.sun_rise_time(day_start, which="next")
        t_sun_set = observer.sun_set_time(day_start, which="next")

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
            f"sun_rise: {t_sun_rise.iso}\n"
            f"sun_set: {t_sun_set.iso}\n"
            f"{i_sun_rise=} {i_sun_set=} "
            f"{i_t0=} {i_t1=}"
        )
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
        seg_slices = [slice(None, b0), slice(b0, b1 + 1), slice(b1 + 1, None)]

        trace_kw_daytime = {
            "line": {"color": "orange"},
            "name": "Daytime",
            "legendgroup": "daytime",
        }
        trace_kw_nighttime = {
            "line": {"color": "blue"},
            "name": "Night",
            "legendgroup": "nighttime",
        }

        fig = make_subplots(1, 1, fig_layout=self.fig_layout_default)

        trace_kw = {
            "type": "scattergl",
            "mode": "lines",
        }
        seg_showlegend = [True, True, False]
        # sometimes the first segment is empty so we put the legend on the
        # third one
        if len(day_grid[seg_slices[0]]) == 0:
            seg_showlegend = [False, True, True]
        # make seg_trace kwargs and create trace for each segment
        for s, is_daytime, showlegend in zip(
            seg_slices, seg_is_daytime, seg_showlegend
        ):
            if is_daytime:
                trace_kw_s = dict(trace_kw, **trace_kw_daytime)
            else:
                trace_kw_s = dict(trace_kw, **trace_kw_nighttime)
            # create and add trace
            fig.add_trace(
                dict(
                    trace_kw_s,
                    **{
                        "x": day_grid[s].to_datetime(),
                        "y": target_coords_altaz_for_day[s].alt.degree,
                        "showlegend": showlegend,
                    },
                )
            )
        # obs period
        fig.add_trace(
            dict(
                trace_kw,
                **{
                    "x": day_grid[i_t0:i_t1].to_datetime(),
                    "y": target_coords_altaz_for_day[i_t0:i_t1].alt.degree,
                    "mode": "markers",
                    "marker": {"color": "red", "size": 8},
                    "name": "Target",
                },
            )
        )
        # shaded region for too low elevation
        fig.add_hrect(
            y0=-90,
            y1=self._target_alt_min.to_value(u.deg),
            line_width=1,
            fillcolor="gray",
            opacity=0.2,
        )

        # update some layout
        fig.update_xaxes(title_text="Time [UT]", automargin=True)
        fig.update_yaxes(
            title_text="Target Altitude [deg]", automargin=True, range=[-10, 90]
        )
        fig.update_layout(
            yaxis_autorange=False,
            legend=dict(
                orientation="h",
                x=0.5,
                y=1.02,
                yanchor="bottom",
                xanchor="center",
                bgcolor="#dfdfdf",
            ),
        )
        return fig

    @timeit
    def _plot_mapping_pattern(self, exec_config, traj_data):
        figs = [
            make_subplots(1, 1, fig_layout=self.fig_layout_default) for _ in range(3)
        ]

        trace_kw = {
            "type": "scattergl",
            "mode": "lines",
            "line": {"color": "red"},
            "showlegend": False,
            # 'marker': {
            #     'color': 'black',
            #     'size': 2
            #     }
        }

        bs_traj_data = traj_data["site"]
        instru_traj_data = traj_data["instru"]

        # offset
        fig = offset_fig = figs[0]
        fig.add_trace(
            dict(
                trace_kw,
                **{
                    "x": bs_traj_data["dlon"].to_value(u.arcmin),
                    "y": bs_traj_data["dlat"].to_value(u.arcmin),
                },
            )
        )
        ref_frame_name = exec_config.mapping.ref_frame.name

        if ref_frame_name == "icrs":
            fig.update_xaxes(title_text="Delta-Source RA [arcmin]")
            fig.update_yaxes(title_text="Delta-Source Dec [arcmin]")
        elif ref_frame_name == "altaz":
            fig.update_xaxes(title_text="Delta-Source Az [arcmin]")
            fig.update_yaxes(title_text="Delta-Source Alt [arcmin]")
        else:
            fig.update_xaxes(title_text="Delta-Source [arcmin]")
            fig.update_yaxes(title_text="Delta-Source [arcmin]")
        if instru_traj_data is not None:
            # add instru traj_data
            overlay_traces = instru_traj_data["overlay_traces"]
            for t in overlay_traces["offset"]:
                fig.add_trace(dict(trace_kw, **t))

        # altaz
        fig = altaz_fig = figs[1]
        fig.add_trace(
            dict(
                trace_kw,
                **{
                    "x": bs_traj_data["az"].to_value(u.deg),
                    "y": bs_traj_data["alt"].to_value(u.deg),
                },
            )
        )
        fig.add_trace(
            dict(
                trace_kw,
                **{
                    "x": bs_traj_data["target_az"].to_value(u.deg),
                    "y": bs_traj_data["target_alt"].to_value(u.deg),
                    "line": {"color": "blue"},
                },
            )
        )
        fig.update_xaxes(title_text="Azimuth [deg]")
        fig.update_yaxes(title_text="Altitude [deg]")

        # icrs
        fig = icrs_fig = figs[2]
        fig.add_trace(
            dict(
                trace_kw,
                **{
                    "x": bs_traj_data["ra"].to_value(u.deg),
                    "y": bs_traj_data["dec"].to_value(u.deg),
                },
            )
        )
        fig.update_xaxes(title_text="RA [deg]")
        fig.update_yaxes(title_text="Dec [deg]")

        # all
        for fig in figs:
            fig.update_xaxes(automargin=True, autorange="reversed")
            fig.update_yaxes(automargin=True)
            fig.update_xaxes(
                row=1,
                col=1,
                scaleanchor="y1",
                scaleratio=1.0,
            )
            fig.update_layout(
                legend=dict(
                    orientation="h",
                    x=0.5,
                    y=1.02,
                    yanchor="bottom",
                    xanchor="center",
                    bgcolor="#dfdfdf",
                )
            )
        return {"offset": offset_fig, "altaz": altaz_fig, "icrs": icrs_fig}
