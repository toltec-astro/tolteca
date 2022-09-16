#!/usr/bin/env python

from html.parser import HTMLParser
from dash_component_template import ComponentTemplate
from dash import html, dcc, Input, Output, State
import dash
import dash_bootstrap_components as dbc

import yaml
import numpy as np
import functools
import astropy.units as u

from tollan.utils.log import get_logger
from tollan.utils.fmt import pformat_yaml

from dasha.web.templates.common import (
    LabeledChecklist,
    LabeledInput,
)

from ..common.misc import HeaderWithToltecLogo
from ..common.simple_basic_obs_select import KidsDataSelect
from ....simu import instrument_registry
from ....simu.toltec.toltec_info import toltec_info
from ....simu.toltec.models import ToltecArrayPowerLoadingModel


@functools.lru_cache(maxsize=None)
def _get_simu_apt(apt_filepath=None):
    instru = instrument_registry.schema.validate(
        {"name": "toltec", "array_prop_table": apt_filepath}, create_instance=True
    )
    simulator = instru.simulator
    return simulator.array_prop_table


class ToltecSensitivityCalculator(ComponentTemplate):
    """A interactive TolTEC sensitivity calculator."""

    class Meta:
        component_cls = dbc.Container

    logger = get_logger()

    def __init__(
        self,
        lmt_tel_surface_rms=None,
        toltec_det_noise_factors=None,
        toltec_apt_path=None,
        title_text="TolTEC Sensitivity Calculator",
        subtitle_text="",
        **kwargs,
    ):
        kwargs.setdefault("fluid", True)
        super().__init__(**kwargs)
        self._lmt_tel_surface_rms = lmt_tel_surface_rms
        self._toltec_det_noise_factors = toltec_det_noise_factors
        self._title_text = title_text
        self._subtitle_text = subtitle_text

        apt_designed = _get_simu_apt()
        if toltec_apt_path is None:
            apt = apt_designed
        else:
            apt = _get_simu_apt(toltec_apt_path.as_posix())
        self._apt_designed = apt_designed
        self._apt = apt
        self._atm_q_values = [25, 50, 75]
        self._atm_q_default = 50

        aplms = self._aplms = dict()
        for amq in self._atm_q_values:
            atm_model_name = f"am_q{amq}"
            if self._toltec_det_noise_factors is None:
                det_noise_factor = None
            else:
                det_noise_factor = self._toltec_det_noise_factors[atm_model_name]
            for array_name in toltec_info["array_names"]:
                aplms[(amq, array_name)] = ToltecArrayPowerLoadingModel(
                    array_name=array_name,
                    atm_model_name=atm_model_name,
                    tel_surface_rms=self._lmt_tel_surface_rms,
                    det_noise_factor=det_noise_factor,
                )

    @classmethod
    def _calc_mapping_speed(cls, aplm, alt, n_dets, polarized):
        mapping_speed = aplm.get_mapping_speed(alt=alt, n_dets=n_dets)
        return {
            "mapping_speed_deg2_per_h_per_mJy2": mapping_speed.to_value(
                u.deg**2 / u.h / u.mJy**2
            )
        }

    def setup_layout(self, app):

        container = self
        header_section, hr_container, body = container.grid(3, 1)
        hr_container.child(html.Hr())
        header_container = header_section.child(
            HeaderWithToltecLogo(logo_colwidth=4)
        ).header_container
        title_container, controls_container = header_container.grid(2, 1)

        title_container = title_container.child(
            html.Div, className="d-flex align-items-baseline"
        )
        title_container.child(html.H2(self._title_text, className="my-2"))
        if self._subtitle_text:
            title_container.child(
                html.P(self._subtitle_text, className="text-secondary mx-2")
            )
        inputs = self._make_inputs(controls_container)
        panels_container, readme_container = containers = body.colgrid(1, 2)
        # for responsive layout
        for c in containers:
            c.width = 12 
            c.xl = 6 
        panels = panels_container.colgrid(1, 3, gy=2)
        for w, s, p in zip([12, 12, 12], [12, 6, 6], panels):
            p.width = w
            p.sm = s
        mapping_speed_store = panels[0].child(dcc.Store)
        (
            mapping_speed_table_values,
            mapping_speed_table_loading_indicator,
        ) = self._make_mapping_speed_panel(panels[0])
        rms_inputs, rms_values, rms_loading_indicator = self._make_result_panel(
            panels[1],
            input_keys=["x", "y", "t"],
            output_key="d",
            title="RMS Given Map Size and Time",
        )
        time_inputs, time_values, time_loading_indicator = self._make_result_panel(
            panels[2],
            input_keys=["x", "y", "d"],
            output_key="t",
            title="Time Given Map Size and RMS",
        )
        self._make_readme_panel(readme_container)
        super().setup_layout(app)

        @app.callback(
            [Output(mapping_speed_store.id, "data")]
            + mapping_speed_table_loading_indicator["outputs"]
            + rms_loading_indicator['outputs']
            + time_loading_indicator['outputs']
            ,
            inputs={k: Input(v.id, "value") for k, v in inputs.items()},
        )
        def calc_mapping_speed(**kwargs):
            logger = get_logger()
            logger.debug(f"input_values: {kwargs}")
            if any(v is None for v in kwargs.values()):
                return None
            polarized = kwargs["stokes_select"] == "QU"
            data = {
                "polarized": polarized 
            }
            for array_name in toltec_info["array_names"]:
                aplm = self._aplms[kwargs["atm_select"], array_name]
                n_dets = (self._apt["array_name"] == array_name).sum()
                data[array_name] = self._calc_mapping_speed(
                    aplm,
                    alt=kwargs["alt_input"] << u.deg,
                    n_dets=n_dets,
                    polarized=polarized,
                )
            logger.info(f"mapping speed data:\n{pformat_yaml(data)}")
            return (
                [data]
                + mapping_speed_table_loading_indicator["values"]
                + rms_loading_indicator['values']
                + time_loading_indicator['values']
                )   

        app.clientside_callback(
            """
            function(data) {
                if (data === null) {
                    return Array(3).fill("...");
                }

                return [
                    data["a1100"]["mapping_speed_deg2_per_h_per_mJy2"].toFixed(),
                    data["a1400"]["mapping_speed_deg2_per_h_per_mJy2"].toFixed(),
                    data["a2000"]["mapping_speed_deg2_per_h_per_mJy2"].toFixed(),
                ]
            }
            """,
            output=[
                Output(
                    mapping_speed_table_values["mapping_speed", array_name].id,
                    "children",
                )
                for array_name in toltec_info["array_names"]
            ],
            inputs=[
                Input(mapping_speed_store.id, "data"),
            ],
        )

        app.clientside_callback(
            """
            function(data, x, y, t) {
                if (data === null || x === null || y === null || t === null) {
                    return Array(3).fill("...");
                }
                const get_rms = function (e) { 
                    var m = e["mapping_speed_deg2_per_h_per_mJy2"];
                    var s = x * y / 3600.;
                    var h = t / 60.
                    var d = Math.sqrt(s / m / h);
                    return d.toFixed(3) + ' mJy';
                };
                return [
                    get_rms(data["a1100"]),
                    get_rms(data["a1400"]),
                    get_rms(data["a2000"])
                ];
            }
            """,
            output=[
                Output(
                    rms_values[array_name].id,
                    "children",
                )
                for array_name in toltec_info["array_names"]
            ],
            inputs=[
                Input(mapping_speed_store.id, "data"),
                Input(rms_inputs['x'].id, "value"),
                Input(rms_inputs['y'].id, "value"),
                Input(rms_inputs['t'].id, "value"),
            ],
        )

        app.clientside_callback(
            """
            function(data, x, y, d) {
                if (data === null || x === null || y === null || d === null) {
                    return Array(3).fill("...");
                }
                const get_time = function (e) { 
                    var m = e["mapping_speed_deg2_per_h_per_mJy2"];
                    var s = x * y / 3600.;
                    var t = s / m / d / d * 60;
                    return t.toFixed(3) + ' min';
                };
                return [
                    get_time(data["a1100"]),
                    get_time(data["a1400"]),
                    get_time(data["a2000"])
                ];
            }
            """,
            output=[
                Output(
                    time_values[array_name].id,
                    "children",
                )
                for array_name in toltec_info["array_names"]
            ],
            inputs=[
                Input(mapping_speed_store.id, "data"),
                Input(time_inputs['x'].id, "value"),
                Input(time_inputs['y'].id, "value"),
                Input(time_inputs['d'].id, "value"),
            ],
        )
        # add polarized indicator to the title
        app.clientside_callback(
            """
            function(data) {
                if (data === null) {
                    return Array(2).fill("...");
                }
                if (data['polarized']) {
                    return Array(2).fill("(Polarized Emission)")
                }
                return Array(2).fill("(Total Intensity)")
            }
            """,
            output=[
                Output(
                    v["stokes_params"].id,
                    "children",
                )
                for v in [rms_values, time_values]
            ],
            inputs=[
                Input(mapping_speed_store.id, "data"),
            ])
 

    def _make_inputs(self, container):
        form = container.child(dbc.Form)
        container = form.child(dbc.Row, className="gy-2 my-1")
        container.child(
            LabeledInput(
                label_text="APT Version",
                className="w-auto",
                size="sm",
                input_props={
                    "type": "text",
                    "readonly": True,
                    "value": self._apt.meta.get("version", "designed"),
                },
            )
        )
        container.child(
            LabeledInput(
                label_text="Enabled Detectors",
                className="w-auto",
                size="sm",
                input_props={
                    "type": "text",
                    "readonly": True,
                    "value": f"{len(self._apt)} / {len(self._apt_designed)} ({len(self._apt) / len(self._apt_designed):.0%})",
                },
            )
        )
        container = form.child(dbc.Row, className="gy-2 my-1")
        atm_select = container.child(
            LabeledChecklist(
                label_text="Atm. Quantile",
                className="w-auto",
                size="sm",
                # set to true to allow multiple check
                multi=False,
            )
        ).checklist
        atm_select.options = [
            {
                "label": f"{q} %",
                "value": q,
            }
            for q in self._atm_q_values
        ]
        atm_select.value = self._atm_q_default
        stokes_select = container.child(
            LabeledChecklist(
                label_text="Stokes Params",
                className="w-auto",
                size="sm",
                multi=False,
            )
        ).checklist
        stokes_select.options = [
            {
                "label": f"Intensity (I)",
                "value": "I",
            },
            {
                "label": f"Polarized (Q, U)",
                "value": "QU",
            },
        ]
        stokes_select.value = "I"
        container = form.child(dbc.Row, className="gy-2 my-1")
        alt_input = container.child(
            LabeledInput(
                label_text="Mean Source Elevation",
                suffix_text="deg",
                className="w-auto",
                size="sm",
                input_props={
                    "type": "number",
                    "min": 0.0,
                    "max": 90.0,
                    "value": 60,
                    "style": {"max-width": "5rem"},
                },
            )
        ).input
        return {
            "atm_select": atm_select,
            "stokes_select": stokes_select,
            "alt_input": alt_input,
        }

    def _make_mapping_speed_panel(self, container):
        container = container.child(dbc.Card, outline=True)
        # header = container.child(dbc.CardHeader).child(html.H5("Mapping Speed"))
        body = container.child(dbc.CardBody)
        col_labels_map = {
            "a1100": "1.1 mm",
            "a1400": "1.4 mm",
            "a2000": "2.0 mm",
        }
        thdr = html.Thead(
            html.Tr(
                [
                    html.Th("", style={"width": "40%"}),
                    html.Th(col_labels_map["a1100"]),
                    html.Th(col_labels_map["a1400"]),
                    html.Th(col_labels_map["a2000"]),
                ]
            )
        )
        row_labels_map = {
            "det_info": "# of Detectors (Enabled / Total)",
            "mapping_speed": "Mapping Speed (deg2 / h / mJy2)",
        }

        def _make_det_info(array_name):
            aapt = self._apt[self._apt["array_name"] == array_name]
            aapt_designed = self._apt_designed[
                self._apt_designed["array_name"] == array_name
            ]
            return f"{len(aapt)} / {len(aapt_designed)} ({len(aapt) / len(aapt_designed):.0%})"

        values = dict()
        spinner_color = "primary"
        loading_indicator = {"outputs": [], "values": []}
        for i, r in enumerate(row_labels_map.keys()):
            for j, c in enumerate(col_labels_map.keys()):
                # get the detector info text
                if r == "det_info":
                    children = _make_det_info(c)
                else:
                    children = "..."
                v = values[r, c] = dbc.Spinner(
                    id=body.id + f"-{i}{j}",
                    children=children,
                    color=spinner_color,
                    size="sm",
                    spinner_class_name="ms-0",
                )
                if r == "mapping_speed":
                    loading_indicator["outputs"].append(Output(v.id, "color"))
                    loading_indicator["values"].append(spinner_color)
        tbody = []
        for r, label in row_labels_map.items():
            trs = [html.Td(label)]
            for c in col_labels_map.keys():
                trs.append(html.Td(values[r, c]))
            tbody.append(html.Tr(trs))
        tbody = html.Tbody(tbody)
        body.child(dbc.Table, [thdr, tbody], size="sm")
        return values, loading_indicator

    def _make_result_panel(self, container, input_keys, output_key, title):
        input_key_component_kw_map = {
            "x": {
                "label_text": "Map x-size",
                "suffix_text": "arcmin",
                "input_props": {
                    "type": "number",
                    "min": 0.0,
                    "value": 20,
                    # "style": {"max-width": "10rem"},
                },
            },
            "y": {
                "label_text": "Map y-size",
                "suffix_text": "arcmin",
                "input_props": {
                    "type": "number",
                    "min": 0.0,
                    "value": 20,
                    # "style": {"max-width": "10rem"},
                },
            },
            "d": {
                "label_text": "Desired Map RMS",
                "suffix_text": "mJy",
                "input_props": {
                    "type": "number",
                    "min": 0.0,
                    "value": 1.0,
                    # "style": {"max-width": "10rem"},
                },
            },
            "t": {
                "label_text": "Integration Time",
                "suffix_text": "min",
                "input_props": {
                    "type": "number",
                    "min": 0.0,
                    "value": 30.0,
                    # "style": {"max-width": "10rem"},
                },
            },
        }
        output_key_component_kw_map = {
            "d": {"label_text": "Estimated Map RMS"},
            "t": {"label_text": "Estimated Mapping Time"},
        }
        container = container.child(dbc.Card, outline=True)
        container.child(dbc.CardHeader).child(html.H5(title))
        body = container.child(dbc.CardBody)
        form = body.child(dbc.Form)
        entries = form.colgrid(3, 1, gy=2)
        inputs = dict()
        for i, d in enumerate(input_keys):
            kw = input_key_component_kw_map[d]
            inputs[d] = (
                entries[i]
                .child(LabeledInput(className="w-auto", size="sm", **kw))
                .input
            )
        body.child(html.Hr())

        spinner_color = "primary"

        def _make_spinner(id):
            return dbc.Spinner(
                id=id,
                children='...',
                color=spinner_color,
                size="sm",
                spinner_class_name="ms-0",
            )

        row_labels_map = {
            "a1100": "1.1 mm",
            "a1400": "1.4 mm",
            "a2000": "2.0 mm",
        }
        values = {
            "stokes_params": _make_spinner(id=body.id + '-stokes')
        }
        thdr = html.Thead(
            html.Tr(
                [
                    html.Th([
                        html.Div(
                        [html.Div(
                            output_key_component_kw_map[output_key]['label_text'], className='me-2'),
                        values['stokes_params']
                        ], className='d-flex flex-wrap') ], colSpan=2),
                ]
            )
        )
        for i, r in enumerate(row_labels_map.keys()):
            v = values[r] = _make_spinner(id=body.id + f"-result{i}")

        loading_indicator = {"outputs": [], "values": []}
        for v in values.values():
            loading_indicator["outputs"].append(Output(v.id, "color"))
            loading_indicator["values"].append(spinner_color)
           
        tbody = []
        for r, label in row_labels_map.items():
            tbody.append(html.Tr([html.Td(label, style={'width': "50%"}), html.Td(values[r])]))
        tbody = html.Tbody(tbody)
        body.child(dbc.Table, [thdr, tbody], size="sm")
        return inputs, values, loading_indicator       

    def _make_readme_panel(self, container):
        container = container.child(html.Div, className='p-3 bg-light rounded-3')
        jcontent = [
            dcc.Markdown(
                """
# Notes on this Calculator

This calculator is powered by the
`tolteca.simu.toltec.ToltecArrayPowerLoadingModel` used in the TolTEC
observation simulator, configured to use the latest instrument
status/measurements from the commissioning data as of September 14, 2022
(revision 20220914.0).

The model implements the mapping speed calculation described in Bryan et al.
2018.  It uses models of the detector noise, the atmosphere, the atmosphere
fluctuations, and the telescope surface.

This calculator does not include the overhead. For a more sophisticated
calculation of observing time, depths, and project overhead that requires
carefully designing the full observation, please use the

### [Fancy ObsPlanner](https://toltec.lmtgtm.org/toltec_obs_planner)

***

** Notes on Polarimetry **

For polarimetry measurements, the relevant quantity is the 1-sigma uncertainty
(error bar) in the polarized intensity. Here the polarized intensity is defined
as the total intensity multiplied by the polarization fraction. Because of the
need to separately measure Stokes Q and Stokes U, the integration time required
to reach a given polarized intensity error bar is double that required to reach
the same total intensity error bar. We apply an additional penalty due to
imperfect polarization modulation (e.g., non-ideal behavior in the half-wave
plate). This penalty is a factor of 10% in sensitivity (~20% in observing
time)

See:

A Primer on Far-Infrared Polarimetry, Hildebrand, R.H., et al. 2000,
Publications of the Astronomical Society of the Pacific, 112, 1215.

Appendix B.1 of Planck Intermediate Results XIX ([link](https://arxiv.org/pdf/1405.0871.pdf))
""",
    link_target="_blank",
            ),
        ]
        container.children = jcontent
