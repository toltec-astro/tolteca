#!/usr/bin/env python


from dash_component_template import ComponentTemplate
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State

from dasha.web.templates.common import (
    CollapseContent,
    LabeledChecklist,
    LabeledInput,
)

import astropy.units as u

from .base import ObsSite
from ....simu.lmt import lmt_info


class Lmt(ObsSite, name="lmt"):
    """An `ObsSite` for LMT."""

    info = lmt_info
    display_name = info["name_long"]
    observer = info["observer"]

    class ControlPanel(ComponentTemplate):
        class Meta:
            component_cls = dbc.Form

        _atm_q_values = [25, 50, 75]
        _atm_q_default = 25
        # _tel_surface_rms_default = 76 << u.um

        def __init__(self, site, **kwargs):
            super().__init__(**kwargs)
            self._site = site
            container = self
            self._info_store = container.child(
                dcc.Store,
                data={
                    "name": site.name,
                    "tel_surface_rms_um": site._tel_surface_rms.to_value(u.um),
                },
            )

        @property
        def info_store(self):
            return self._info_store

        def setup_layout(self, app):
            container = self.child(dbc.Row, className="gy-2")
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

            if False:
                tel_surface_input = container.child(
                    LabeledInput(
                        label_text="Tel. Surface RMS",
                        className="w-auto",
                        size="sm",
                        input_props={
                            # these are the dbc.Input kwargs
                            "type": "number",
                            "min": 0,
                            "max": 200,
                            "placeholder": "0-200",
                            "style": {"flex": "0 1 5rem"},
                        },
                        suffix_text="μm",
                    )
                ).input
                tel_surface_input.value = self._tel_surface_rms_default.to_value(u.um)
            super().setup_layout(app)

            # collect inputs to store
            app.clientside_callback(
                """
                // function(atm_select_value, tel_surface_value, data_init) {
                function(atm_select_value, data_init) {
                    data = {...data_init}
                    data['atm_model_name'] = 'am_q' + atm_select_value
                    // data['tel_surface_rms'] = tel_surface_value + ' um'
                    return data
                }
                """,
                Output(self.info_store.id, "data"),
                [
                    Input(atm_select.id, "value"),
                    # Input(tel_surface_input.id, "value"),
                    State(self.info_store.id, "data"),
                ],
            )

    class InfoPanel(ComponentTemplate):
        class Meta:
            component_cls = dbc.Container

        def __init__(self, site, **kwargs):
            kwargs.setdefault("fluid", True)
            super().__init__(**kwargs)
            self._site = site

        def setup_layout(self, app):
            container = self.child(
                CollapseContent(button_text="Current Site Info ...")
            ).content
            info = self._site.info
            location = info["location"]
            timezone_local = info["timezone_local"]
            info_store = container.child(
                dcc.Store,
                data={
                    "name": self._site.name,
                    "display_name": self._site.display_name,
                    "lon": location.lon.degree,
                    "lat": location.lat.degree,
                    "height_m": location.height.to_value(u.m),
                    "timezone_local": timezone_local.zone,
                },
            )
            pre_kwargs = {"className": "mb-0", "style": {"font-size": "80%"}}
            loc_display = container.child(
                html.Pre,
                f"Location: {location.lon.to_string()} " f"{location.lat.to_string()}",
                # f' {location.height:.0f}'
                **pre_kwargs,
            )
            loc_display.className += " mb-0"
            time_display = container.child(html.Pre, **pre_kwargs)
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
                Output(time_display.id, "children"),
                Input(timer.id, "n_intervals"),
                Input(info_store.id, "data"),
                prevent_initial_call=True,
            )
