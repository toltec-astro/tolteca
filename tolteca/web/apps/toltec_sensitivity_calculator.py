#!/usr/bin/env python

from dataclasses import dataclass, field
from pathlib import Path
from schema import Or
from typing import Union

from tollan.utils.dataclass_schema import add_schema

import astropy.units as u

from .. import apps_registry, get_app_config
from ...utils.common_schema import PhysicalTypeSchema, RelPathSchema


@apps_registry.register("toltec_sensitivity_calculator")
@add_schema
@dataclass
class ToltecSensitivityCalculatorConfig:
    """The config class for the TolTEC sensitivity calculator app."""

    lmt_tel_surface_rms: Union[None, u.Quantity] = field(
        default=None,
        metadata={
            "description": "The LMT surface RMS.",
            "schema": Or(None, PhysicalTypeSchema("length")),
        },
    )
    toltec_det_noise_factors: Union[None, dict] = field(
        default=None,
        metadata={
            "description": "The TolTEC detector noise factor, per atm_model_name.",
            "schema": Or({str: float}, None),
        },
    )
    toltec_apt_path: Union[None, Path] = field(
        default=None,
        metadata={
            "description": "The apt filepath for TolTEC.",
            "schema": Or(RelPathSchema(), None),
        },
    )
    title_text: str = field(
        default="TolTEC Sensitivity Calculator",
        metadata={"description": "The title text of the page."},
    )
    subtitle_text: str = field(
        default="",
        metadata={"description": "The subtitle text of the page."},
    )


def DASHA_SITE():
    """The dasha site entry point."""

    dasha_config = get_app_config(ToltecSensitivityCalculatorConfig).to_dict()
    dasha_config.update(
        {
            "template": "tolteca.web.templates.toltec_sensitivity_calculator:ToltecSensitivityCalculator",
            # 'THEME': dbc.themes.LUMEN,
            # 'ASSETS_IGNORE': 'bootstrap.*',
            # 'DEBUG': True,
        }
    )
    return {
        "extensions": [
            {"module": "dasha.web.extensions.dasha", "config": dasha_config},
        ]
    }
