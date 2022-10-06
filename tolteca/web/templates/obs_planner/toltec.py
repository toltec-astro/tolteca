#!/usr/bin/env python

import functools
from dash_component_template import ComponentTemplate
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
from io import StringIO
from astropy.time import Time

from dasha.web.templates.common import LabeledChecklist, DownloadButton
import dash_js9 as djs9

import astropy.units as u
from astropy.table import QTable
from astropy.modeling.functional_models import GAUSSIAN_SIGMA_TO_FWHM
from astropy.coordinates import SkyCoord
from astropy.convolution import convolve_fft
from astropy.convolution import Gaussian2DKernel
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import pandas as pd
import cv2
from io import BytesIO
from base64 import b64encode
import functools

from tollan.utils.log import get_logger, timeit

from ....utils import yaml_dump
from ....simu.toltec.toltec_info import toltec_info
from ....simu import instrument_registry
from ....simu.utils import SkyBoundingBox, make_wcs
from ....simu.exports import LmtOTExporterConfig
from ....simu.toltec.toltec_overhead import OverheadCalculation

from .base import ObsInstru, ObsSite


@functools.lru_cache(maxsize=None)
def _get_apt_designed():
    instru = instrument_registry.schema.validate(
        {"name": "toltec", "polarized": False}, create_instance=True
    )
    simulator = instru.simulator
    return simulator.array_prop_table


class Toltec(ObsInstru, name="toltec"):
    """An `ObsInstru` for TolTEC."""

    info = toltec_info
    display_name = info["name_long"]

    class ResultPanel(ComponentTemplate):
        class Meta:
            component_cls = dbc.Container

        def __init__(self, instru, **kwargs):
            kwargs.setdefault("fluid", True)
            super().__init__(**kwargs)
            self._instru = instru
            container = self
            fitsview_container, info_container = container.colgrid(1, 2)
            self._fitsview_loading = fitsview_container.child(
                dbc.Spinner,
                show_initially=False,
                color="primary",
                spinner_style={"width": "5rem", "height": "5rem"},
            )
            self._fitsview = self._fitsview_loading.child(
                djs9.DashJS9,
                style={"width": "100%", "min-height": "500px", "height": "40vh"},
            )
            self._info_loading = info_container.child(
                dbc.Spinner,
                show_initially=False,
                color="primary",
                spinner_style={"width": "5rem", "height": "5rem"},
            )

        def make_callbacks(self, app, exec_info_store_id):
            fitsview = self._fitsview
            info = self._info_loading.child(html.Div, style={"min-height": "500px"})

            app.clientside_callback(
                """
                function(exec_info) {
                    // console.log(exec_info);
                    if (!exec_info) {
                        return Array(1).fill(window.dash_clientside.no_update);
                    }
                    return [exec_info.instru.fits_images];
                }
                """,
                [
                    Output(fitsview.id, "data"),
                ],
                [
                    Input(exec_info_store_id, "data"),
                ],
            )

            app.clientside_callback(
                """
                function(exec_info) {
                    // console.log(exec_info);
                    if (!exec_info) {
                        return Array(1).fill(window.dash_clientside.no_update);
                    }
                    return [exec_info.instru.info];
                }
                """,
                [
                    Output(info.id, "children"),
                ],
                [
                    Input(exec_info_store_id, "data"),
                ],
            )

        @property
        def loading_indicators(self):
            return {
                "outputs": [
                    Output(self._fitsview_loading.id, "color"),
                    Output(self._info_loading.id, "color"),
                ],
                "states": [
                    State(self._fitsview_loading.id, "color"),
                    State(self._info_loading.id, "color"),
                ],
            }

    class ResultControlPanel(ComponentTemplate):
        class Meta:
            component_cls = dbc.Container

        def __init__(self, instru, **kwargs):
            kwargs.setdefault("fluid", True)
            super().__init__(**kwargs)
            container = self
            dlbtn_props = {"disabled": True}

            self._lmtot_download = container.child(
                DownloadButton(
                    button_text="LMT OT Script",
                    className="me-2",
                    button_props=dlbtn_props,
                    tooltip=(
                        "Download LMT Observation Tool script to "
                        "execute the observation at LMT."
                    ),
                )
            )
            self._simuconfig_download = container.child(
                DownloadButton(
                    button_text="Simu. Config",
                    className="me-2",
                    button_props=dlbtn_props,
                    tooltip=(
                        "Download tolteca.simu 60_simu.yaml config file to "
                        "run the observation simulator."
                    ),
                )
            )
            self._fits_download = container.child(
                DownloadButton(
                    button_text="Coverage Map",
                    className="me-2",
                    button_props=dlbtn_props,
                    tooltip=(
                        "Download the generated FITS (approximate) coverage "
                        "image for the SINGLE observation (NOT COADDED)."
                    ),
                )
            )
            results_dlbtn_props = dlbtn_props.copy()
            results_dlbtn_props["color"] = "success"
            self._results_download = container.child(
                DownloadButton(
                    button_text="Export Results for Proposal Submission",
                    className="me-2",
                    button_props=results_dlbtn_props,
                    tooltip=(
                        "Export the current state and calculation results as ECSV table "
                        " to be used for proposal submission."
                    ),
                )
            )

        def make_callbacks(self, app, exec_info_store_id):

            for dl in [
                self._lmtot_download,
                self._simuconfig_download,
                self._fits_download,
                self._results_download,
            ]:
                app.clientside_callback(
                    """
                    function(exec_info) {
                        if (!exec_info) {
                            return true;
                        }
                        return false;
                    }
                    """,
                    Output(dl.button.id, "disabled"),
                    [
                        Input(exec_info_store_id, "data"),
                    ],
                )

            app.clientside_callback(
                r"""
                function(n_clicks, exec_info) {
                    // console.log(exec_info);
                    if (!exec_info) {
                        return window.dash_clientside.no_update;
                    }
                    target = exec_info.exec_config.mapping.target

                    filename = (
                        'target_' + target + '.lmtot').replace(/\s+/g, '-');
                    return {
                        content: exec_info.instru.lmtot,
                        base64: false,
                        filename: filename,
                        type: 'text/plain;charset=UTF-8'
                        };
                }
                """,
                Output(self._lmtot_download.download.id, "data"),
                [
                    Input(self._lmtot_download.button.id, "n_clicks"),
                    State(exec_info_store_id, "data"),
                ],
            )

            app.clientside_callback(
                """
                function(n_clicks, exec_info) {
                    // console.log(exec_info);
                    if (!exec_info) {
                        return window.dash_clientside.no_update;
                    }
                    filename = '60_simu.yaml'
                    return {
                        content: exec_info.instru.simu_config,
                        base64: false,
                        filename: filename,
                        type: 'text/plain;charset=UTF-8'
                        };
                }
                """,
                Output(self._simuconfig_download.download.id, "data"),
                [
                    Input(self._simuconfig_download.button.id, "n_clicks"),
                    State(exec_info_store_id, "data"),
                ],
            )

            app.clientside_callback(
                """
                function(n_clicks, exec_info) {
                    // console.log(exec_info);
                    if (!exec_info) {
                        return window.dash_clientside.no_update;
                    }
                    im = exec_info.instru.fits_images[0]
                    return {
                        content: im.blob,
                        base64: true,
                        filename: im.options.file,
                        type: 'application/fits'
                        };
                }
                """,
                Output(self._fits_download.download.id, "data"),
                [
                    Input(self._fits_download.button.id, "n_clicks"),
                    State(exec_info_store_id, "data"),
                ],
            )

            app.clientside_callback(
                """
                function(n_clicks, exec_info) {
                    // console.log(exec_info);
                    if (!exec_info) {
                        return window.dash_clientside.no_update;
                    }
                    filename = 'toltec_obs_planner_results.ecsv'
                    return {
                        content: exec_info.instru.results,
                        base64: false,
                        filename: filename,
                        type: 'text/plain;charset=UTF-8'
                        };
                }
                """,
                Output(self._results_download.download.id, "data"),
                [
                    Input(self._results_download.button.id, "n_clicks"),
                    State(exec_info_store_id, "data"),
                ],
            )

    class ControlPanel(ComponentTemplate):
        class Meta:
            component_cls = dbc.Form

        def __init__(self, instru, **kwargs):
            super().__init__(**kwargs)
            self._instru = instru
            container = self
            self._info_store = container.child(
                dcc.Store,
                data={
                    "name": instru.name,
                    "det_noise_factors": instru._det_noise_factors,
                    "apt_path": instru._apt_path.as_posix()
                    if instru._apt_path is not None
                    else None,
                    "revision": instru._revision,
                },
            )

        @property
        def info_store(self):
            return self._info_store

        def setup_layout(self, app):
            toltec_info = self._instru.info
            container = self.child(dbc.Row, className="gy-2")
            readme_btn = container.child(
                dbc.Button,
                "README Before Use: Notice of Shared Risks",
                color="danger",
                outline=True,
                size="sm",
            )
            readme_content = [
                dcc.Markdown(
                    """
** WARNING **

The TolTEC team has made its best guesses about the sensitivity and mapping
speed of the instrument based on 8 nights of commissioning data taken in July,
2022.

Users of this tool should be aware that these estimates are very uncertain. We
have made some improvements to the camera since July, but we are still
extrapolating the summer performance over a very wide range of observing
conditions to predict the sensitivity in the much drier winter and spring
months. Users are encouraged to use a very healthy safety margin in their
requested time when writing their proposals.

""",
                    link_target="_blank",
                )
            ]
            container.child(
                dbc.Popover(
                    readme_content,
                    target=readme_btn.id,
                    body=True,
                    trigger="hover",
                )
            )
            # generate apt info

            container.child(
                dbc.Label(
                    f'APT version: {self._instru._apt.meta.get("version", "designed")}',
                    className="ms-2 w-auto mb-0",
                    style={"font-size": "0.875em"},
                )
            )
            container.child(
                dbc.Label(
                    f"Enabled detectors: {len(self._instru._apt)} / {len(self._instru._apt_designed)} ({len(self._instru._apt) / len(self._instru._apt_designed):.0%})",
                    className="ms-2 w-auto mt-0",
                    style={"font-size": "0.875em"},
                )
            )
            poltype_select = container.child(
                LabeledChecklist(
                    label_text="Stokes Params",
                    className="w-auto",
                    size="sm",
                    # set to true to allow multiple check
                    multi=False,
                )
            ).checklist
            poltype_select.options = [
                {
                    "label": "Total Intensity (I)",
                    "value": "I",
                },
                {
                    "label": "Polarized Emission (Q/U)",
                    "value": "QU",
                },
            ]
            poltype_select.value = "I"
            pol_readme_content = [
                dcc.Markdown(
                    """
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
                )
            ]
            container.child(
                dbc.Popover(
                    pol_readme_content,
                    target=poltype_select.id,
                    body=True,
                    trigger="hover",
                )
            )
            band_select = container.child(
                LabeledChecklist(
                    label_text="TolTEC band",
                    className="w-auto",
                    size="sm",
                    # set to true to allow multiple check
                    multi=False,
                    input_props={"style": {"text-transform": "none"}},
                )
            ).checklist
            band_select.options = [
                {
                    "label": str(toltec_info[a]["wl_center"]),
                    "value": a,
                }
                for a in toltec_info["array_names"]
            ]
            band_select.value = toltec_info["array_names"][0]

            covtype_select = container.child(
                LabeledChecklist(
                    label_text="Coverage Unit",
                    className="w-auto",
                    size="sm",
                    # set to true to allow multiple check
                    multi=False,
                )
            ).checklist
            covtype_select.options = [
                {
                    "label": "mJy/beam",
                    "value": "depth",
                },
                {
                    "label": "s/pixel",
                    "value": "time",
                },
            ]
            covtype_select.value = "depth"
            super().setup_layout(app)

            # collect inputs to store
            app.clientside_callback(
                """
                function(poltype_select_value, band_select_value, covtype_select_value, data_init) {
                    data = {...data_init}
                    data['stokes_params'] = poltype_select_value
                    data['polarized'] = (poltype_select_value === "QU")
                    data['array_name'] = band_select_value
                    data['coverage_map_type'] = covtype_select_value
                    return data
                }
                """,
                Output(self.info_store.id, "data"),
                [
                    Input(poltype_select.id, "value"),
                    Input(band_select.id, "value"),
                    Input(covtype_select.id, "value"),
                    State(self.info_store.id, "data"),
                ],
            )

    @staticmethod
    def _hdulist_to_base64(hdulist):
        fo = BytesIO()
        hdulist.writeto(fo, overwrite=True)
        return b64encode(fo.getvalue()).decode("utf-8")

    @classmethod
    def make_traj_data(cls, exec_config, bs_traj_data):
        logger = get_logger()
        logger.debug("make traj data for instru toltec")
        # get observer from site name
        observer = ObsSite.get_observer(exec_config.site_data["name"])
        mapping_model = exec_config.mapping.get_model(observer=observer)

        # tolteca.simu
        simrt = exec_config.get_simulator_runtime()
        simu_config = simrt.config
        simu_config_yaml = yaml_dump(simrt.config.to_config_dict())

        # we need both the designed apt and the simulator apt
        apt_designed = _get_apt_designed()

        apt = simu_config.instrument.simulator.array_prop_table
        array_name = exec_config.instru_data["array_name"]
        # apt_0 is the apt for the current selected array
        apt_0 = apt[apt["array_name"] == array_name]

        # this is the apt including only detectors on the edge
        # useful for making the footprint outline
        # we use the full designed apt here
        apt_0_designed = apt_designed[apt_designed["array_name"] == array_name]
        ei = apt_designed.meta[array_name]["edge_indices"]
        apt_outline = apt_0_designed[ei]

        det_dlon = apt_0["x_t"]
        det_dlat = apt_0["y_t"]

        det_outline_dlon = apt_outline["x_t"]
        det_outline_dlat = apt_outline["y_t"]

        # apply the footprint on target
        # to do so we find the closest poinit in the trajectory to
        # the target and do the transformation
        bs_coords_icrs = SkyCoord(bs_traj_data["ra"], bs_traj_data["dec"], frame="icrs")
        target_icrs = mapping_model.target.transform_to("icrs")
        i_closest = np.argmin(target_icrs.separation(bs_coords_icrs))
        # the center of the array overlay in altaz
        az1 = bs_traj_data["az"][i_closest]
        alt1 = bs_traj_data["alt"][i_closest]
        t1 = bs_traj_data["time_obs"][i_closest]
        c1 = SkyCoord(az=az1, alt=alt1, frame=observer.altaz(time=t1))
        det_altaz = SkyCoord(
            det_dlon, det_dlat, frame=c1.skyoffset_frame()
        ).transform_to(c1.frame)
        det_icrs = det_altaz.transform_to("icrs")

        det_outline_altaz = SkyCoord(
            det_outline_dlon, det_outline_dlat, frame=c1.skyoffset_frame()
        ).transform_to(c1.frame)
        det_outline_icrs = det_outline_altaz.transform_to("icrs")

        det_sky_bbox_icrs = SkyBoundingBox.from_lonlat(det_icrs.ra, det_icrs.dec)
        # make coverage fits image in s_per_pix
        # we'll init the power loading model to estimate the conversion factor
        # of this to mJy/beam
        cov_hdulist_s_per_pix = cls._make_cov_hdulist(ctx=locals())
        # overlay traces
        # each trace is for one polarimetry group
        offset_traces = list()
        for i, (pg, marker) in enumerate([(0, "cross"), (1, "x")]):
            mask = apt_0["pg"] == pg
            offset_traces.append(
                {
                    "x": det_dlon[mask].to_value(u.arcmin),
                    "y": det_dlat[mask].to_value(u.arcmin),
                    "mode": "markers",
                    "marker": {
                        "symbol": marker,
                        "color": "gray",
                        "size": 6,
                    },
                    "legendgroup": "toltec_array_fov",
                    "showlegend": i == 0,
                    "name": f"Toggle FOV: {cls.info[array_name]['name_long']}",
                }
            )

        # skyview layers
        skyview_layers = list()
        n_dets = len(det_icrs)
        det_tbl = pd.DataFrame.from_dict(
            {
                "ra": det_icrs.ra.degree,
                "dec": det_icrs.dec.degree,
                "color": ["blue"] * n_dets,
                "type": ["circle"] * n_dets,
                "radius": [cls.info[array_name]["a_fwhm"].to_value(u.deg) * 0.5]
                * n_dets,
            }
        )
        skyview_layers.extend(
            [
                {
                    "type": "overlay",
                    "data": det_tbl.to_dict(orient="records"),
                    "options": {
                        "name": f"Detectors: {cls.info[array_name]['name_long']}",
                        "show": False,
                    },
                },
                {
                    "type": "overlay",
                    "data": [
                        {
                            "type": "polygon",
                            "data": list(
                                zip(
                                    det_outline_icrs.ra.degree,
                                    det_outline_icrs.dec.degree,
                                )
                            ),
                        }
                    ],
                    "options": {
                        "name": f"FOV: {cls.info[array_name]['name_long']}",
                        "color": "#cc66cc",
                        "show": True,
                        "lineWidth": 8,
                    },
                },
            ]
        )

        # use power loading model to infer the sensitivity
        # this is rough esitmate based on the mean altitude of the observation.
        tplm = simu_config.sources[0].get_power_loading_model()
        target_alt = bs_traj_data["target_alt"]
        alt_mean = target_alt.mean()
        t_exp = bs_traj_data["t_exp"]
        # calcuate overhead
        if simu_config.mapping.type in [
            "raster",
        ]:
            t_exp_eff = t_exp - simu_config.mapping.t_turnaround * (
                simu_config.mapping.n_scans - 1
            )
        else:
            t_exp_eff = t_exp

        # for this purpose we generate the info for all the three arrays
        sens_coeff = np.sqrt(2.0)
        sens_tbl = list()
        array_names = cls.info["array_names"]

        def _get_pol_noise_factor():
            return 2**0.5 + 0.1

        for an in array_names:
            # TODO fix the api
            aplm = tplm._array_power_loading_models[an]
            aapt = apt[apt["array_name"] == an]
            aapt_designed = apt_designed[apt_designed["array_name"] == an]
            n_dets_info = f"{len(aapt)} / {len(aapt_designed)} ({len(aapt) / len(aapt_designed):.0%})"
            result = {
                "array_name": an,
                "alt_mean": alt_mean,
                "P": aplm._get_P(alt_mean),
                "n_dets_info": n_dets_info,
            }
            result.update(aplm._get_noise(alt_mean, return_avg=True))
            result["nefd_I"] = result["nefd"]
            result["nefd_QU"] = _get_pol_noise_factor() * result["nefd_I"]
            result["dsens_I"] = sens_coeff * result["nefd_I"].to(u.mJy * u.s**0.5)
            result["dsens_QU"] = sens_coeff * result["nefd_QU"].to(u.mJy * u.s**0.5)
            # format the mapping speed here because of the issue in unicode unit formatting
            ms = aplm.get_mapping_speed(alt_mean, n_dets=len(aapt))
            ms_pol = ms / _get_pol_noise_factor() ** 2
            result["mapping_speed_I"] = f"{ms.value:.0f} {ms.unit:s}"
            result["mapping_speed_QU"] = f"{ms_pol.value:.0f} {ms_pol.unit:s}"
            sens_tbl.append(result)

        if exec_config.instru_data["stokes_params"] == "I":
            polarized = False
        elif exec_config.instru_data["stokes_params"] == "QU":
            polarized = True

        sens_tbl = QTable(rows=sens_tbl)
        logger.debug(f"summary table for all arrays:\n{sens_tbl}")

        # for the current array we get the mapping area from the cov map
        # and convert to mJy/beam if requested
        def _get_entry(an):
            return sens_tbl[sens_tbl["array_name"] == an][0]

        sens_entry = _get_entry(array_name)
        cov_data = cov_hdulist_s_per_pix[1].data
        cov_wcs = WCS(cov_hdulist_s_per_pix[1].header)
        cov_pixarea = cov_wcs.proj_plane_pixel_area()
        cov_max = cov_data.max()
        m_cov = cov_data > 0.02 * cov_max
        m_cov_01 = cov_data > 0.1 * cov_max
        map_area = (m_cov_01.sum() * cov_pixarea).to(u.deg**2)
        a_stddev = cls.info[array_name]["a_fwhm"] / GAUSSIAN_SIGMA_TO_FWHM
        b_stddev = cls.info[array_name]["b_fwhm"] / GAUSSIAN_SIGMA_TO_FWHM
        beam_area = 2 * np.pi * a_stddev * b_stddev
        beam_area_pix2 = (beam_area / cov_pixarea).to_value(u.dimensionless_unscaled)
        if polarized:
            nefd_key = "nefd_QU"
        else:
            nefd_key = "nefd_I"

        def _get_cov_data_mJy_per_beam(cov_data):
            cov_data_mJy_per_beam = np.zeros(cov_data.shape, dtype="d")
            cov_data_mJy_per_beam[m_cov] = (
                sens_coeff
                * sens_entry[nefd_key]
                / np.sqrt(cov_data[m_cov] * beam_area_pix2)
            )
            return cov_data_mJy_per_beam

        cov_data_mJy_per_beam = _get_cov_data_mJy_per_beam(cov_data)
        # calculate rms depth from the depth map
        depth_rms = np.median(cov_data_mJy_per_beam[m_cov_01]) << u.mJy
        # scale the depth rms to all arrays and update the sens tbl
        sens_tbl["polarized"] = polarized
        sens_tbl["depth_stokes_params"] = (
            "Total Intensity (I)" if not polarized else "Polarized (Q/U)"
        )
        sens_tbl["depth_rms"] = depth_rms / sens_entry[nefd_key] * sens_tbl[nefd_key]
        sens_tbl["t_exp"] = t_exp
        sens_tbl["t_exp_eff"] = t_exp_eff
        sens_tbl["map_area"] = map_area
        # calcuate the number of passes to achieve desired depth
        desired_sens = exec_config.desired_sens
        n_passes = int(
            np.ceil((_get_entry(array_name)["depth_rms"] / desired_sens) ** 2)
        )
        sens_tbl["n_passes"] = n_passes
        sens_tbl["depth_rms_coadd_desired"] = desired_sens
        sens_tbl["depth_rms_coadd_actual"] = sens_tbl["depth_rms"] / (
            sens_tbl["n_passes"] ** 0.5
        )
        # calcuate overhead
        science_time = (n_passes * t_exp).to(u.h)
        science_time_per_night = 4.0 << u.h
        n_nights = int(
            np.ceil(
                (science_time / science_time_per_night).to_value(
                    u.dimensionless_unscaled
                )
            )
        )
        sens_tbl["proj_science_time"] = science_time
        sens_tbl["proj_science_time_per_night"] = n_nights
        sens_tbl["proj_n_nights"] = n_nights
        overhead = OverheadCalculation(
            map_type=simu_config.mapping.type,
            science_time=science_time.to_value(u.s),
            add_nights=n_nights,
            science_time_overhead_fraction=(1 - t_exp_eff / t_exp).to_value(
                u.dimensionless_unscaled
            ),
        ).make_output_dict()
        sens_tbl["proj_science_overhead_time"] = (
            overhead["science_overhead"] << u.s
        ).to(u.h)
        sens_tbl["proj_total_time"] = (overhead["total_time"] << u.s).to(u.h)
        sens_tbl["proj_overhead_time"] = (overhead["overhead_time"] << u.s).to(u.h)
        sens_tbl["proj_overhead_percent"] = f"{overhead['overhead_percent']:.1%}"

        # make cov hdulist depending on the instru_data cov unit settings
        if exec_config.instru_data["coverage_map_type"] == "depth":
            cov_hdulist = cov_hdulist_s_per_pix.copy()
            for i in range(1, 3):
                cov_hdulist[i].header["BUNIT"] = "mJy / beam"
                cov_hdulist[i].data = _get_cov_data_mJy_per_beam(cov_hdulist[i].data)
        else:
            cov_hdulist = cov_hdulist_s_per_pix

        # from the cov image we can create a countour showing the outline of
        # the observation on the skyview
        cov_ctr = cov_hdulist[1].data.copy()
        cov_ctr[~m_cov_01] = 1
        cov_ctr[m_cov_01] = 0
        # get the largest connected component from the center
        # then use that to derive the contour
        im = cv2.normalize(
            src=cov_ctr,
            dst=None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8UC1,
        )
        h, w = im.shape[:2]
        im_mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(im, im_mask, (w // 2, h // 2), 255, flags=4 | (255 << 8))
        im_mask = im_mask[1:-1, 1:-1]
        cxy = cv2.findContours(im_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        # the cxy is a tuple of multiple contours
        # we select the first significant one to use
        # hopefully this is the outline...
        for c in cxy:
            if c.shape[0] > 2:
                cxy = c
                break
        else:
            # no coutrous found, just set to the last one
            # current one
            logger.debug("unabled to generate outline contour")
            cxy = c
        cxy_s = cv2.approxPolyDP(cxy, 0.002 * cv2.arcLength(cxy, True), True)[:, 0, :]
        cra, cdec = cov_wcs.pixel_to_world_values(cxy_s[:, 0], cxy_s[:, 1])
        skyview_layers.extend(
            [
                {
                    "type": "overlay",
                    "data": [
                        {
                            "type": "polygon",
                            "data": list(zip(cra, cdec)),
                        }
                    ],
                    "options": {
                        "name": "Coverage Outline",
                        "color": "#66cccc",
                        "show": True,
                        "lineWidth": 4,
                    },
                },
            ]
        )

        # create the layout to display the sensitivity info table
        def _make_sens_tab_content(an):
            entry = _get_entry(an)

            def _fmt(v):
                if isinstance(v, u.Quantity):
                    return f"{v.value:.3g} {v.unit:unicode}"
                return v

            key_labels = {
                "array_name": (
                    "Array Name",
                    "The TolTEC array name, e.g., a1100 for the 1.1mm array.",
                ),
                "alt_mean": ("Mean Alt.", "The mean altitude of the mapping pattern."),
                "n_dets_info": (
                    "# of Detectors (Enabled / Total)",
                    "Number of detectors configured for science data vs total. These numbers are likely to change following fall commissioning.",
                ),
            }
            if polarized:
                key_labels.update(
                    {
                        "dsens_QU": (
                            "Detector Sens. (Polarized Emission)",
                            "Estimated detector sensitivity for polarized emission.",
                        ),
                        "mapping_speed_QU": (
                            "Mapping Speed (Polarized Emission)",
                            "Estimated mapping speed for polarized emission.",
                        ),
                    }
                )
            else:
                key_labels.update(
                    {
                        "dsens_I": (
                            "Detector Sens. (Total Intensity)",
                            "Estimated detector sensitivity for total intensity.",
                        ),
                        "mapping_speed_I": (
                            "Mapping Speed (Total Intensity)",
                            "Estimated mapping speed for total intensity.",
                        ),
                    }
                )
            key_labels.update(
                {
                    "map_area": ("Map Area", "Effective area of the mapping pattern."),
                    "t_exp": (
                        "Exp. Time per Pass",
                        "The time needed to completed one pass of the mapping pattern.",
                    ),
                    "t_exp_eff": (
                        "Effective On-target Time per Pass",
                        'The effective time on target. This equals to "Exp. Time" except for raster maps where the turnaround time is excluded.',
                    ),
                    "depth_stokes_params": (
                        "Stokes Params of Depth Values",
                        "Indicated whether the depth values reported in this table are for total intensity or polarimetry.",
                    ),
                    "depth_rms": (
                        "Median RMS Sens. per Pass",
                        "Estimated median of the RMS sensitivity for a single pass",
                    ),
                    "n_passes": (
                        "Number of Passes",
                        "The number of passes to execute the mapping pattern for to reach the desired coadded map RMS.",
                    ),
                    "depth_rms_coadd_actual": (
                        "Coadded Map RMS Sens.",
                        'The estimated map RMS sensitivity after coadding "Number of Passes" individual exposures.',
                    ),
                    "proj_science_time": (
                        "Project Total Science Time",
                        "The total time to finish all the observations",
                    ),
                    "proj_science_overhead_time": (
                        "Science Time Overhead",
                        'The time that is considered overhead within the "Project Science Time".',
                    ),
                    "proj_n_nights": (
                        "Assumed Number of Obs. Nights",
                        "Number of nights the project needs to finish, assuming 4h of up-time per night.",
                    ),
                    "proj_total_time": (
                        "Project Total Time (incl. Overhead)",
                        "The time to finish the project, including all overheads. Refer to this number for proposal sumission.",
                    ),
                    "proj_overhead_time": (
                        "Project Overhead",
                        'The time that is considered overhead within the "Project Total Time".',
                    ),
                    "proj_overhead_percent": (
                        "Project Overhead %",
                        "The project overhead percentage.",
                    ),
                }
            )
            data = {v[0]: _fmt(entry[k]) for k, v in key_labels.items()}
            # data["Coverage Map Unit"] = cov_hdulist[1].header["BUNIT"]
            df = pd.DataFrame(data.items(), columns=["", ""])
            t = dbc.Table.from_dataframe(
                df, striped=True, bordered=True, hover=True, className="mx-0 my-0"
            )
            # build the table
            tbody = []
            # this is to ensure unique id for the generated layout
            id_base = f"trajdata-{id(exec_config)}"
            for k, v in key_labels.items():
                help_id = f"{id_base}-{k}"
                if k == "proj_total_time":
                    tr_kw = {"className": "bg-info"}
                else:
                    tr_kw = {}
                trow = html.Tr(
                    [
                        html.Td(
                            [
                                v[0],
                                html.Span(
                                    className="ms-2 fa-regular fa-circle-question",
                                    id=help_id,
                                ),
                                dbc.Popover(
                                    v[1],
                                    target=help_id,
                                    body=True,
                                    trigger="hover",
                                ),
                            ]
                        ),
                        html.Td(_fmt(entry[k])),
                    ],
                    **tr_kw,
                )
                tbody.append(trow)
            tbody = html.Tbody(tbody)
            t = dbc.Table(
                [tbody], striped=True, bordered=True, hover=True, className="mx-0 my-0"
            )
            # get rid of the first child which is the header
            # t.children = t.children[1:]
            # get the index of the total time
            # i_total_time = list(key_labels.keys()).index("proj_total_time")
            # t.children[0].children[i_total_time].className = "bg-info"
            return dbc.Card(
                [
                    dbc.CardBody(
                        t, className="py-0 px-0", style={"border-width": "0px"}
                    ),
                ],
            )

        sens_tbl_layout = dbc.Tabs(
            [
                dbc.Tab(
                    _make_sens_tab_content(an),
                    label=str(cls.info[an]["wl_center"]),
                    tab_id=an,
                    activeTabClassName="fw-bold",
                )
                for an in array_names
            ],
            active_tab=array_name,
        )

        # lmtot script export
        lmtot_exporter = LmtOTExporterConfig(save=False)
        lmtot_content = lmtot_exporter(simu_config)
        # collect all results as ECSV
        sens_tbl.meta["created_at"] = Time.now().isot
        sens_tbl.meta["exec_config"] = exec_config.to_yaml_dict()
        results_content = StringIO()
        sens_tbl.write(results_content, format="ascii.ecsv", overwrite=True)
        results_content = results_content.getvalue()

        return {
            "dlon": det_dlon,
            "dlat": det_dlat,
            "az": det_altaz.az,
            "alt": det_altaz.alt,
            "ra": det_icrs.ra,
            "dec": det_icrs.dec,
            "sky_bbox_icrs": det_sky_bbox_icrs,
            "overlay_traces": {"offset": offset_traces},
            "skyview_layers": skyview_layers,
            "results": {
                "fits_images": [
                    {
                        "options": {
                            "file": f"obsplanner_toltec_{array_name}_cov.fits",
                        },
                        "blob": cls._hdulist_to_base64(cov_hdulist),
                    }
                ],
                "lmtot": lmtot_content,
                "simu_config": simu_config_yaml,
                "info": sens_tbl_layout,
                "results": results_content,
            },
        }

    @classmethod
    def _make_cov_hdu_approx(cls, ctx):
        logger = get_logger()
        # unpack the cxt
        bs_traj_data = ctx["bs_traj_data"]
        det_icrs = ctx["det_icrs"]
        det_sky_bbox_icrs = ctx["det_sky_bbox_icrs"]
        dt_smp = bs_traj_data["time_obs"][1] - bs_traj_data["time_obs"][0]
        array_name = ctx["array_name"]

        # create the wcs
        pixscale = u.pixel_scale(4.0 << u.arcsec / u.pix)
        # the pixsize will be int factor of 2 arcsec.
        adaptive_pixscale_factor = 0.5
        n_pix_max = 1e6  # 8 MB of data
        bs_sky_bbox_icrs = bs_traj_data["sky_bbox_icrs"]
        sky_bbox_wcs = bs_sky_bbox_icrs.pad_with(
            det_sky_bbox_icrs.width + (2 << u.arcmin),
            det_sky_bbox_icrs.height + (2 << u.arcmin),
        )
        wcsobj = make_wcs(
            sky_bbox=sky_bbox_wcs,
            pixscale=pixscale,
            n_pix_max=n_pix_max,
            adaptive_pixscale_factor=adaptive_pixscale_factor,
        )

        bs_xy = wcsobj.world_to_pixel_values(
            bs_traj_data["ra"].degree,
            bs_traj_data["dec"].degree,
        )
        bs_xy_overhead = wcsobj.world_to_pixel_values(
            bs_traj_data["ra_overhead"].degree,
            bs_traj_data["dec_overhead"].degree,
        )
        det_xy = wcsobj.world_to_pixel_values(
            det_icrs.ra.degree,
            det_icrs.dec.degree,
        )
        # because these are bin edges, we add 1 at end to
        # makesure the nx and ny are included in the range.
        xbins = np.arange(wcsobj.pixel_shape[0] + 1)
        ybins = np.arange(wcsobj.pixel_shape[1] + 1)
        det_x_min = int(np.floor(det_xy[0].min()))
        det_x_max = int(np.ceil(det_xy[0].max()))
        if (det_x_max + 1 - det_x_min) % 2 == 1:
            det_x_max += 1
        det_y_min = int(np.floor(det_xy[1].min()))
        det_y_max = int(np.ceil(det_xy[1].max()))
        if (det_y_max + 1 - det_y_min) % 2 == 1:
            det_y_max += 1
        det_xbins = np.arange(det_x_min, det_x_max + 1)
        det_ybins = np.arange(det_y_min, det_y_max + 1)
        # note the axis order ij -> yx
        bs_im, _, _ = np.histogram2d(bs_xy[1], bs_xy[0], bins=[ybins, xbins])
        bs_im_overhead, _, _ = np.histogram2d(
            bs_xy_overhead[1], bs_xy_overhead[0], bins=[ybins, xbins]
        )
        logger.debug(f"number pixels in bs_im: {(bs_im > 0).sum()}")
        logger.debug(f"number pixels in bs_im_overhead: {(bs_im_overhead > 0).sum()}")
        # scale to coverage image of unit s / pix
        bs_im *= dt_smp.to_value(u.s)
        # generate image with only non-overhead section
        bs_im_nooverhead = bs_im.copy()
        bs_im_nooverhead[bs_im_overhead > 0] = 0
        det_im, _, _ = np.histogram2d(det_xy[1], det_xy[0], bins=[det_ybins, det_xbins])
        # convolve boresignt image with the detector image
        with timeit("convolve with array layout"):
            cov_im = convolve_fft(
                bs_im, det_im, normalize_kernel=False, allow_huge=True
            )
            cov_im_nooverhead = convolve_fft(
                bs_im_nooverhead, det_im, normalize_kernel=False, allow_huge=True
            )

        with timeit("convolve with beam"):
            a_stddev = cls.info[array_name]["a_fwhm"] / GAUSSIAN_SIGMA_TO_FWHM
            b_stddev = cls.info[array_name]["b_fwhm"] / GAUSSIAN_SIGMA_TO_FWHM
            a_stddev_pix = a_stddev.to_value(u.pix, equivalencies=pixscale)
            b_stddev_pix = b_stddev.to_value(u.pix, equivalencies=pixscale)
            if a_stddev_pix > 1:
                g = Gaussian2DKernel(
                    a_stddev_pix,
                    b_stddev_pix,
                )
                cov_im = convolve_fft(cov_im, g, normalize_kernel=False)
                cov_im_nooverhead = convolve_fft(
                    cov_im_nooverhead, g, normalize_kernel=False
                )
        logger.debug(
            f"total exp time on coverage map: "
            f"{(cov_im.sum() / det_im.sum() << u.s).to(u.min)}"
        )
        logger.debug(
            f"total time of observation: " f'{bs_traj_data["t_exp"].to(u.min)}'
        )
        cov_hdr = wcsobj.to_header()
        cov_hdr["BUNIT"] = "s / pix"
        cov_hdr.append(("ARRAYNAM", array_name, "The name of the TolTEC array"))
        cov_hdr.append(("BAND", array_name, "The name of the TolTEC array"))
        cov_hdr_nooverhead = cov_hdr.copy()
        cov_hdr.append(("EXTNAME", "cov_raw", "The name of this extension"))
        cov_hdr_nooverhead.append(
            ("EXTNAME", "cov_nooverhead", "The name of this extension")
        )
        return [
            fits.ImageHDU(data=cov_im_nooverhead, header=cov_hdr_nooverhead),
            fits.ImageHDU(data=cov_im, header=cov_hdr),
        ]

    @classmethod
    def _make_cov_hdulist(cls, ctx):
        bs_traj_data = ctx["bs_traj_data"]
        t_exp = bs_traj_data["t_exp"]
        target_alt = bs_traj_data["target_alt"]
        site_info = cls.info["site"]
        phdr = fits.Header()
        phdr.append(
            ("ORIGIN", "The TolTEC Project", "Organization generating this FITS file")
        )
        phdr.append(
            ("CREATOR", cls.__qualname__, "The software used to create this FITS file")
        )
        phdr.append(("TELESCOP", site_info["name"], site_info["name_long"]))
        phdr.append(("INSTRUME", cls.info["name"], cls.info["name_long"]))
        phdr.append(("EXPTIME", f"{t_exp.to_value(u.s):.3g}", "Exposure time (s)"))
        phdr.append(("OBSDUR", f"{t_exp.to_value(u.s):g}", "Observation duration (s)"))
        phdr.append(
            (
                "MEANALT",
                "{0:f}".format(target_alt.mean().to_value(u.deg)),
                "Mean altitude of the target during observation (deg)",
            )
        )
        hdulist = [fits.PrimaryHDU(header=phdr), *cls._make_cov_hdu_approx(ctx)]
        hdulist = fits.HDUList(hdulist)
        return hdulist
