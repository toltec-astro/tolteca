from __future__ import annotations
from tollan.utils.general import rupdate
from tollan.utils.fmt import pformat_yaml, pformat_mask
from tollan.utils.log import logger, timeit, reset_logger
from tolteca_config.core import RuntimeContext
from astropy.table import QTable
from pathlib import Path
import astropy.units as u
import plotly
import numpy as np
import pandas as pd
from tollan.utils.plot.plotly import make_range
from tolteca_config.core import ConfigHandler, ConfigModel, SubConfigKeyTransformer
from typing import Literal, TYPE_CHECKING
from pydantic import Field
from tollan.config.types import AbsDirectoryPath
from tollan.config.types import FrequencyQuantityField
from tolteca_kids.filestore import FileStoreConfigMixin
from tolteca_kids.roach_tone import TlalocEtcDataStore
from tolteca_kids.plot import PlotMixin
from tolteca_kids.match1d import Match1D, Match1DResult
from tolteca_kidsproc.kidsdata import MultiSweep
from tolteca_datamodels.toltec.file import (
    guess_info_from_sources,
)

if TYPE_CHECKING:
    from tolteca_datamodels.toltec.file import SourceInfoDataFrame


_adrv_meta_key = "atten_drive"


def _validate_inputs(tbl: SourceInfoDataFrame):
    """Return validated inputs for drive fit pipeline."""
    stbl = tbl.query("file_suffix in ['tune', 'targsweep'] & file_ext == '.nc'").copy()
    if len(stbl) == 0:
        raise ValueError("no sweep data found in inputs.")
    stbl = stbl.toltec_file.read()
    adrv_meta_key = _adrv_meta_key
    logger.debug(
        f"{len(stbl)} of {len(tbl)} sweep data found in inputs\n"
        f"{stbl.toltec_file.pformat(type='short', include_cols=[adrv_meta_key])}"
    )
    # check channels are the same
    # TODO: implement more stringent check to ensure the channel frequencies
    # are matched

    n_chans_unique = np.unique(stbl["n_chans"])
    if len(n_chans_unique) > 1:
        raise ValueError(f"inconsistent sweep channels: {n_chans_unique=}")

    roach_unique = np.unique(stbl["roach"])
    if len(roach_unique) > 1:
        raise ValueError(f"inconsistent roach: {roach_unique=}")

    n_chans = n_chans_unique[0].item()
    roach = roach_unique[0].item()
    logger.debug(f"validate input for drivefit pipeline: {roach=} {n_chans=}\n")
    return stbl


class DriveFitConfig(ConfigModel):
    """The config for drive fit."""

    n_adrvs_min: int = Field(
        default=5,
        description="Minimum number of drive atten values required.",
    )
    output_path: AbsDirectoryPath = Field(
        description="Output path.",
    )
    save_plot: bool = Field(default=False, description="Save plots.")
    weight_window_Qr: float = Field(
        default=12000.0,
        description="Use weighting window of specfiied Qr.",
    )
    a_ref: float = Field(
        default=0.2,
        description="Non-linearity parameter to use for deriving best driving power.",
    )
    a_min: float = Field(
        default=-0.2,
        description="Minimum a value used for a_best determination.",
    )
    a_max: float = Field(
        default=1.5,
        description="Maximum a value used for a_best determination.",
    )


class DriveFit(
    SubConfigKeyTransformer[Literal["drivefit"]], ConfigHandler[DriveFitConfig]
):
    """The class to perform drivefit."""

    def __call__(self, tbl: SourceInfoDataFrame):
        """Run drive fit for given dataset."""
        cfg = self.config
        tbl = _validate_inputs(tbl)
        adrvs = np.unique(tbl[_adrv_meta_key])
        n_adrvs = len(adrvs)
        n_adrvs_min = cfg.n_adrvs_min
        if n_adrvs < n_adrvs_min:
            raise ValueError(
                f"too few atten_drive steps for running drive fit: {n_adrvs=} {n_adrvs_min=}"
            )
        return self.drivefit(tbl)

    @timeit
    def drivefit(self, tbl: SourceInfoDataFrame):
        obsnum = tbl["obsnum"].min()
        roach = tbl["roach"].min()

        cfg = self.config
        output_path = cfg.output_path
        plot_save_path = output_path.joinpath(f"{obsnum}")

        kpf_cfg = {
            "load": {
                "save_dir": output_path,
                "fig_dir": plot_save_path,
            },
            "save": {
                "use_save_fig": False,
                "use_save_pdf": False and cfg.save_plot,
                "use_save_file": True,
                "save_name": "adrv",
            },
            "preview": {
                "show_plots": False,
            },
            "weight": {
                "window_Qr": cfg.weight_window_Qr,
                "weight_type": "lorentz",
            },
            "flag_settings": {
                "a_predict_guess": 0.10,
                "a_predict_threshold": 0.25,
                "pherr_threshold": 0.25,
                "pherr_threshold_num": 10,
            },
            "fit_settings": {
                "powlist_start": None,
                "powlist_end": None,
                "numspan": 1,
                # "tone_range": list(range(100, 120)),
                "tone_range": "all",
            },
        }
        from kid_phase_fit import do_toltec_drive_fit

        kpf_ctx = do_toltec_drive_fit(kpf_cfg, roach, obsnum, tbl["filepath"])
        chan_indices = kpf_ctx["tone_range"]
        all_fits = kpf_ctx["all_fits"]

        # extract fit info and interpolate to get best drivefit value
        drivefit_ctxs = {}
        for ci in chan_indices:
            fit_obj = all_fits[ci]
            drivefit_ctxs[ci] = self._find_best_adrv(fit_obj, a_ref=cfg.a_ref)
        # make result table
        kpf_file_io = kpf_ctx["files_io"][0]
        tbl_drivefit = QTable(
            rows=[
                {
                    "idx_chan": ci,
                    # TODO: make the proper mapping between power and attenuation
                    "adrv_best": -drivefit_ctxs[ci]["p_best"],
                    "a_best": drivefit_ctxs[ci]["a_best"],
                    "p_best": drivefit_ctxs[ci]["p_best"],
                    "Qr_best": drivefit_ctxs[ci]["Qr_best"],
                    "p_flag": drivefit_ctxs[ci]["p_flag"],
                    "f_chan": kpf_file_io.tone_freq_lo[ci] << u.Hz,
                    "amp_chan": kpf_file_io.tone_amps[ci],
                }
                for ci in chan_indices
            ]
        )
        drivefit_pows = next(iter(drivefit_ctxs.values()))["pows"]
        tbl_drivefit.meta.update({"drivefit_pows": drivefit_pows})

        n_success = np.sum(tbl_drivefit["p_flag"] == 0).item()
        n_chans = len(chan_indices)
        result = locals()
        if cfg.save_plot:
            self.plot_drivefit(result)
        self.save_drivefit(result)
        return result

    @classmethod
    def _find_best_adrv(
        cls,
        fit_obj,
        a_ref=DriveFitConfig.model_fields["a_ref"].default,
        a_min=DriveFitConfig.model_fields["a_min"].default,
        a_max=DriveFitConfig.model_fields["a_max"].default,
        a_tolerance=0.01,
    ):
        """This function determining the best driving power empirically."""
        a_values = np.array(fit_obj.bif_list)
        pows = np.array(fit_obj.powlist)
        fit_flags = np.array(fit_obj.fit_flag_list)
        Qr_values = np.array(fit_obj.Q_list)
        m = (fit_flags == 0) & (a_values <= a_max) & (a_values >= a_min)
        n_interp = m.sum()
        p_interp = pows[m]
        a_interp = a_values[m]
        Qr_interp = Qr_values[m]
        if n_interp == 0:
            p_best = np.nan
            a_best = np.nan
            Qr_best = np.nan
            p_flag = 1
            logger.debug(
                "no valid data point for finding best atten drive.",
            )
            return locals()
        if n_interp < 2:
            # just get the only data point
            p_best = p_interp[0]
            a_best = a_interp[0]
            Qr_best = Qr_interp[0]
            p_flag = 1
            logger.debug(
                f"only one data point found, used as best atten drive {p_best=}",
            )
            return locals()
        # sort by p and check zero crossing for nonlinearity
        isort = np.argsort(p_interp)
        p_sorted = p_interp[isort]
        a_sorted = a_interp[isort]
        Qr_sorted = Qr_interp[isort]
        p_best, y, p_flag = cls._find_first_zerocrossing1d(
            p_sorted, a_sorted - a_ref, tolerance=a_tolerance
        )
        a_best = y + a_ref
        Qr_best = np.interp(p_best, p_sorted, Qr_sorted)
        logger.debug(
            f"use {n_interp} data points, found {p_best=:f} {a_best=:f} {Qr_best=:f} {p_flag=:d}",
        )
        return locals()

    @staticmethod
    def _find_first_zerocrossing1d(x, y, tolerance=1e-5):
        # no crossing and all positive, return the first one
        if y[0] >= 0:
            return x[0], y[0], 1
        for i in range(len(y) - 1):
            y0, y1 = y[i], y[i + 1]
            x0, x1 = x[i], x[i + 1]
            if np.isclose(y0, 0, atol=tolerance):
                return x0, y0, 0
            if y0 < 0 and y1 >= 0:
                # interpolate
                dx = x1 - x0
                dy = y1 - y0
                return x0 - y0 * dx / dy, 0.0, 0
        else:
            # no crossing and all negative, return the last one
            return x[-1], y[-1], 1

    @timeit
    def save_drivefit(self, ctx):
        """Save drivefit result."""
        output_path = ctx["output_path"]
        obsnum = ctx["obsnum"]
        save_path = output_path.joinpath(f"{obsnum}")
        save_name = Path(ctx["tbl"]["filepath"].iloc[0]).stem + "_adrv.ecsv"
        FileStoreConfigMixin.save_table(
            save_path.joinpath(save_name), ctx["tbl_drivefit"]
        )

    @timeit
    def plot_drivefit(self, ctx):
        """Make plot to show the fitting."""

        save_path = ctx["plot_save_path"]
        roach = ctx["roach"]
        obsnum = ctx["obsnum"]
        if not save_path.exists():
            save_path.mkdir(parents=True)

        # grid
        fig = self._make_drivefit_grid_fig(ctx)
        save_name = f"toltec{roach}_{obsnum:06d}_drivefit_grid.html"
        FileStoreConfigMixin.save_plotly_fig(save_path.joinpath(save_name), fig)

        # summary
        fig = self._make_drivefit_summary_fig(ctx)
        save_name = f"toltec{roach}_{obsnum:06d}_drivefit_summary.html"
        FileStoreConfigMixin.save_plotly_fig(save_path.joinpath(save_name), fig)

    def _make_drivefit_summary_fig(self, ctx):
        tbl_drivefit = ctx["tbl_drivefit"]
        fig = PlotMixin.make_subplots(
            n_rows=2,
            n_cols=2,
            vertical_spacing=40 / 1200,
            fig_layout=PlotMixin.fig_layout_default
            | {
                "showlegend": False,
                "height": 1200,
            },
        )
        p_panel_kw = {"row": 1, "col": 1}
        a_panel_kw = {"row": 1, "col": 2}
        fr_panel_kw = {"row": 2, "col": 1}
        Qr_panel_kw = {"row": 2, "col": 2}

        tbl_good = tbl_drivefit[tbl_drivefit["p_flag"] == 0]
        tbl_bad = tbl_drivefit[
            (tbl_drivefit["p_flag"] == 1) & (~np.isnan(tbl_drivefit["p_best"]))
        ]
        c00, c100 = plotly.colors.sample_colorscale(
            "rdylgn",
            samplepoints=[0, 1],
        )

        # p hist
        drivefit_pows = ctx["drivefit_pows"]
        p_step = 0.25
        p_bins = np.arange(
            np.min(drivefit_pows), np.max(drivefit_pows) + p_step, p_step
        )
        p_centers = 0.5 * (p_bins[1:] + p_bins[:-1])
        for t, name, color in [
            (tbl_bad, "bad", c00),
            (tbl_good, "good", c100),
        ]:
            fig.add_bar(
                x=p_centers,
                y=np.histogram(t["p_best"], bins=p_bins)[0],
                marker={
                    "color": color,
                },
                name=name,
                **p_panel_kw,
            )
        fig.update_yaxes(
            title="Count",
            **p_panel_kw,
        )
        fig.update_xaxes(
            title="p_best",
            **p_panel_kw,
        )

        # a hist
        a_bins = np.arange(-0.5, 1, 0.1)
        a_centers = 0.5 * (a_bins[1:] + a_bins[:-1])
        for t, name, color in [
            (tbl_bad, "bad", c00),
            (tbl_good, "good", c100),
        ]:
            fig.add_bar(
                x=a_centers,
                y=np.histogram(t["a_best"], bins=a_bins)[0],
                marker={
                    "color": color,
                },
                name=name,
                **a_panel_kw,
            )
        fig.update_yaxes(
            title="Count",
            **a_panel_kw,
        )
        fig.update_xaxes(
            title="a_best",
            **a_panel_kw,
        )
        fig.update_layout(barmode="stack")

        # fr vs p_best
        for t, name, color in [
            (tbl_bad, "bad", c00),
            (tbl_good, "good", c100),
        ]:
            fig.add_scattergl(
                x=t["f_chan"].to_value(u.MHz),
                y=t["p_best"],
                mode="markers",
                marker={
                    "color": color,
                },
                name=name,
                **fr_panel_kw,
            )
        fig.update_yaxes(
            title="p_best",
            **fr_panel_kw,
        )
        fig.update_xaxes(
            title="f_chan (MHz)",
            **fr_panel_kw,
        )

        # Qr vs p_best
        for t, name, color in [
            (tbl_bad, "bad", c00),
            (tbl_good, "good", c100),
        ]:
            fig.add_scattergl(
                x=t["Qr_best"],
                y=t["p_best"],
                mode="markers",
                marker={
                    "color": color,
                },
                name=name,
                **Qr_panel_kw,
            )
        fig.update_yaxes(
            title="p_best",
            **Qr_panel_kw,
        )
        fig.update_xaxes(
            title="Qr",
            **Qr_panel_kw,
        )
        obsnum = ctx["obsnum"]
        fig.update_layout(
            title={
                "text": f"DriveFit summary: {obsnum:d}",
            },
            margin={
                "t": 100,
                "b": 100,
            },
        )
        return fig

    def _make_drivefit_grid_fig(self, ctx):
        drivefit_ctxs = ctx["drivefit_ctxs"]
        drivefit_pows = ctx["drivefit_pows"]
        chan_indices = ctx["chan_indices"]
        n_chans = len(chan_indices)
        n_pows = len(drivefit_pows)

        n_rows = 7
        n_cols = n_pows + 1

        color_cycle = PlotMixin.color_palette.cycle()

        def logmag(z):
            return 20.0 * np.log10(np.abs(z))

        def _make_drivefit_panel(row, col, panel_id, dummy=False, init=False):
            i = panel_id // n_cols
            pi = panel_id % n_cols
            if pi == n_cols - 1:
                # last panel is the adrv summary plot
                pi = None
            if i >= n_chans:
                x = []
                y = []
                color = None
                name = None
            else:
                ci = chan_indices[i]
                drivefit_ctx = drivefit_ctxs[ci]
                p_best = drivefit_ctx["p_best"]
                a_best = drivefit_ctx["a_best"]
                fit_obj = drivefit_ctx["fit_obj"]
                color = next(color_cycle)
                pp = drivefit_pows[pi]
                if pi is not None:
                    if pp in fit_obj.z1:
                        x = fit_obj.f[pp]
                        y = logmag(fit_obj.z1[pp])
                    else:
                        x = []
                        y = []
                else:
                    x = drivefit_ctx["p_interp"]
                    y = drivefit_ctx["a_interp"]
                name = f"chan {ci} pow {pi}"
            trace = {
                "type": "scatter",
                "x": x,
                "y": y,
                "name": name,
                "marker": {
                    "color": color,
                },
            }
            if init:
                rupdate(
                    trace,
                    {
                        "mode": "markers",
                        "marker": {
                            "size": 12 if pi is None else 4,
                        },
                    },
                )
            data = [trace]
            # add best adrv values in summary panel
            if i < n_chans and pi is None and p_best is not None:
                data.append(
                    {
                        "type": "scatter",
                        "x": [p_best],
                        "y": [a_best],
                        "mode": "markers",
                        "marker": {
                            "size": 16,
                            "symbol": "x",
                            "color": "rgba(0, 0, 0, 0)",
                            "line": {"color": "red", "width": 2},
                        },
                    }
                )
            # text annos
            if i < n_chans:
                if pi is None:
                    text_lns = [
                        f"p_flag={drivefit_ctx['p_flag']}",
                    ]
                    if p_best is not None and a_best is not None:
                        text_lns.extend(
                            [
                                f"a_best={drivefit_ctx['a_best']:.2f}",
                                f"p_best={drivefit_ctx['p_best']:.2f}",
                            ]
                        )
                else:
                    text_lns = []
                    if row == 1:
                        text_lns.append(f"pow: {drivefit_pows[pi]}")
                    if col == 1:
                        text_lns.append(f"chan: {ci}")
                    if pp in fit_obj.result:
                        fit_flag = fit_obj.fit_flag[pp]
                        a = fit_obj.result[pp].result["bif"]
                        text_lns.extend(
                            [
                                f"fit_flag: {fit_flag}",
                                f"a: {a:.2g}",
                            ]
                        )
                    else:
                        text_lns.append("fit_flag: 1")
                if text_lns:
                    annos = [
                        {
                            "x": 0.4,
                            "y": 0.8,
                            "xref": "x domain",
                            "yref": "y domain",
                            "text": "<br>".join(text_lns),
                            "align": "left",
                            "valign": "top",
                        }
                    ]
                else:
                    annos = []
            else:
                annos = []
            if dummy:
                xrange = [0, 1]
                yrange = [0, 1]
            else:
                x = data[0]["x"]
                y = data[0]["y"]
                if len(x) == 0:
                    xrange = [0, 1]
                    yrange = [0, 1]
                else:
                    xrange = make_range(data[0]["x"])
                    yrange = make_range(data[0]["y"])
            layout = {
                "xaxis": {
                    "range": xrange,
                },
                "yaxis": {
                    "range": yrange,
                },
            }
            if init:
                if row == n_rows and col == 1:
                    rupdate(
                        layout,
                        {
                            "xaxis": {
                                "title": {
                                    "text": "f (Hz)",
                                },
                            },
                            "yaxis": {
                                "title": {
                                    "text": "S21 (dB)",
                                },
                            },
                        },
                    )
                if row == n_rows and col == n_cols:
                    rupdate(
                        layout,
                        {
                            "xaxis": {
                                "title": {
                                    "text": "Drive Power (dB)",
                                },
                            },
                            "yaxis": {
                                "title": {
                                    "text": "Non-linearity",
                                },
                            },
                        },
                    )
            rupdate(
                layout,
                {
                    "annotations": annos,
                },
            )
            return {
                "data": data,
                "layout": layout,
            }

        n_items = n_chans * n_cols
        obsnum = ctx["obsnum"]
        fig = PlotMixin.make_data_grid_anim(
            name=f"Drive Fit Details: {obsnum:d}",
            n_rows=n_rows,
            n_cols=n_cols,
            n_items=n_items,
            make_panel_func=_make_drivefit_panel,
            frame_name_func=lambda s, e: chan_indices[s // n_cols],
            redraw=False,
            subplot_titles=None,
        )
        fig.update_layout(
            margin={
                "t": 100,
                "b": 100,
            }
        )
        return fig


class DriveFitCommitConfig(ConfigModel):
    """The config for drive fit."""

    match: Match1D = Field(
        default={
            "method": "dtw_python",
        },
        description="tone matching settings.",
    )
    match_shift_max: FrequencyQuantityField = Field(
        default=10 << u.MHz,
        description="The maximum shift allowed in match.",
    )
    adrv_max: float = Field(
        default=30,
        description="The maximum adrv allowed",
    )
    adrv_fill_perc: int = Field(
        default=50, description="The percentile to compute fill value for missing adrv."
    )
    adrv_ref_perc: int = Field(
        default=10, description="The percentile to compute global reference adrv."
    )
    adrv_ref_step: float = Field(
        default=0.25,
        description="The global reference adrv step.",
    )
    output_path: AbsDirectoryPath = Field(
        description="Output path.",
    )
    tlaloc_etc_path: None | AbsDirectoryPath = Field(
        description="Tlaloc etc path.",
    )

    save_plot: bool = Field(default=False, description="Save plots.")


class DriveFitCommit(
    SubConfigKeyTransformer[Literal["drivefit_commit"]],
    ConfigHandler[DriveFitCommitConfig],
):
    """The class to perform drivefit commit."""

    def __call__(self, tbl: SourceInfoDataFrame, tbl_drivefit: QTable):
        """Run drive fit commit for given dataset."""
        tbl = _validate_inputs(tbl)
        return self.drivefit_commit(tbl, tbl_drivefit)

    @timeit
    def drivefit_commit(self, tbl: SourceInfoDataFrame, tbl_drivefit: QTable):
        cfg = self.config
        swp: MultiSweep = tbl.toltec_file.read().toltec_file.data_objs.iloc[0]
        tbl_chans = swp.meta["chan_axis_data"]
        s21_f = swp.frequency
        s21_f_step = s21_f[0, 1] - s21_f[0, 0]

        logger.debug(f"run drive fit commit for {swp}")

        def _match_postproc(r: Match1DResult):
            matched = r.matched
            # current chan info
            iq = matched["idx_query"]
            matched["idx_chan"] = tbl_chans["id"][iq]
            matched["f_chan"] = tbl_chans["f_chan"][iq]
            matched["amp_chan"] = tbl_chans["amp_tone"][iq]

            # drivefit chan info
            ir = matched["idx_ref"]
            matched["idx_chan_adrv"] = tbl_drivefit["idx_chan"][ir]
            matched["flag_adrv"] = tbl_drivefit["p_flag"][ir]
            matched["f_chan_adrv"] = tbl_drivefit["f_chan"][ir]
            matched["amp_chan_adrv"] = tbl_drivefit["amp_chan"][ir]
            matched["adrv_best_adrv"] = tbl_drivefit["adrv_best"][ir]
            matched["a_best_adrv"] = tbl_drivefit["a_best"][ir]
            Qr = matched["Qr_adrv"] = tbl_drivefit["Qr_best"][ir]
            rr = 0.5 / Qr
            xx = matched["dist"] / matched["ref"]
            matched["d_phi"] = np.rad2deg(np.arctan2(xx, rr)) << u.deg
            return r

        matched = cfg.match(
            query=tbl_chans["f_chan"].to(u.MHz),
            ref=tbl_drivefit["f_chan"].to(u.MHz),
            postproc_hook=_match_postproc,
            shift_kw={
                "shift_max": cfg.match_shift_max,
                "dx": s21_f_step / 2,
            },
        )
        tbl_adrv = tbl_chans.copy()
        for colname in [
            "dist",
            "d_phi",
            "idx_chan_adrv",
            "flag_adrv",
            "a_best_adrv",
            "adrv_best_adrv",
            "f_chan_adrv",
            "amp_chan_adrv",
            "Qr_adrv",
        ]:
            tbl_adrv[colname] = matched.data["query_matched"][colname]
        logger.debug(f"tbl_adrv:\n{tbl_adrv}")

        ctx_ampcor = self.compute_ampcor(tbl_adrv)
        tbl_adrv["adrv_ref"] = ctx_ampcor["adrv_ref"]
        tbl_adrv["adrv_ref_total"] = ctx_ampcor["adrv_ref_total"]
        tbl_adrv["adrv_best"] = ctx_ampcor["adrv_filled"]
        tbl_adrv["amp_best"] = ctx_ampcor["ampcor"]
        tbl_adrv["amp_flag"] = ~ctx_ampcor["m_good"]
        tbl_adrv.meta.clear()
        tbl_adrv.meta.update(tbl_drivefit.meta)
        tbl_adrv.meta["atten_drive_global"] = ctx_ampcor["adrv_ref_total"]

        obsnum = swp.meta["obsnum"]
        roach = swp.meta["roach"]

        output_path = cfg.output_path
        plot_save_path = output_path.joinpath(f"{obsnum}")

        n_success = ctx_ampcor["m_good"].sum().item()
        n_chans = len(tbl_adrv)
        result = locals()
        if cfg.save_plot:
            self.plot_drivefit_commit(result)
        self.save_drivefit_commit(result)
        return result

    def compute_ampcor(self, tbl_adrv):
        """Return ampcor table."""
        cfg = self.config
        adrv_ref_perc = cfg.adrv_ref_perc
        adrv_ref_step = cfg.adrv_ref_step
        adrv_fill_perc = cfg.adrv_fill_perc

        m_good = (tbl_adrv["flag_adrv"] == 0) & (
            tbl_adrv["adrv_best_adrv"] < cfg.adrv_max
        )
        adrv_values = tbl_adrv["adrv_best_adrv"][m_good][:]
        # make sure they are all positive
        adrv_values[adrv_values < 0] = 0.0

        adrv_ref = np.percentile(adrv_values, adrv_ref_perc).item()
        # round to adrv ref step
        # adrv_ref = (np.floor(adrv_ref / adrv_ref_step) * adrv_ref_step).item()

        # generate fill values
        adrv_fill = np.percentile(adrv_values, adrv_fill_perc).item()
        adrv_filled = tbl_adrv["adrv_best_adrv"][:]
        adrv_filled[m_good] = adrv_values
        adrv_filled[~m_good] = adrv_fill
        # cut all values less then adrv_ref
        m_underdrive = adrv_filled < adrv_ref
        adrv_filled[m_underdrive] = adrv_ref
        logger.info(f"underdrive mask: {pformat_mask(m_underdrive)}")

        # make ampcor, which should take into account the drivefit LUT.
        ampcor_lut = tbl_adrv["amp_chan_adrv"] / tbl_adrv["amp_chan_adrv"].max()
        ampcor_adjust = 10 ** ((adrv_ref - adrv_filled) / 20.0)
        ampcor = ampcor_lut * ampcor_adjust
        # the new ampcor will have compressed dynamic range, which after re-scale, requires
        # extra adrv_ref
        ampcor_norm = np.max(ampcor)
        adrv_ref_extra = -20.0 * np.log10(ampcor_norm)
        ampcor = ampcor / ampcor_norm
        adrv_ref_total = adrv_ref_extra + adrv_ref

        # finally, round to adrv ref step
        adrv_ref_total = (
            np.floor(adrv_ref_total / adrv_ref_step) * adrv_ref_step
        ).item()
        logger.info(
            f"found {adrv_ref_total=} {adrv_ref=} {adrv_fill=} from {pformat_mask(m_good)} adrv values"
        )
        return locals()

    def plot_drivefit_commit(self, ctx):
        save_path = ctx["plot_save_path"]
        roach = ctx["roach"]
        obsnum = ctx["obsnum"]
        if not save_path.exists():
            save_path.mkdir(parents=True)

        fig = self._make_drivefit_commit_fig(ctx)
        save_name = f"toltec{roach}_{obsnum:06d}_drivefit_commit.html"
        FileStoreConfigMixin.save_plotly_fig(save_path.joinpath(save_name), fig)

    def _make_drivefit_commit_fig(self, ctx):
        matched = ctx["matched"]
        tbl_adrv = ctx["tbl_adrv"]
        fig = PlotMixin.make_subplots(
            n_rows=5,
            n_cols=1,
            vertical_spacing=40 / 1200,
            fig_layout=PlotMixin.fig_layout_default
            | {
                "showlegend": False,
                "height": 1200,
            },
        )
        # a_panel_kw = {"row": 1, "col": 1}
        fr_panel_kw = {"row": 2, "col": 1}
        Qr_panel_kw = {"row": 3, "col": 1}
        match_panel_kw = {"row": 4, "col": 1}
        match_density_panel_kw = {"row": 5, "col": 1}

        tbl_good = tbl_adrv[tbl_adrv["amp_flag"] == 0]
        tbl_bad = tbl_adrv[tbl_adrv["amp_flag"] != 0]
        c00, c100 = plotly.colors.sample_colorscale(
            "rdylgn",
            samplepoints=[0, 1],
        )

        # adrv hist
        # adrv_bins = np.arange(
        #     0, cfg.adrv_max + cfg.adrv_ref_step, cfg.adrv_ref_step,
        # )
        # adrv_centers = 0.5 * (adrv_bins[1:] + adrv_bins[:-1])
        for t, name, color in [
            (tbl_bad, "bad", c00),
            (tbl_good, "good", c100),
        ]:
            fig.add_scattergl(
                x=t["f_chan"].to_value(u.MHz),
                y=t["amp_best"],
                mode="markers",
                marker={
                    "color": color,
                },
                name=name,
                **fr_panel_kw,
            )
        fig.update_yaxes(
            title="Amp Cor.",
            **fr_panel_kw,
        )
        fig.update_xaxes(
            title="f_chan (MHz)",
            **fr_panel_kw,
        )

        for t, name, color in [
            (tbl_bad, "bad", c00),
            (tbl_good, "good", c100),
        ]:
            fig.add_scattergl(
                x=t["Qr_adrv"],
                y=t["amp_best"],
                mode="markers",
                marker={
                    "color": color,
                },
                name=name,
                **Qr_panel_kw,
            )
        fig.update_yaxes(
            title="Amp Cor.",
            **Qr_panel_kw,
        )
        fig.update_xaxes(
            title="Qr",
            **Qr_panel_kw,
        )

        matched.make_plotly_fig(
            type="match",
            fig=fig,
            panel_kw=match_panel_kw,
            label_value="Frequency (MHz)",
            label_ref="Drive Fit",
            label_query="Current Sweep",
        )
        matched.make_plotly_fig(
            type="density",
            fig=fig,
            panel_kw=match_density_panel_kw,
            label_ref="Ref Id (Drive Fit)",
            label_query="Current Sweep Channel Id",
        )
        obsnum = ctx["obsnum"]
        fig.update_layout(
            title={
                "text": f"DriveFit summary: {obsnum:d}",
            },
            margin={
                "t": 100,
                "b": 100,
            },
        )

        return fig

    def save_drivefit_commit(self, ctx):
        """Save drivefit commit result."""
        output_path = ctx["output_path"]
        obsnum = ctx["obsnum"]
        save_path = output_path.joinpath(f"{obsnum}")
        save_name = Path(ctx["tbl"]["filepath"].iloc[0]).stem + "_adrv_commit.ecsv"
        tbl_adrv = ctx["tbl_adrv"]
        FileStoreConfigMixin.save_table(save_path.joinpath(save_name), tbl_adrv)

        tlaloc_etc_path = self.config.tlaloc_etc_path
        if tlaloc_etc_path is None:
            return
        tlaloc_etc = TlalocEtcDataStore(tlaloc_etc_path)
        roach = ctx["roach"]
        tbl_amp = QTable({"amp_tone": tbl_adrv["amp_best"]})
        tbl_amp.meta["roach"] = roach
        tlaloc_etc.write_targ_amps_table(tbl_amp)
        tlaloc_etc.write_atten_drive_value(roach, tbl_adrv.meta["atten_drive_global"])


def _run(
    rc: RuntimeContext,
    tbl: SourceInfoDataFrame,
    step_name: str,
    step_cls: type[ConfigHandler],
    step_kw: None | dict = None,
):
    roach_unique = np.unique(tbl["roach"])
    if len(roach_unique) > 1:
        message = "inconsistent roach"
        return 1, locals()
    roach = roach_unique[0].item()
    with rc.config_backend.set_context({"roach": roach}):
        step = step_cls(rc)
        logger.debug(f"{step_name} config:\n{step.config.model_dump_yaml()}")
        try:
            ctx = step(tbl, **(step_kw or {}))
        except Exception as e:
            logger.opt(exception=True).error(f"{step_name} failed for {roach=}: {e}")
            returncode = 1
            message = f"{step_name} failed: {e}"
        else:
            n_success = ctx["n_success"]
            n_chans = ctx["n_chans"]
            message = f"{step_name} {n_success=} {n_chans=}"
            returncode = 0
        return returncode, locals()


def run_drivefit_pipeline(
    rc: RuntimeContext,
    tbl: SourceInfoDataFrame,
    drivefit_output_search_paths: None | list = None,
    mode: Literal[None, "auto", "drivefit", "drivefit_commit"] = "auto",
):
    mode = mode or "auto"
    tbl = _validate_inputs(tbl)
    adrvs = np.unique(tbl[_adrv_meta_key])
    n_adrvs = len(adrvs)

    def _run_drivefit(tbl):
        return _run(rc, tbl, "drive fit", DriveFit)

    def _run_drivefit_commit(tbl):
        returncode_drivefit, ctx_drivefit = _load_drivefit_data(
            tbl, drivefit_output_search_paths=drivefit_output_search_paths
        )
        if returncode_drivefit != 0:
            return returncode_drivefit, ctx_drivefit
        return _run(
            rc,
            tbl,
            "drive fit commit",
            DriveFitCommit,
            step_kw={"tbl_drivefit": ctx_drivefit["tbl_drivefit"]},
        )

    if n_adrvs > 1:
        # run drivefit
        if mode in ["auto", "drivefit"]:
            logger.info("found multiple atten_drives, run drive fit")
            return _run_drivefit(tbl)
        if mode == "drivefit_commit":
            adrv0 = adrvs[0]
            tbl = tbl.query(f"atten_drive == {adrv0}")
            logger.info(
                f"found multiple atten_drives, run {mode=} with first adrv={adrv0}"
            )
            return _run_drivefit_commit(tbl)
        message = f"invalid mode {mode=}"
    elif n_adrvs == 1:
        if mode in ["auto", "drivefit_commit"]:
            logger.info(
                "found single sweep, run drive fit commit with previous drive fit data"
            )
            return _run_drivefit_commit(tbl)
        if mode == "drivefit":
            logger.error(f"found single sweep, unable to run {mode=}")
        message = f"invalid mode {mode=}"
    else:
        message = "invalid input for drive fit pipeline"
    return 1, locals()


def run_drivefit_quicklook(
    rc: RuntimeContext,
    ctxs: list[dict],
):
    n_ctxs = len(ctxs)
    fig = PlotMixin.make_subplots(
        n_rows=n_ctxs,
        n_cols=4,
        vertical_spacing=80 / 1200,
        fig_layout=PlotMixin.fig_layout_default
        | {
            "showlegend": False,
            "height": 2000,
        },
    )
    for i, _ctx in enumerate(ctxs):
        ctx = _ctx["ctx"]
        if "matched" not in ctx:
            continue
        matched = ctx["matched"]
        tbl_adrv = ctx["tbl_adrv"]
        row = i + 1
        a_panel_kw = {"row": row, "col": 1}
        fr_panel_kw = {"row": row, "col": 2}
        Qr_panel_kw = {"row": row, "col": 3}
        match_density_panel_kw = {"row": row, "col": 4}

        tbl_good = tbl_adrv[tbl_adrv["amp_flag"] == 0]
        tbl_bad = tbl_adrv[tbl_adrv["amp_flag"] != 0]
        c00, c100 = plotly.colors.sample_colorscale(
            "rdylgn",
            samplepoints=[0, 1],
        )

        # adrv hist
        # adrv_bins = np.arange(
        #     0, cfg.adrv_max + cfg.adrv_ref_step, cfg.adrv_ref_step,
        # )
        # adrv_centers = 0.5 * (adrv_bins[1:] + adrv_bins[:-1])
        # a hist
        a_bins = np.arange(-0.5, 1, 0.1)
        a_centers = 0.5 * (a_bins[1:] + a_bins[:-1])
        for t, name, color in [
            (tbl_bad, "bad", c00),
            (tbl_good, "good", c100),
        ]:
            fig.add_bar(
                x=a_centers,
                y=np.histogram(t["a_best_adrv"], bins=a_bins)[0],
                marker={
                    "color": color,
                },
                name=name,
                **a_panel_kw,
            )
        roach = ctx["roach"]
        adrv_global = tbl_adrv.meta["atten_drive_global"]
        fig.update_yaxes(
            title=f"nw{roach} {adrv_global}",
            **a_panel_kw,
        )
        fig.update_xaxes(
            title="Non-linearity",
            **a_panel_kw,
        )
        fig.update_layout(barmode="stack")

        for t, name, color in [
            (tbl_bad, "bad", c00),
            (tbl_good, "good", c100),
        ]:
            fig.add_scattergl(
                x=t["f_chan"].to_value(u.MHz),
                y=t["amp_best"],
                mode="markers",
                marker={
                    "color": color,
                },
                name=name,
                **fr_panel_kw,
            )
        fig.update_yaxes(
            title="Amp Cor.",
            **fr_panel_kw,
        )
        fig.update_xaxes(
            title="f_chan (MHz)",
            **fr_panel_kw,
        )

        for t, name, color in [
            (tbl_bad, "bad", c00),
            (tbl_good, "good", c100),
        ]:
            fig.add_scattergl(
                x=t["Qr_adrv"],
                y=t["amp_best"],
                mode="markers",
                marker={
                    "color": color,
                },
                name=name,
                **Qr_panel_kw,
            )
        fig.update_yaxes(
            title="Amp Cor.",
            **Qr_panel_kw,
        )
        fig.update_xaxes(
            title="Qr",
            **Qr_panel_kw,
        )

        matched.make_plotly_fig(
            type="density",
            fig=fig,
            panel_kw=match_density_panel_kw,
            label_ref="Ref Id (Drive Fit)",
            label_query="Current Sweep Channel Id",
        )
        obsnum = ctx["obsnum"]
        fig.update_layout(
            title={
                "text": f"DriveFit summary: {obsnum:d}",
            },
            margin={
                "t": 100,
                "b": 100,
            },
        )
    obsnum = ctxs[0]["ctx"]["obsnum"]
    save_path = ctxs[0]["ctx"]["plot_save_path"]
    if not save_path.exists():
        save_path.mkdir(parents=True)
    save_name = f"ql_{obsnum:06d}_drivefit_commit.html"
    FileStoreConfigMixin.save_plotly_fig(save_path.joinpath(save_name), fig)
    return fig


def _load_drivefit_data(
    tbl: SourceInfoDataFrame, drivefit_output_search_paths: None | list
):
    swp: MultiSweep = tbl.toltec_file.read().toltec_file.data_objs.iloc[0]
    logger.debug(f"locate drivefit data for {swp.meta["uid_raw_obs_file"]}")
    roach = swp.meta["roach"]
    file_timestamp = swp.meta["file_timestamp"]
    drivefit_files = []
    for p in drivefit_output_search_paths or []:
        drivefit_files.extend(p.glob(f"*/toltec{roach}_*_*_adrv.ecsv"))

    if not drivefit_files:
        message = f"no drivefit data found for {swp.meta["uid_raw_obs_file"]}"
        logger.error(message)
        return 1, locals()
    tbl_drivefit_files = guess_info_from_sources(drivefit_files)
    logger.debug(f"found drivefit files:\n{tbl_drivefit_files.toltec_file.pformat()}")
    i_closest = np.argmin(
        np.abs(tbl_drivefit_files["file_timestamp"] - file_timestamp)
    ).item()
    logger.debug(f"locate drive fit data {i_closest=}")
    tbl_drivefit_file = tbl_drivefit_files.iloc[i_closest]["filepath"]
    tbl_drivefit = QTable.read(tbl_drivefit_file, format="ascii.ecsv")
    logger.info(f"use drivefit table: {tbl_drivefit_file}")
    logger.debug(f"drivefit table:\n{tbl_drivefit}")
    return 0, locals()


if __name__ == "__main__":
    import sys
    import argparse
    import numpy as np
    from toltec_file_utils import LmtToltecPathOption
    from tollan.utils.cli import dict_from_cli_args, split_cli_args

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="tolteca.yaml")
    parser.add_argument("--log_level", default="INFO", help="The log level.")
    parser.add_argument("--save_plot", action="store_true", help="Save plots.")
    parser.add_argument("--select", default=None, help="Select input data.")
    parser.add_argument(
        "--mode", choices=["auto", "drivefit", "drivefit_commit"], default="auto"
    )
    LmtToltecPathOption.add_args_to_parser(
        parser, obs_spec_required=True, obs_spec_multi=True
    )

    drivefit_cli_args, args = split_cli_args(r"^drivefit(_commit)?\..+", sys.argv[1:])
    logger.debug(
        f"drivefit_cli_args:\n{pformat_yaml(drivefit_cli_args)}\n"
        f"other_args:\n{pformat_yaml(args)}",
    )
    option = parser.parse_args(args)
    reset_logger(level=option.log_level)
    logger.debug(f"parsed options: {option}")
    path_option = LmtToltecPathOption(option)

    tbl = path_option.get_raw_obs_info_table(raise_on_empty=True).sort_values("roach")
    if option.select is not None:
        with tbl.toltec_file.open() as fo:
            tbl = fo.query(option.select)
    drivefit_cli_args[:0] = [
        "--drivefit.output_path",
        path_option.dataprod_path,
        "--drivefit_commit.output_path",
        path_option.dataprod_path,
    ]

    if option.save_plot:
        drivefit_cli_args.extend(
            [
                "--drivefit.save_plot",
                "--drivefit_commit.save_plot",
            ],
        )
    if option.tlaloc_etc_path is not None:
        drivefit_cli_args.extend(
            ["--drivefit_commit.tlaloc_etc_path", option.tlaloc_etc_path]
        )
    rc = RuntimeContext(option.config)
    rc.config_backend.update_override_config(dict_from_cli_args(drivefit_cli_args))
    logger.info(f"{pformat_yaml(rc.config.model_dump())}")

    drivefit_output_search_paths = [path_option.dataprod_path]
    with timeit("run drive fit pipeline"):
        report = []
        ctxs = []
        for roach, stbl in tbl.groupby("roach", sort=False):
            logger.debug(f"{roach=} n_files={len(stbl)}")
            r, ctx = run_drivefit_pipeline(
                rc,
                stbl,
                drivefit_output_search_paths=drivefit_output_search_paths,
                mode=option.mode,
            )
            report.append(
                {
                    "roach": roach,
                    "n_items": len(stbl),
                    "returncode": r,
                    "message": ctx.get("message", None),
                }
            )
            ctxs.append(ctx)
    if option.save_plot:
        with timeit("generate drive fit quicklook"):
            run_drivefit_quicklook(rc, ctxs)
    report = pd.DataFrame.from_records(report)
    logger.info(f"run status:\n{report}")
    n_failed = np.sum(report["returncode"] != 0)
    if n_failed == 0:
        logger.info("Job's done!")
    else:
        logger.error("Job's failed.")
    sys.exit(n_failed)
