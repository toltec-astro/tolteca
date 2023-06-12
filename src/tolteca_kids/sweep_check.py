from enum import IntFlag, auto
from typing import cast

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from pydantic import Field
from scipy.ndimage import median_filter
from tollan.utils.log import logger, timeit
from functools import lru_cache

from tolteca_kidsproc.kidsdata import MultiSweep

from .mpl import KidsPlotter, plot_axvlines
from .utils import WorkflowStepBase

__all__ = ["SweepChecker"]


class SweepDataBitMask(IntFlag):
    """A bit mask for sweep data."""

    s21_high_rms = auto()
    s21_small_range = auto()
    s21_spike = auto()


class DespikeStep(WorkflowStepBase):
    """Despike step."""

    min_spike_height_frac: float = Field(
        default=0.1,
        description=(
            "The minimum range of spike, measured as fraction to the S21 range."
        ),
    )
    min_S21_range_db: float = Field(
        default=0.1,
        description="The minimum S21 range in dB to find spike.",
    )

    plotter_kw: dict = Field(
        default_factory=dict,
        description="options passed to plotter.",
    )

    @staticmethod
    def calc_y(swp):
        """Return the proxy value to run the algorithms on."""
        S21 = swp.S21.to_value(u.adu)
        return 20.0 * np.log10(np.abs(S21))

    @lru_cache
    @classmethod
    def _find_spike_S21(cls, swp: MultiSweep, min_spike_height_frac, min_S21_range_db):
        y = cls.calc_y(swp)
        y_med = median_filter(y, size=(1, 5))
        y_range = np.max(y_med, axis=-1) - np.min(y_med, axis=-1)
        s_spike = spike_height_frac = (y - y_med) / y_range[:, np.newaxis]
        m_data_spike = np.abs(s_spike) >= min_spike_height_frac
        m_chan_small_range = y_range < min_S21_range_db
        m_data_small_range = m_chan_small_range[:, np.newaxis]
        bitmask = (SweepDataBitMask.s21_small_range * m_data_small_range) | (
            SweepDataBitMask.s21_spike * m_data_spike
        )
        logger.debug(
            (
                "found low signal range channel:"
                f" {m_chan_small_range.sum()}/{m_chan_small_range.size}"
            ),
        )
        logger.debug(f"found spike {m_data_spike.sum()}/{m_data_spike.size}")
        # create spike mask, which is all spikes found in good channel.
        mask = m_data_spike & (~m_data_small_range)
        logger.debug(f"masked spike {mask.sum()}/{mask.size}")
        return locals()

    def find_spike_S21(self, arg):
        """Find spike in S21."""
        swp = cast(MultiSweep, self.ensure_input_data(arg))
        return self._mark_sub_context(
            self._find_spike_S21(
                swp,
                min_spike_height_frac=self.min_spike_height_frac,
                min_S21_range_db=self.min_S21_range_db,
            ),
        )

    @lru_cache
    @classmethod
    def _despike(cls, swp: MultiSweep, min_spike_height_frac, min_S21_range_db):
        ctx_spike = cls._find_spike_S21(
            swp,
            min_spike_height_frac=min_spike_height_frac,
            min_S21_range_db=min_S21_range_db,
        )
        swp = ctx_spike["swp"]
        spike_mask = ctx_spike["mask"]
        goodmask = ~spike_mask
        fs_Hz = swp.frequency.to_value(u.Hz)
        S21_adu = swp.S21.to_value(u.adu).copy()
        for ci in range(fs_Hz.shape[0]):
            m = goodmask[ci]
            swp.S21[ci] = np.interp(fs_Hz[ci], fs_Hz[ci][m], S21_adu[ci][m]) << u.adu
        # make despiked y
        y_nospike = cls.calc_y(swp)
        return locals()

    def despike(self, arg):
        """Apply spike mask on data."""
        swp = cast(MultiSweep, self.ensure_input_data(arg))
        return self._mark_sub_context(
            self._despike(
                swp,
                min_spike_height_frac=self.min_spike_height_frac,
                min_S21_range_db=self.min_S21_range_db,
            ),
        )

    @timeit
    def run(self, arg, parent_workflow=None):
        """Return sweep data that have spikes identified and interpolated away."""
        self._mark_start(parent_workflow=parent_workflow)
        if self._should_skip():
            return self._mark_skipped()
        # run
        self.despike(
            self.find_spike_S21(arg),
        )
        ctx_spike = self._get_sub_context(self.find_spike_S21)
        ctx_despike = self._get_sub_context(self.despike)
        # attach the despike context to data meta
        swp = ctx_despike["swp"]
        swp.meta[self._make_sub_context_key()] = self.context  # type: ignore
        return self._mark_success(locals())

    def plot(self, save_filepath=None):  # noqa: PLR0915
        """Make plot."""
        if self.run_state is None:
            raise ValueError("nothing to plot, step is not run.")
        if self.input is None or self.input.data is None:
            raise ValueError("nothing to plot, no valid input")

        fig = plt.figure()
        gs0 = gridspec.GridSpec(2, 2, figure=fig)
        gs1 = gs0[:, 0].subgridspec(2, 1)
        kph = KidsPlotter(
            figure=fig,
            gs_nav=gs1[0, 0],
            gs_grid=gs1[1, 0],
            **self.plotter_kw,
        )
        ctx = self.context
        swp = ctx["swp"]
        ctx_spike = ctx["ctx_spike"]
        ctx_despike = ctx["ctx_despike"]

        kph.plot_sweep_chan_nav(swp)

        # plot flags
        ax_chan_flag = fig.add_subplot(gs0[0, 1])
        ax_data_flag = fig.add_subplot(gs0[1, 1])
        bm = ctx_spike["bitmask"]
        m_spike = bm & SweepDataBitMask.s21_spike
        m_small_range = bm & SweepDataBitMask.s21_small_range
        n_chan, n_sweepsteps = swp.frequency.shape
        chan_id = np.arange(n_chan)
        chan_id_small_range = np.where(np.any(m_small_range, axis=1))[0]
        plot_axvlines(
            ax_chan_flag,
            chan_id_small_range,
            label="low sig chan",
            color="gray",
            linestyle="--",
        )
        ax_chan_flag.plot(
            chan_id,
            m_spike.sum(axis=1),
            drawstyle="steps-mid",
            label="spike count",
        )
        ax_chan_flag.legend()
        sweep_axis_data = swp.meta["sweep_axis_data"]
        extent = [
            -0.5,
            n_chan - 0.5,
            sweep_axis_data["f_sweep"][0].to_value(u.kHz),
            sweep_axis_data["f_sweep"][-1].to_value(u.kHz),
        ]
        ax_data_flag.imshow(
            bm.T, extent=extent, aspect="auto", origin="lower", interpolation="none"
        )

        def plot_page_func(axes, page_id, ci0, ci1):
            logger.debug(f"plot {page_id=} {ci0=} {ci1=}")
            kph.plot_nav_indicator(ax_chan_flag, page_id, ci0, ci1)
            kph.plot_nav_indicator(ax_data_flag, page_id, ci0, ci1)
            for i, ax in enumerate(axes):
                divider = make_axes_locatable(ax)
                if hasattr(ax, "kph_ax_res"):
                    ax_res = ax.kph_ax_res
                    ax_res.clear()
                else:
                    ax_res = ax.kph_ax_res = divider.append_axes(
                        "bottom",
                        size="20%",
                        pad="2%",
                        sharex=ax,
                    )
                ci = i + ci0
                if ci >= ci1:
                    continue
                ax.text(
                    0.05,
                    0.95,
                    f"ci={ci}",
                    ha="left",
                    va="top",
                    transform=ax.transAxes,
                )
                xx = swp.frequency[ci].to_value(u.MHz)
                yy_orig = ctx_spike["y"]
                yy_nospike = ctx_despike["y_nospike"]
                yy_med = ctx_spike["y_med"]
                ax.plot(xx, yy_orig[ci], color="C0", label="orig")
                ax.plot(xx, yy_med[ci], color="C1", label="S21_medfilt", linestyle="--")
                ax.plot(xx, yy_nospike[ci], color="C2", label="despiked")
                # ax.plot(xx, yy - yy_orig, color='C3', label='diff')
                spike_height_frac = ctx_spike["spike_height_frac"]
                min_spike_height_frac = ctx_spike["min_spike_height_frac"]
                ax_res.plot(xx, spike_height_frac[ci], color="C1")
                ax_res.axhline(min_spike_height_frac, color="gray", linestyle=":")
                ax_res.axhline(-min_spike_height_frac, color="gray", linestyle=":")
                m = ctx_despike["goodmask"]
                tshade = np.full((len(xx),), min_spike_height_frac)
                tshade[m[ci]] = 0
                tshaden = np.full((len(xx),), -min_spike_height_frac)
                tshaden[m[ci]] = 0
                ax_res.fill_between(xx, tshade, color="black", alpha=0.8)
                ax_res.fill_between(xx, tshaden, color="black", alpha=0.8)
                ax_res.set_ylim(
                    -min_spike_height_frac * 1.1, min_spike_height_frac * 1.1
                )
                if i == 0:
                    ax.legend()
                    ax.set_xlabel("f (MHz)")
                    ax.xaxis.set_label_position("top")
                    ax.set_ylabel("S21 (dB)")

        kph.plot_grid(plot_page_func)
        if save_filepath is None:
            kph.show()
        else:
            kph.save(save_filepath)


class SweepChecker(WorkflowStepBase):
    """The sweep checker."""

    despike: DespikeStep = Field(
        default_factory=dict,
        description="Despike step.",
    )

    s21_rms_thresh: float = Field(
        default=3,
        description="Threshold to flag high rms data.",
    )

    def run(self, arg, parent_workflow=None):
        """Run checking of sweep data."""
        self._mark_start(parent_workflow=parent_workflow)
        dctx = self.prepare_input(arg, read=True)
        swp_filepath = dctx.filepath
        swp = dctx.data
        logger.debug(f"check sweep {swp_filepath=} {swp}")
        if self.despike.enabled:
            ctx_despike = self.despike.run(arg, parent_workflow=self)
            if self.despike.plot_enabled if plot is None else plot
        ctx_check = self.check(dctx)
        ctx_spike = self._get_sub_context(self.despike.find_spike_S21)
        return self._mark_success(locals())

    def run_despike(self, arg, plot=None):
        """Run despikes."""
        ctx = self.despike.run(arg, parent_workflow=self)
        if plot:
            self.despike.plot()
        return self._mark_sub_context(ctx)

    def check(self, arg, plot=None):
        """Run despikes."""
        # locate despike finder ctx
        ctx_spike = self._get_or_create_sub_context(self.despike.find_spike_S21, arg)

        ctx = self.despike.run(arg, parent_workflow=self)
        plot = self.despike.plot_enabled if plot is None else plot
        if plot:
            self.despike.plot()
        return self._mark_sub_context(ctx)

    def check_data(self, arg):
        """Check sweep data."""
        swp = cast(MultiSweep, self.ensure_input_data(arg))
        return self._mark_sub_context(locals())

    def check_noise(self, arg):
        """Check noise of `swp`."""
