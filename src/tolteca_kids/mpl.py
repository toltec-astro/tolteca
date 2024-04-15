import dataclasses
from dataclasses import field
from typing import Any

import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.transforms as tx
import numpy as np
from matplotlib import gridspec


@dataclasses.dataclass
class KidsPlotter:
    """A helper class to make kids plot."""

    n_rows: int = 5
    n_cols: int = 5
    panel_width: float = 3
    panel_height: float = 3
    figure_kw: None | dict = None
    ax_nav_kw: None | dict = None
    axes_grid_kw: None | dict = None
    figure: Any = None
    gs_nav: Any = None
    gs_grid: Any = None
    ax_nav: Any = field(init=False, repr=False)
    axes_grid: Any = field(init=False, repr=False)

    def __post_init__(self):
        plt.style.use("seaborn-v0_8-muted")
        self._init_figure()

    def _init_figure(self):  # noqa: C901, PLR0915
        nr = self.n_rows
        nc = self.n_cols
        pw = self.panel_width
        ph = self.panel_height
        fig_kw = self.figure_kw or {}
        an_kw = self.ax_nav_kw or {}
        ag_kw = self.axes_grid_kw or {}

        grid_size = (nc * pw, nr * ph)
        nav_height = ph * 0.5
        fig_kw = {
            "figsize": (grid_size[0], grid_size[1] + nav_height),
        } | fig_kw
        if self.figure is None:
            fig = plt.figure(**fig_kw)
        else:
            fig = self.figure
            fig.set_size_inches(fig_kw.get("figsize"))

        if sum([self.gs_nav is None, self.gs_grid is None]) == 1:
            raise ValueError("both gs_nav and gs_grid need to be None or set.")
        if self.gs_nav is None:
            gs0 = gridspec.GridSpec(
                2,
                1,
                figure=fig,
                height_ratios=[nav_height, grid_size[1]],
            )
            gs_nav = gs0[0, 0]
            gs_grid = gs0[1, 0]
        else:
            gs_nav = self.gs_nav
            gs_grid = self.gs_grid
            # set the ratios if these are asscociated with the same gridspec
            gs0 = gs_nav.get_gridspec()
            if gs_grid.get_gridspec() is gs0:
                hr = gs0.get_height_ratios()
                if hr is None:
                    hr = [1] * gs0.nrows
                hr = np.array(hr, dtype=float)
                hr_nav = hr[gs_nav.rowspan]
                hr_grid = hr[gs_grid.rowspan]
                hr[gs_nav.rowspan] = hr[gs_nav.rowspan] * (
                    nav_height * hr_grid.sum() / grid_size[1] / hr_nav.sum()
                )
                gs0.set_height_ratios(hr)
            # set figure size proportionally if gs0 is a subgridspec
            if isinstance(gs0, gridspec.GridSpecFromSubplotSpec):
                gs_fig = gs0._subplot_spec  # type: ignore  # noqa: SLF001
                hr = gs_fig.get_gridspec().get_height_ratios()
                wr = gs_fig.get_gridspec().get_width_ratios()
                w, h = fig.get_size_inches()
                if hr is not None:
                    hr = np.array(hr, dtype=float)
                    h = h * hr.sum() / hr[gs_fig.rowspan].sum()
                if wr is not None:
                    wr = np.array(wr, dtype=float)
                    w = w * wr.sum() / wr[gs_fig.colspan].sum()
                fig.set_size_inches(w, h)

        ax_nav = fig.add_subplot(gs_nav, **an_kw)  # type: ignore
        gs1 = gs_grid.subgridspec(nr, nc)
        axes_grid = []
        kw = {}
        for i in range(nr):
            for j in range(nc):
                ax = fig.add_subplot(gs1[i, j], **ag_kw, **kw)  # type: ignore
                # if not axes_grid:
                #     kw.update(
                #         {
                #             "sharex": ax,
                #             "sharey": ax,
                #         },
                #     )
                axes_grid.append(ax)
        axes_grid = np.array(axes_grid).reshape((nr, nc))
        self.__dict__.update(
            {
                "figure": fig,
                "gs_nav": gs_nav,
                "gs_grid": gs_grid,
                "ax_nav": ax_nav,
                "axes_grid": axes_grid,
            },
        )

    def set_title(self, arg):
        """Set figure title."""
        if isinstance(arg, str):
            name = arg
        elif hasattr(arg, "meta"):
            meta = arg.meta
            kind_str = meta["data_kind"].name.lower()
            if "chan_axis_data" in meta:
                block_index = meta["chan_axis_data"].meta["block_index"]
            else:
                block_index = -1
            name = (
                ("{kind_str}-nw{nw}-{obsnum}-{subobsnum}-{scannum}[{{block_index}}]")
                .format(
                    kind_str=kind_str,
                    **meta,
                )
                .format(block_index=block_index)
            )
        else:
            name = str(arg)
        self.figure.suptitle(name)

    def plot_nav(self, n_items, plot_page_func):
        """Make nav bar plot."""
        nr = self.n_rows
        nc = self.n_cols
        ns = nr * nc
        ax = self.ax_nav
        n_pages = n_items // ns + int(n_items % ns != 0)
        page_data = []
        id_to_page_id = {}
        page_plot_ctx = []
        for page_id in range(n_pages):
            i0 = ns * page_id
            i1 = i0 + ns
            if i1 > n_items:
                i1 = n_items
            page_plot_ctx.append(plot_page_func(ax, page_id, i0, i1))
            page_data.append(
                {
                    "i0": i0,
                    "i1": i1,
                    "page_id": page_id,
                },
            )
            for i in range(i0, i1):
                id_to_page_id[i] = page_id
        # ax.tick_params(
        #     axis="x", which="both", bottom=False, top=True,
        #     labelbottom=False, labeltop=True,
        #     )
        self._ctx_nav = locals()

    def plot_sweep_chan_nav(self, swp):
        """Make nav plot for sweep channels."""
        self.set_title(swp)
        n_chan, n_sweepsteps = swp.frequency.shape
        chan_id = np.arange(n_chan)
        chan_axis_data = swp.meta["chan_axis_data"]
        f_chans = chan_axis_data["f_chan"]
        f_range = swp.frequency.max(axis=1) - swp.frequency.min(axis=1)

        def plot_page_func(ax, page_id, i0, i1):
            ax.errorbar(
                chan_id[i0:i1],
                f_chans[i0:i1].to(u.MHz),
                yerr=f_range[i0:i1].to(u.MHz) / 2.0,
                color=f"C{page_id % 5}",
                marker=".",
                drawstyle="steps-mid",
            )
            ax.axvspan(
                i0 - 0.3,
                i1 + 0.3,
                color="#cccccc" if (page_id % 2 == 0) else "#eeccee",
            )
            if page_id == 0:
                # setup axis
                ax.set_xlabel("chan_id")
                # ax.xaxis.set_label_position("top")
                ax.set_ylabel("f_chan (MHz)")
            return locals()

        return self.plot_nav(n_chan, plot_page_func)

    def plot_nav_indicator(self, ax, _page_id, i0, i1):
        """Plot nav indicator."""
        ctx_nav = self._ctx_nav
        if ctx_nav is None:
            raise ValueError("run plot_nav to setup nav context.")
        if "nav_indicators" not in ctx_nav:
            nav_indicators = ctx_nav["nav_indicators"] = {}
        else:
            nav_indicators = ctx_nav["nav_indicators"]
        if ax in nav_indicators:
            patch = nav_indicators[ax]
            patch.remove()
        # make patch
        nav_indicators[ax] = ax.axvspan(i0 - 0.5, i1 + 0.5, color="red", fill=False)

    def plot_grid(self, plot_page_func):
        """Make grid plot."""
        ctx_nav = getattr(self, "_ctx_nav", None)
        if ctx_nav is None:
            raise ValueError("run plot_nav to setup nav context.")

        def plot_page_wrapper(page_id):
            pd = ctx_nav["page_data"][page_id]
            i0 = pd["i0"]
            i1 = pd["i1"]
            axes = self.axes_grid.ravel()
            for i, ax in enumerate(axes):
                ax.clear()
                if i + i0 >= i1:
                    ax.text(0.5, 0.5, "NO DATA", ha="center", va="center", color="red")
            self.plot_nav_indicator(self.ax_nav, page_id, i0, i1)
            plot_page_func(axes, page_id, i0, i1)
            n_pages = ctx_nav["n_pages"]
            ax_nav = self.ax_nav
            ax_nav.set_title(f"current page [{page_id}/{n_pages}] item_id=[{i0}:{i1}]")
            plt.draw()
            set_to_shared_scale(axes)
            plt.draw()

        def onclick(event):
            if event.inaxes != ctx_nav["ax"]:
                return
            i = int(np.round(event.xdata))
            page_id = ctx_nav["id_to_page_id"][i]
            plot_page_wrapper(page_id)

        self.figure.canvas.mpl_connect("button_press_event", onclick)
        # initialize
        plot_page_wrapper(0)

    def show(self):
        """Show the plot."""
        self.figure.tight_layout()
        plt.show()

    def save(self, *args, **kwargs):
        """Save the plot."""
        self.figure.savefig(*args, **kwargs)


def set_to_shared_scale(axes):
    """Set the axes to have same scale, but limits may vary."""
    # https://stackoverflow.com/a/66062280
    ax_selec = [(ax, ax.get_ylim()) for ax in axes]
    max_delta = max([lmax - lmin for _, (lmin, lmax) in ax_selec])
    for ax, (lmin, lmax) in ax_selec:
        ax.set_ylim(
            lmin - (max_delta - (lmax - lmin)) / 2,
            lmax + (max_delta - (lmax - lmin)) / 2,
        )


def plot_axvlines(ax, x, y0=0, y1=1, **kwargs):
    """Plot vlines."""
    trans = tx.blended_transform_factory(ax.transData, ax.transAxes)
    ax.plot(
        np.repeat(x, 3),
        np.tile([y0, y1, np.nan], len(x)),
        transform=trans,
        **kwargs,
    )
