from dataclasses import dataclass
from typing import Any, Literal

import astropy.units as u
import dtw
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt
from astropy.table import QTable, unique
from pydantic import Field
from scipy.signal import correlate
from tollan.config.types import ImmutableBaseModel
from tollan.utils.fmt import pformat_mask, pformat_yaml
from tollan.utils.log import logger, timeit
from tollan.utils.np import attach_unit, qrange, strip_unit
from tollan.utils.plot.mpl import move_axes
from tollan.utils.plot.plotly import adjust_subplot_colorbars, make_subplots
from typing_extensions import assert_never


@timeit
def calc_shift1d_trace(x, y0, y1, shift_max=np.inf, return_locals=False):
    """Find relative shift between two traces.

    Parameters
    ----------
    x: 1-d array
        The vectors to compute shift for.
    y0, y1: 1-d array
        The signal.
    shift_max: float
        The maximum value to limit the peak search region.
    return_locals: bool
        If set, the ``locals()`` is returned for debugging purpose.

    Returns
    -------
    float:
        The shift such that (x+shift, y0) matches with (x, y1).
    dict: optional
        The ``locals()`` dict, if `return_locals` is True.
    """
    nx = len(x)
    cross_correlation = correlate(y0, y1, mode="same")
    dx = np.arange(-(nx // 2), nx - nx // 2) * (x[1] - x[0])
    mask_max = np.abs(dx) < shift_max
    logger.debug(f"use mask_max={pformat_mask(mask_max)}")
    ccm = cross_correlation_masked = np.ma.array(cross_correlation, mask=~mask_max)
    i = np.ma.argmax(ccm) if mask_max.sum() > 0 else nx // 2
    # note the negative sign
    shift = -dx[i]
    shift_snr = (ccm[i] - np.ma.median(ccm)) / np.std(ccm)
    if return_locals:
        return shift, locals()
    return shift


def calc_shift1d(  # noqa: PLR0913
    x0,
    x1,
    dx=None,
    shift_max=np.inf,
    dx_resample=1,
    return_locals=False,
):
    """Find relative shift between two vectors."""
    x_min = min(np.min(x0), np.min(x1))
    x_max = max(np.max(x0), np.max(x1))
    if dx is None:
        dx = np.min(np.r_[np.abs(np.diff(x0)), np.abs(np.diff(x1))]) / dx_resample
        logger.debug(f"use {dx=} computed from {dx_resample=}")
    x_pad = 3 * dx
    x_bins = qrange(x_min - x_pad, x_max + x_pad + dx, dx)
    x = (x_bins[:-1] + x_bins[1:]) * 0.5
    logger.debug(f"use x grid {x_min=} {x_max=} {dx=} {x_pad=} nx={len(x)}")
    y0 = np.histogram(x0, bins=x_bins)[0].astype(float)
    y1 = np.histogram(x1, bins=x_bins)[0].astype(float)
    r = calc_shift1d_trace(
        x,
        y0,
        y1,
        shift_max=shift_max,
        return_locals=return_locals,
    )
    if return_locals:
        return r[0], locals() | {"tracedata": r[1]}
    return r


Match1DMethod = Literal["dtw_python",]


class Match1D(ImmutableBaseModel):
    """Match 1d data."""

    method: Match1DMethod = Field(
        default="dtw_python",
        description="The matching method.",
    )

    def __call__(
        self,
        query,
        ref,
        postproc_hook=None,
        shift_kw=None,
    ):
        """Run the detection."""
        query, ref = self._check_xy(query, ref)
        logger.debug(
            f"match 1d on data shape {query.shape} "
            f"config:\n{pformat_yaml(self.model_dump())}",
        )

        method = self.method
        if method == "dtw_python":
            result = self._dtw_python(
                query=query,
                ref=ref,
                # open_begin=True,
                # open_end=True,
                # step_pattern="asymmetric",
                shift_kw=shift_kw,
            )
        else:
            assert_never()

        if not result.matched:
            logger.debug(f"{method} has failed.")
        else:
            if postproc_hook is not None:
                result = postproc_hook(result)
            # extract best match
            matched_sorted = result.matched.copy()
            matched_sorted.sort("adist_shifted")
            t = unique(matched_sorted, "idx_query", keep="first")
            t.sort("idx_query")
            result.data["query_matched"] = t
            t = unique(matched_sorted, "idx_ref", keep="first")
            t.sort("idx_ref")
            result.data["ref_matched"] = t
            logger.debug(f"{method} succeeded:\n{result.matched}.")
        return result

    @timeit
    def _dtw_python(
        self,
        query,
        ref,
        sort=True,
        shift_kw=None,
        **kwargs,
    ):
        """Run dtw_python algorithm."""
        query, ref = self._check_xy(query, ref)
        # do shift first
        shift, shiftdata = calc_shift1d(
            ref,
            query,
            return_locals=True,
            **shift_kw or {},
        )
        logger.debug(f"find shift query - ref = {shift}")
        ref_shifted = ref + shift

        ref_shifted_value, ref_unit = strip_unit(ref_shifted)
        query_value = query.to_value(ref_unit)
        if sort:
            isort_query = np.argsort(query_value)
            isort_ref = np.argsort(ref_shifted_value)
        else:
            isort_query = np.arange(len(query_value))
            isort_ref = np.arange(len(ref_shifted_value))
        query_sorted_value = query_value[isort_query]
        ref_shifted_sorted_value = ref_shifted_value[isort_ref]
        data = {
            "isort_query": isort_query,
            "isort_ref": isort_ref,
            "query_sorted": attach_unit(query_sorted_value, ref_unit),
            "ref_sorted": ref[isort_ref],
            "ref_shifted": ref_shifted,
            "ref_shifted_sorted": attach_unit(
                ref_shifted_sorted_value,
                ref_unit,
            ),
        }
        a = data["alignment"] = dtw.dtw(
            x=query_sorted_value,
            y=ref_shifted_sorted_value,
            keep_internals=True,
            **kwargs,
        )
        # original index
        iq = isort_query[a.index1]
        ir = isort_ref[a.index2]
        # matched values
        qm = query[iq]
        rm = ref[ir]
        rsm = ref_shifted[ir]
        matched = QTable(
            {
                "idx_query": iq,
                "idx_ref": ir,
                "isort_query": a.index1,
                "isort_ref": a.index2,
                "query": qm,
                "ref": rm,
                "ref_shifted": rsm,
                "dist": qm - rm,
                "dist_shifted": qm - rsm,
                "adist_shifted": np.abs(qm - rsm),
            },
            meta={
                "shift": shift,
            },
        )
        # mask_matched = np.zeros((len(query), len(ref)), dtype=bool)
        # for iq, ir in matched.iterrows("idx_query", "idx_ref"):
        #     mask_matched[iq, ir] = True
        # generate connectivity info
        g = nx.DiGraph()
        nq = matched["idx_query"]
        nr = len(query) + matched["idx_ref"]
        g.add_nodes_from(nq)
        g.add_nodes_from(nr)
        for i in range(len(matched)):
            g.add_edge(
                nq[i],
                nr[i],
                dist=matched["dist"][i],
                dist_shifted=matched["dist_shifted"][i],
            )
        unq = set(nq)
        n_matched_ref, n_matched_query = nx.bipartite.degrees(g, unq)
        n_matched_ref = data["n_matched_ref"] = np.array(
            list(dict(sorted(dict(n_matched_ref).items())).values()),
        )
        n_matched_query = data["n_matched_query"] = np.array(
            list(dict(sorted(dict(n_matched_query).items())).values()),
        )

        ug = g.to_undirected(as_view=True)
        group_items_query = data["items_query"] = np.array(
            [nx.node_connected_component(ug, q) for q in unq],
        )
        group_sizes_query = data["sizes_query"] = np.array(
            list(map(len, group_items_query)),
        )
        matched["n_matched_query"] = n_matched_query[matched["idx_query"]]
        matched["n_matched_ref"] = n_matched_ref[matched["idx_ref"]]
        matched["items"] = group_items_query[matched["idx_query"]]
        matched["size"] = group_sizes_query[matched["idx_query"]]

        return Match1DResult(
            config=self,
            query=query,
            ref=ref,
            shift=shift,
            shiftdata=shiftdata,
            ref_shifted=ref_shifted,
            matched=matched,
            # mask_matched=mask_matched,
            data=data,
        )

    @staticmethod
    def _check_xy(query, ref):
        if not hasattr(query, "shape"):
            raise ValueError("unknown query data shape.")
        if ref is not None:
            if not hasattr(ref, "shape"):
                raise ValueError("unknown ref data shape.")
            if len(ref.shape) != 1 or len(query.shape) != 1:
                raise ValueError("data has to be 1-d")
        else:
            raise NotImplementedError
        return query, ref


@dataclass(kw_only=True)
class Match1DResult:
    """Result from Match1D."""

    config: Match1D = ...
    query: npt.NDArray = ...
    ref: npt.NDArray = ...
    ref_shifted: npt.NDArray = ...
    shift: float | u.Quantity = ...
    shiftdata: dict[str, Any] = ...
    matched: QTable = ...
    # mask_matched: npt.NDArray = ...
    data: dict[str, Any] = ...

    def _plot_dtw_python_mpl(self, ax=None, type="density"):
        alignment: dtw.DTW = self.data["alignment"]
        if ax is None:
            ax = plt.subplots(1, 1)
        type = {
            "density": "density",
            "match": "twoway",
        }[type]
        return move_axes(alignment.plot(type=type), ax)

    def _plot_dtw_python_plotly(self, type="density", **kwargs):
        plot_func = {
            "density": self._plot_dtw_python_density_plotly,
            "match": self._plot_dtw_python_match_plotly,
        }[type]
        return plot_func(**kwargs)

    def _plot_dtw_python_density_plotly(
        self,
        fig=None,
        panel_kw=None,
        label_query=None,
        label_ref=None,
        **_kwargs,
    ):
        alignment: dtw.DTW = self.data["alignment"]
        # idx_ref = self.data["idx_ref"]
        # idx_query = self.data["idx_query"]
        cm = alignment.costMatrix

        fig = fig or make_subplots(1, 1)
        panel_kw = panel_kw or {}
        z = cm.T
        fig.add_heatmap(
            z=z,
            # x=np.arange(z.shape[1])[idx_query],
            # y=np.arange(z.shape[0])[idx_ref],
            colorscale="rdylgn_r",
            **panel_kw,
        )
        fig.update_xaxes(
            title=label_query or "Query Id",
            **panel_kw,
        )
        fig.update_yaxes(
            title=label_ref or "ref Id",
            **panel_kw,
        )
        fig.add_scatter(
            x=alignment.index1,
            y=alignment.index2,
            mode="markers+lines",
            marker={
                "size": 4,
                "color": "black",
            },
            line={
                "color": "gray",
            },
            showlegend=False,
            **panel_kw,
        )
        adjust_subplot_colorbars(fig)
        return fig

    def _plot_dtw_python_match_plotly(  # noqa: PLR0913
        self,
        fig=None,
        panel_kw=None,
        label_query=None,
        label_ref=None,
        label_value=None,
        **_kwargs,
    ):
        fig = fig or make_subplots(1, 1)
        panel_kw = panel_kw or {}

        matched = self.matched
        query_sorted_value, data_unit = strip_unit(self.data["query_sorted"])
        ref_sorted_value, _ = strip_unit(self.data["ref_sorted"])
        n_ref = len(ref_sorted_value)
        n_query = len(query_sorted_value)
        y_ref = np.ones((n_ref,))
        y_query = np.ones((n_query,)) + 1

        fig.add_scatter(
            x=query_sorted_value,
            y=y_query,
            mode="markers",
            marker={
                "size": 4,
                "color": "blue",
            },
            showlegend=False,
            **panel_kw,
        )
        fig.add_scatter(
            x=ref_sorted_value,
            y=y_ref,
            mode="markers",
            marker={
                "size": 4,
                "color": "black",
            },
            showlegend=False,
            **panel_kw,
        )

        # plot match
        for i in range(len(matched)):
            fig.add_scatter(
                x=[
                    query_sorted_value[matched["isort_query"][i]],
                    ref_sorted_value[matched["isort_ref"][i]],
                ],
                y=[
                    y_query[matched["isort_query"][i]],
                    y_ref[matched["idx_ref"][i]],
                ],
                mode="lines",
                line={
                    "color": "red",
                    "width": 1,
                },
                **panel_kw,
            )
        fig.update_xaxes(
            title=label_value or f"Value ({data_unit})",
            **panel_kw,
        )
        fig.update_yaxes(
            tickvals=[1, 2],
            ticktext=[label_ref or "ref", label_query or "query"],
            autorange=False,
            range=[0, 3],
            **panel_kw,
        )
        return fig

    def make_mpl_fig(self, **kwargs):
        """Return matplotlib figure."""
        if self.config.method == "dtw_python":
            return self._plot_dtw_python_mpl(**kwargs)
        assert_never()

    def make_plotly_fig(self, **kwargs):
        """Return plotly figure."""
        if self.config.method == "dtw_python":
            return self._plot_dtw_python_plotly(**kwargs)
        assert_never()
