"""Shared bounded figures for saved calibrated timestream inspections."""

import numpy as np
import plotly.graph_objects as go
from tollan.plot.plotly import PlotlyViz

_MAX_CHANNELS = 100
_MAX_TIMES = 4096
_MAX_WINDOWS = 512


class TimestreamInspectionViz(PlotlyViz):
    """Native windows, slow trends, spectra and time/frequency waterfalls."""

    def __call__(self, tree, *, mode, channels, signal="x", window=0):
        """Read only the selected signal and channels from an open inspection.

        Parameters
        ----------
        tree : xarray.DataTree
            Saved TimestreamInspect result.
        mode : str
            native, trend, native_psd, trend_psd or waterfall.
        channels : sequence of int
            Positional channel selection, at most 100.
        signal : str, optional
            I, Q, r or x, default x.
        window : int, optional
            Native-window ordinal, default zero.

        Returns
        -------
        plotly.graph_objects.Figure
            WebGL curves or a spectral waterfall. Full-duration displays cap
            presentation to 4096 time coordinates or 512 window coordinates;
            scientific arrays in the store remain unchanged.
        """
        if signal not in {"I", "Q", "r", "x"}:
            msg = "Unknown inspection signal"
            raise ValueError(msg)
        if not channels or len(channels) > _MAX_CHANNELS:
            msg = "Select between 1 and 100 inspection channels"
            raise ValueError(msg)
        node = {
            "native": f"windows/{window}",
            "trend": "trend",
            "native_psd": "window_psd",
            "trend_psd": "trend_psd",
            "waterfall": "window_psd",
        }[mode]
        try:
            saved = tree[node]
        except KeyError as exc:
            msg = (
                "This inspection has no requested "
                "window or spectrum (insufficient duration)"
            )
            raise ValueError(msg) from exc
        arr = saved.dataset[signal]
        if mode == "native_psd":
            arr = arr.isel(window=window)
        arr = arr.isel(chan=channels)
        fig = go.Figure()
        if mode == "waterfall":
            arr = arr.isel(chan=0)
            if arr.sizes["window"] > _MAX_WINDOWS:
                arr = arr.isel(
                    window=np.linspace(
                        0, arr.sizes["window"] - 1, _MAX_WINDOWS, dtype=int
                    )
                )
            arr = arr.transpose("window", "f_psd").load()
            values = arr.values
            z = np.log10(np.where(values > 0, values, np.nan))
            fig.add_trace(
                go.Heatmap(
                    x=arr.f_psd.values,
                    y=arr.window.values,
                    z=z,
                    colorscale="Viridis",
                    colorbar={"title": f"log10 {signal} PSD"},
                )
            )
            fig.update_xaxes(title_text="Frequency (Hz)")
            fig.update_yaxes(title_text="Window start (s)")
            if arr.sizes["window"] == 1:
                # A single saved window has no intermediate acquisition times.
                fig.update_yaxes(tickmode="array", tickvals=arr.window.values)
        else:
            dim = "f_psd" if "psd" in mode else "time"
            if arr.sizes[dim] > _MAX_TIMES:
                arr = arr.isel(
                    {dim: np.linspace(0, arr.sizes[dim] - 1, _MAX_TIMES, dtype=int)}
                )
            arr = arr.load()
            for i, chan in enumerate(arr.chan.values):
                fig.add_trace(
                    go.Scattergl(
                        x=arr[dim].values,
                        y=arr.isel(chan=i).values,
                        name=f"{signal} / {chan}",
                        mode="lines",
                        connectgaps=False,
                    )
                )
            unit = arr.attrs.get("units", "")
            fig.update_xaxes(
                title_text="Frequency (Hz)" if "psd" in mode else "Elapsed time (s)",
                type="log" if "psd" in mode else "linear",
            )
            fig.update_yaxes(
                title_text=f"{signal} PSD ({unit})"
                if "psd" in mode
                else f"{signal} ({unit or 'dimensionless'})",
                type="log" if "psd" in mode else "linear",
            )
        fig.update_layout(height=520, margin={"l": 80, "r": 40, "t": 30, "b": 60})
        return fig
