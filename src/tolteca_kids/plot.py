from typing import ClassVar

import plotly
import plotly.graph_objects as go
from plotly.basedatatypes import BasePlotlyType
from tollan.config.types import AbsDirectoryPath
from tollan.utils.general import rupdate, slugify
from tollan.utils.log import logger, logit, timeit
from tollan.utils.plot.plotly import (
    ColorPalette,
    SubplotGrid,
    make_subplot_layout,
    make_subplots,
    show_in_dash,
    update_subplot_layout,
)

from .pipeline import Step, StepConfig, StepContext

__all__ = [
    "PlotConfig",
    "PlotMixin",
]


# this speeds up exceptions containing plotly objects.
def _plotly_build_repr_for_class(
    props,  # noqa: ARG001
    class_name,
    parent_path_str=None,
):
    return BasePlotlyType._build_repr_for_class(  # noqa: SLF001
        [],
        class_name,
        parent_path_str=parent_path_str,
    )


BasePlotlyType._build_repr_for_class = staticmethod(  # noqa: SLF001
    _plotly_build_repr_for_class,
)


class PlotConfig(StepConfig):
    """A base model for plot config."""

    save: bool = True
    save_rootpath: None | AbsDirectoryPath = None
    show: bool = False
    show_in_dash_port: int = 8888
    show_in_dash_host: str = "0.0.0.0"  # noqa: S104

    def save_or_show(
        self,
        data_items: list | go.Figure,
        name=None,
        save_name=None,
    ):
        """Render figure."""
        if self.show:
            show_in_dash(
                data_items,
                host=self.show_in_dash_host,
                port=self.show_in_dash_port,
                title_text=name,
            )
        if self.save and self.save_rootpath is not None:
            save_name = save_name or name
            for data_item in data_items:
                data = data_item["data"]
                item_name = data_item["title_text"]
                if save_name is None:
                    save_name = item_name
                else:
                    save_name = f"{save_name}_{item_name}"
                if isinstance(data, go.Figure):
                    self._save_fig(data, save_name, save_rootpath=self.save_rootpath)
                else:
                    pass

    @staticmethod
    def _save_fig(fig: go.Figure, save_name, save_rootpath):
        save_name = save_name or fig.layout["title"]["text"]
        if save_name is None:
            raise ValueError("no save name specified or implied.")
        # sanitize save name
        save_name = slugify(save_name)
        save_path = save_rootpath.joinpath(f"{save_name}.html")
        save_dir = save_path.parent
        if not save_dir.exists():
            with logit(logger.debug, "create figure save dir {save_dir}"):
                save_dir.mkdir(parents=True, exist_ok=True)
        with logit(logger.debug, f"save {save_name} to {save_path}"):
            plotly.offline.plot(
                fig,
                filename=save_path.as_posix(),
                auto_open=False,
            )
        return save_path


class PlotMixin:
    """A mixin class for plotting step."""

    @classmethod
    def make_subplots(cls, *args, **kwargs):
        """Return a figure with subplots."""
        kwargs.setdefault("fig_layout", cls.fig_layout_default)
        return make_subplots(*args, **kwargs)

    @classmethod
    def make_subplot_grid(cls, *args, **kwargs):
        """Return a subplot grid."""
        kwargs.setdefault("fig_layout", cls.fig_layout_default)
        return SubplotGrid(*args, **kwargs)

    @classmethod
    def _make_show_name(cls, _data, _context):
        """Return the show name."""
        return f"{cls.context_key}"

    @classmethod
    def _make_save_name(cls, data, _context):
        """Return the save name."""
        meta = data.meta
        base = meta["name"] if "name" in meta else meta["source"].path.stem
        suffix = cls.__name__.lower()
        return f"{base}_{suffix}"

    @classmethod
    def save_or_show(cls, data, context: StepContext[Step, PlotConfig]):
        """Save or show the data on context."""
        data_items = []
        for name, value in context.model_dump(include=["data"])["data"].items():
            data_items.append({"title_text": name, "data": value})
        context.config.save_or_show(
            data_items,
            name=cls._make_show_name(data, context),
            save_name=cls._make_save_name(data, context),
        )

    color_palette: ClassVar = ColorPalette()
    fig_layout_default: ClassVar[dict] = {
        "xaxis": {
            "showline": True,
            "showgrid": False,
            "showticklabels": True,
            "linecolor": "black",
            "linewidth": 1,
            "ticks": "outside",
        },
        "yaxis": {
            "showline": True,
            "showgrid": False,
            "showticklabels": True,
            "linecolor": "black",
            "linewidth": 1,
            "ticks": "outside",
        },
        "plot_bgcolor": "white",
        "autosize": True,
        "margin": {
            "autoexpand": True,
            "l": 0,
            "b": 0,
            "t": 20,
        },
        "modebar": {
            "orientation": "v",
        },
    }

    @classmethod
    def make_data_grid_anim(  # noqa: PLR0913, C901
        cls,
        name,
        n_rows,
        n_cols,
        n_items,
        make_panel_func,
        fig=None,
        **kwargs,
    ):
        """Create data grid figure with pager."""
        fig = fig or cls.make_subplots(
            n_rows,
            n_cols,
            **kwargs,
        )
        frame_data = []
        n_items_per_frame = n_rows * n_cols
        n_frames = n_items // n_items_per_frame + (n_items % n_items_per_frame > 0)
        logger.debug(
            f"make data grid {n_rows=} {n_cols=} {n_items} "
            f"{n_items_per_frame=} {n_frames=}",
        )
        frames = []
        for i in range(n_frames):
            start = i * n_items_per_frame
            stop = start + n_items_per_frame
            frames.append(
                {
                    "id": i,
                    "start": start,
                    "stop": stop,
                    "name": f"{start}",
                    "indices": list(range(start, stop)),
                    "slice": slice(start, stop),
                },
            )

        def _make_frame_panels(frame, init=False):
            panel_indices = frame["indices"]
            panels = []
            panel_id = 0
            for row in range(1, n_rows + 1):
                for col in range(1, n_cols + 1):
                    pi = panel_indices[panel_id]
                    panel = make_panel_func(
                        row,
                        col,
                        pi,
                        dummy=(pi >= n_items),
                        init=init,
                    )
                    panel["panel_kw"] = {
                        "row": row,
                        "col": col,
                    }
                    panels.append(panel)
                    panel_id += 1
            return panels

        # add data for first frame
        for panel in _make_frame_panels(frames[0], init=True):
            panel_kw = panel["panel_kw"]
            for trace in panel["data"]:
                fig.add_trace(trace, **panel_kw)
            update_subplot_layout(fig, panel["layout"], **panel_kw)
        # create frame
        frame_data = []
        steps_data = []
        with timeit("create frames"):
            for frame in frames:
                fname = frame["name"]
                panels = _make_frame_panels(frame, init=False)
                # get flat list of traces and the valid trace ids
                traces = [trace for panel in panels for trace in panel["data"]]
                trace_data = []
                trace_ids = []
                for ti, t in enumerate(traces):
                    if t is not None:
                        trace_data.append(t)
                        trace_ids.append(ti)
                # collate layout
                layout = {}
                for panel in panels:
                    panel_kw = panel["panel_kw"]
                    rupdate(
                        layout,
                        make_subplot_layout(
                            fig,
                            panel["layout"],
                            **panel_kw,
                        ),
                    )
                frame_data.append(
                    go.Frame(
                        {
                            "data": trace_data,
                            "traces": trace_ids,
                            "name": fname,
                            "layout": layout,
                        },
                    ),
                )
                steps_data.append(
                    {
                        "args": [
                            [fname],
                            {
                                "frame": {
                                    "duration": 30,
                                    "redraw": False,
                                },
                                "mode": "immediate",
                                "transition": {"duration": 30},
                            },
                        ],
                        "label": fname,
                        "method": "animate",
                    },
                )
        sliders = [
            {
                # "y": 1,
                # "yanchor": "bottom",
                "pad": {
                    "t": 40,
                    "b": 20,
                },
                "steps": steps_data,
            },
        ]
        with timeit("create figure"):
            fig["frames"] += tuple(frame_data)
            fig.update_layout(
                title={
                    "text": name,
                },
                # margin={"l": 75, "r": 50},
                # margin={"l": 100, "r": 100},
                margin={"t": 40},
                sliders=sliders,
                # uirevision=True,
                showlegend=False,
            )
        return fig
