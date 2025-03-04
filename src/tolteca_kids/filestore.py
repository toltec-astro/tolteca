from pathlib import Path
from typing import ClassVar, Literal, get_args

import astropy.units as u
import dill
import plotly
import plotly.graph_objects as go
from astropy.table import Table
from astropy.utils.masked import Masked
from pydantic import BaseModel, Field
from tollan.utils.log import logger, logit
from tollan.utils.plot.plotly import show_in_dash
from tollan.utils.sys import ensure_path_parent_exists

__all__ = [
    "FileStoreConfigMixin",
]


# fix pickling of masked quantity.
MaskedQuantityInfo = Masked._get_masked_cls(u.Quantity).info.__class__  # noqa: SLF001


def _recreate_masked_quantity_info(attrs):
    if attrs is None:
        return MaskedQuantityInfo()
    obj = MaskedQuantityInfo(bound=True)
    obj._attrs = attrs  # noqa: SLF001
    return obj


@dill.register(MaskedQuantityInfo)
def _save_data_info(pickler, obj):
    pickler.save_reduce(
        _recreate_masked_quantity_info,
        (getattr(obj, "_attrs", None),),
        obj=obj,
    )


class PlotlyFigExportHtml(BaseModel):
    """A model to control plotly fig save as html."""

    auto_open: bool = False
    auto_plot: bool = False


class PlotlyFigExportImage(BaseModel):
    """A model to control plotly fig save as static image."""

    scale: None | float = 1.0
    width: None | int = 1600
    height: None | int = None


PlotlyFigExportImageFormat = Literal["png", "jpg", "jpeg", "pdf"]
PlotlyFigExportFormat = Literal["html",] | PlotlyFigExportImageFormat


class PlotlyFigExport(BaseModel):
    """A model to control plotly fig save."""

    html: PlotlyFigExportHtml = Field(default_factory=PlotlyFigExportHtml)
    image: PlotlyFigExportImage = Field(default_factory=PlotlyFigExportImage)

    def __call__(self, path: Path, fig: go.Figure):
        """Save ploty figure."""
        path = ensure_path_parent_exists(path)
        with logit(logger.info, f"save fig to {path}"):
            fmt = path.suffix.lstrip(".")
            if fmt == "html":
                plotly.offline.plot(
                    fig,
                    filename=path.as_posix(),
                    auto_open=self.html.auto_open,
                    auto_play=self.html.auto_plot,
                )
            elif fmt in get_args(PlotlyFigExportImageFormat):
                plotly.io.write_image(
                    fig,
                    path.as_posix(),
                    format=fmt,
                    scale=self.image.scale,
                    width=self.image.width,
                    height=self.image.height,
                )
            else:
                raise ValueError(f"unknown figure format: {path}")
        return path


class PlotlyFigShowInDash(BaseModel):
    """A model to control plotly fig show."""

    host: str = "0.0.0.0"  # noqa: S104
    port: int = 8888

    def __call__(self, data_items, title_text=None):
        """Save ploty figure."""
        show_in_dash(
            data_items,
            host=self.host,
            port=self.port,
            title_text=title_text or "Show in Dash",
        )


class FileStoreConfigMixin(BaseModel):
    """A mixin class for file store."""

    subdir_fmt: None | str = Field(
        default="{obsnum}-{subobsnum}-{scannum}",
        description="subdirectory format",
    )
    _filestore_path_attr: ClassVar[str] = "path"
    save_plotly: PlotlyFigExport = Field(
        default_factory=PlotlyFigExport,
        description="options for save plotly figure.",
    )
    show_plotly: PlotlyFigShowInDash = Field(
        default_factory=PlotlyFigShowInDash,
        description="options for show plotly live.",
    )

    def make_data_path(
        self,
        data=None,
        meta=None,
        suffix=None,
        name=None,
    ):
        """Return the path for given data."""
        if data is not None and meta is None:
            meta = data.meta
        if meta is not None and name is None:
            name = meta["filepath"].stem
        if name is None:
            raise ValueError("cannot infer name")
        name = f"{name}" if suffix is None else f"{name}{suffix}"
        subdir_fmt = self.subdir_fmt
        subdir = None if subdir_fmt is None else subdir_fmt.format_map(meta or {})
        path: Path = getattr(self, self._filestore_path_attr)
        parent = path if subdir is None else path.joinpath(subdir)
        return parent.joinpath(name)

    @classmethod
    def save_obj_pickle(
        cls,
        path: Path,
        obj,
    ):
        """Save object as pickle."""
        path = ensure_path_parent_exists(path)
        with (
            logit(
                logger.info,
                f"save {type(obj).__name__} obj to {path}",
            ),
            path.open("wb") as fo,
        ):
            dill.dump(obj, fo)
        return path

    @classmethod
    def save_table(
        cls,
        path: Path,
        tbl: Table,
    ):
        """Save astropy table."""
        logger.debug(f"save table\n{tbl}")
        _dispatch_tbl_fmt = {
            ".ecsv": "ascii.ecsv",
        }
        tbl_fmt = _dispatch_tbl_fmt[path.suffix]

        path = ensure_path_parent_exists(path)
        with logit(
            logger.info,
            f"save table of size={len(tbl)} to {path}",
        ):
            tbl.write(path, overwrite=True, format=tbl_fmt)
        return path
