from pathlib import Path
from typing import ClassVar

import astropy.units as u
import dill
import plotly
import plotly.graph_objects as go
from astropy.table import Table
from astropy.utils.masked import Masked
from pydantic import BaseModel, Field
from tollan.utils.log import logger, logit

__all__ = [
    "OutputConfigMixin",
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


class OutputConfigMixin(BaseModel):
    """A mixin class for output."""

    subdir_fmt: None | str = Field(
        default="{obsnum}-{subobsnum}-{scannum}",
        description="subdirectory format",
    )
    _output_rootpath_attr: ClassVar[str] = "path"

    def make_output_path(
        self,
        data=None,
        meta=None,
        suffix=None,
        name=None,
    ):
        """Make output path."""
        if data is not None and meta is None:
            meta = data.meta
        if meta is not None and name is None:
            name = meta["filepath"].stem
        if name is None:
            raise ValueError("cannot infer name")
        name = f"{name}" if suffix is None else f"{name}{suffix}"
        subdir_fmt = self.subdir_fmt
        subdir = None if subdir_fmt is None else subdir_fmt.format_map(meta or {})
        rootpath: Path = getattr(self, self._output_rootpath_attr)
        parent = rootpath if subdir is None else rootpath.joinpath(subdir)
        return parent.joinpath(name)

    @staticmethod
    def _ensure_parents(path: Path):
        if path.is_dir():
            raise ValueError("invalid path type.")
        parent = path.parent
        if not parent.exists():
            with logit(logger.info, f"create dir {parent}"):
                parent.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def save_obj_pickle(
        cls,
        path: Path,
        obj,
    ):
        """Save object as pickle."""
        path = cls._ensure_parents(path)
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

        path = cls._ensure_parents(path)
        with logit(
            logger.info,
            f"save table of size={len(tbl)} to {path}",
        ):
            tbl.write(path, overwrite=True, format=tbl_fmt)
        return path

    @classmethod
    def save_plotly_fig(cls, path: Path, fig: go.Figure, **kwargs):
        """Save ploty figure."""
        path = cls._ensure_parents(path)
        # fig_name = fig["layout"].get("title", {}).get("text", None)
        # if not fig_name:
        #     fig_name = "<unknown>"
        with logit(logger.info, f"save fig to {path}"):
            plotly.offline.plot(
                fig,
                filename=path.as_posix(),
                auto_open=False,
                auto_play=False,
                **kwargs,
            )
        return path
