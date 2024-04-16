from dataclasses import dataclass
from typing import Any, Literal

import dtw
import numpy.typing as npt
from pydantic import Field
from tollan.config.types import ImmutableBaseModel
from tollan.utils.fmt import pformat_yaml
from tollan.utils.log import logger, timeit
from tollan.utils.np import strip_unit
from typing_extensions import assert_never

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
        reference,
        postproc_hook=None,
    ):
        """Run the detection."""
        query, reference = self._check_xy(query, reference)
        logger.debug(
            f"match 1d on data shape {query.shape} "
            f"config:\n{pformat_yaml(self.model_dump())}",
        )
        method = self.method
        if method == "dtw_python":
            result = self._dtw_python(
                query=query,
                reference=reference,
            )
        else:
            assert_never()
        if not result.success:
            logger.debug(f"{method} has failed.")
        else:
            if postproc_hook is not None:
                result = postproc_hook(result)
            alignment = result.alignment
            logger.debug(f"{method} succeeded, {alignment=}")
        return result

    @timeit
    def _dtw_python(
        self,
        query,
        reference,
        **kwargs,
    ):
        """Run dtw_python algorithm."""
        query, reference = self._check_xy(query, reference)
        reference_value, reference_unit = strip_unit(reference)
        query_value = query.to_value(reference_unit)
        alignment = dtw.dtw(x=query_value, y=reference_value, **kwargs)
        matched = alignment
        success = True

        return Match1DResult(
            query=query,
            reference=reference,
            matched=matched,
            alignment=alignment,
            success=success,
        )

    @staticmethod
    def _check_xy(query, reference):
        if not hasattr(query, "shape"):
            raise ValueError("unknown query data shape.")
        if reference is not None:
            if not hasattr(reference, "shape"):
                raise ValueError("unknown reference data shape.")
            if len(reference.shape) != 1 or len(query.shape) != 1:
                raise ValueError("data has to be 1-d")
        else:
            raise NotImplementedError
        return query, reference


@dataclass(kw_only=True)
class Match1DResult:
    """Result from Match1D."""

    query: npt.NDArray = ...
    reference: npt.NDArray = ...
    matched: Any = ...
    alignment: dtw.DTW = ...
    success: bool = False
