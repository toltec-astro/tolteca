"""Select header-populated inputs for KIDs processing."""

from __future__ import annotations

import pandas as pd  # noqa: TC002 -- Pydantic resolves the context annotation
from pydantic import ConfigDict
from tollan.pipeline import PIPELINE_CONTEXT_KEY, Step, StepContext
from tollan.utils.log import logger

from tolteca_datamodels.core.file import SourceTrace, SourceTraceKind
from tolteca_datamodels.toltec.types import ToltecDataKind

from .config import InputConfig

__all__ = ["Input", "InputContext"]


class InputContext(StepContext["Input", InputConfig]):
    """Input-selection results."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    data: pd.DataFrame | None = None


class Input(Step[InputConfig, InputContext]):
    """Select inputs before per-file processing.

    Parameters
    ----------
    config : InputConfig, optional
        Query and error policy. Defaults to keeping all usable prepared inputs.
    """

    @classmethod
    def run(cls, data: pd.DataFrame, context: InputContext) -> None:
        """Evaluate the query on an opened input table.

        Parameters
        ----------
        data : pandas.DataFrame
            Header-populated table with live ``io_obj`` handles.
        context : InputContext
            Invocation settings and selected rows.

        Returns
        -------
        None
            Selected rows are stored in the context, including when none match.

        Raises
        ------
        ValueError
            If the selected table lacks preparation columns, or selected inputs
            are unusable and ``raise_on_errors`` is True. Invalid queries raise
            regardless of the error policy.
        """
        rows = data.copy(deep=False)
        # The result must not contain its own context through DataFrame.attrs.
        rows.attrs.pop(PIPELINE_CONTEXT_KEY, None)
        selected = (
            rows.toltec_file.query(context.config.select)
            if context.config.enabled and context.config.select is not None
            else rows
        )
        logger.debug(
            "KIDs input selection {!r} (enabled={}): {} -> {} rows",
            context.config.select,
            context.config.enabled,
            len(data),
            len(selected),
        )
        context.data = cls._filter_usable_inputs(
            selected, raise_on_errors=context.config.raise_on_errors
        )

    @staticmethod
    def _filter_usable_inputs(
        selected: pd.DataFrame, *, raise_on_errors: bool
    ) -> pd.DataFrame:
        """Check selected inputs without reading or closing their resources."""
        if selected.empty:
            return selected
        if not {"io_obj", "data_kind_toltec"} <= set(selected):
            msg = "Input selection requires an opened, header-populated table"
            raise ValueError(msg)
        raw = (
            ToltecDataKind.raw_kids_data
            | ToltecDataKind.lmt_tel
            | ToltecDataKind.lmt_tel2
        ).value
        opened = selected.io_obj.notna().to_numpy()
        recognized = (
            (selected.data_kind_toltec.fillna(0).astype(int) & raw).ne(0).to_numpy()
        )
        usable = opened & recognized
        policy = "raise" if raise_on_errors else "skip"
        failures = []
        diagnostic_lists = selected.get("source_trace", [None] * len(selected))
        for position, (source_loc, diagnostics) in enumerate(
            zip(selected.toltec_file.source_loc, diagnostic_lists, strict=True)
        ):
            reasons = []
            if not opened[position]:
                reasons.append("source is not open")
            if not recognized[position]:
                reasons.append("no recognized acquisition metadata")
            for diagnostic in (
                diagnostics.iter_error_entries()
                if isinstance(diagnostics, SourceTrace)
                else ()
            ):
                logger.debug(
                    "Selected source {}: {}: {} ({}, policy={})",
                    source_loc.url,
                    diagnostic.kind,
                    diagnostic.message,
                    diagnostic.context,
                    policy,
                )
                if diagnostic.kind not in {
                    SourceTraceKind.filepath_parse_failed,
                    SourceTraceKind.filepath_rule_unmatched,
                }:
                    reasons.append(diagnostic.message or diagnostic.kind.value)
            if reasons:
                usable[position] = False
                failure = f"{source_loc.url}: {'; '.join(reasons)}"
                failures.append(failure)
                logger.debug("Unusable selected input (policy={}): {}", policy, failure)
        logger.debug(
            "KIDs input validation (policy={}): {} selected, {} usable, {} unusable",
            policy,
            len(selected),
            int(usable.sum()),
            len(failures),
        )
        if failures and raise_on_errors:
            msg = f"{len(failures)} unusable selected input(s): " + "; ".join(failures)
            raise ValueError(msg)
        return selected.iloc[usable]

    def select(self, table: pd.DataFrame) -> pd.DataFrame:
        """Run selection without modifying the shared input table.

        Parameters
        ----------
        table : pandas.DataFrame
            Opened, header-populated inputs. The caller owns their lifetime.

        Returns
        -------
        pandas.DataFrame
            Usable selected rows. Row order and index labels are preserved.
            Handles remain owned by the caller;
            selection does not open, close or read files.
        """
        _, context = self(table.copy(deep=False), return_context=True)
        assert context.data is not None
        return context.data
