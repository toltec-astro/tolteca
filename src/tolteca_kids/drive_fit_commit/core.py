"""Independent commit workflow consuming a drive-fit product and target sweep."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import ConfigDict, Field
from tollan.config import ConfigHandler, FrozenBaseModel, SubConfigKeyTransformer
from tollan.config.types import AbsAnyPath

from tolteca_datamodels.data_prod import DataProdIndex, DataProdIndexAssoc
from tolteca_kids.config import DataProdOutputConfig
from tolteca_kids.drive_fit.core import _execute_workflow
from tolteca_kids.drive_fit.input import prepare_drive_fit_inputs
from tolteca_kids.drive_fit.product import _open_interface, open_drive_fit_product
from tolteca_kids.drive_fit.runner import _discover
from tolteca_kids.drive_fit.step import DriveFitStep
from tolteca_kids.product import _product_index
from tolteca_kids.tlaloc_output import TlalocOutput, TlalocOutputConfig

from .plot import DriveFitCommitPlotStep, DriveFitCommitPlotStepConfig
from .step import DriveFitCommitStep, DriveFitCommitStepConfig

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from pathlib import Path

    import xarray as xr
    from tollan.config import RuntimeContext
    from tollan.utils.fileloc import FileLoc


class DriveFitCommitConfig(FrozenBaseModel):
    """Workflow settings under the runtime's root drive_fit_commit section.

    Parameters
    ----------
    reference : pathlib.Path or None, optional
        Existing drive-fit product directory or index. Required by discovery/CLI;
        Python callers may supply a DataProdIndex or path directly instead.
        A version-group path resolves its current main tag once at preparation.
    commit : DriveFitCommitStepConfig, optional
        Shared tone-matcher and amplitude-normalization settings.
    plot : DriveFitCommitPlotStepConfig, optional
        Commit diagnostics; disabled by default.
    output : DataProdOutputConfig, optional
        Product publication and registration settings; disabled by default.
    tlaloc_output : TlalocOutputConfig, optional
        Optional application of settings after product output/registration.
        Disabled by default; requires an explicit destination when enabled.
    select : str or None, optional
        Target input metadata query. Each interface must select one sweep from
        the same acquisition. None retains all discovered target/tune sweeps.
    """

    model_config = ConfigDict(extra="forbid")
    reference: AbsAnyPath | None = None
    commit: DriveFitCommitStepConfig = Field(default_factory=DriveFitCommitStepConfig)
    plot: DriveFitCommitPlotStepConfig = Field(
        default_factory=lambda: DriveFitCommitPlotStepConfig(enabled=False)
    )
    output: DataProdOutputConfig = Field(default_factory=DataProdOutputConfig)
    tlaloc_output: TlalocOutputConfig = Field(default_factory=TlalocOutputConfig)
    select: str | None = None


def prepare_drive_fit_commit_inputs(
    sources: Iterable[str | Path | FileLoc],
    reference: Path | DataProdIndex,
) -> xr.DataTree:
    """Pin a fitted product and prepare one target sweep per selected interface.

    Parameters
    ----------
    sources : iterable of str, pathlib.Path or FileLoc
        Exact target sweep locations, all from the same raw acquisition.
    reference : pathlib.Path or DataProdIndex
        Published drive fit. Its namespace and realized run must be present.

    Returns
    -------
    xarray.DataTree
        Working collection with loaded reference selections and target identities.
        Raw headers and published-array storage are closed before return.

    Raises
    ------
    ValueError
        Reference is not a realized drive fit, or target selection is ambiguous.
    """
    data = prepare_drive_fit_inputs(sources)
    index = _product_index(reference)
    if (
        index.meta["data_prod_type"] != "dp_drivefit"
        or index.namespace is None
        or index.run is None
    ):
        msg = "Commit requires a published drive-fit realization"
        raise ValueError(msg)
    targets = set()
    for interface, tree in data["interfaces"].children.items():
        if tree["inputs"].sizes["input"] != 1:
            msg = "Drive-fit commit requires exactly one target sweep per interface"
            raise ValueError(msg)
        targets.update(tree["inputs"].uid_raw_obs.values.tolist())
        with open_drive_fit_product(index, interface=interface) as saved:
            records = [
                DriveFitStep.load_context(ctx)
                for key, ctx in saved["execution"].attrs.items()
                if key == DriveFitStep.get_context_key()
                or str(key).startswith(DriveFitStep.get_context_key() + ":")
            ]
            if len(records) != 1 or not records[0].completed:
                msg = "Commit requires exactly one completed fit per interface"
                raise ValueError(msg)
            config = records[0].config
            result = DriveFitStep.view(saved, alias=config.alias).data
            if "power_fraction" not in result:
                msg = "Rerun drive fitting to record full-comb power normalization"
                raise ValueError(msg)
            fields = [
                "f_chan",
                "amp_chan",
                "power_fraction",
                "p_flag",
                "p_best",
                "a_best",
                "Qr_best",
                "adrv_best",
            ]
            tree["reference"] = result[fields].assign_coords(power=result.power).load()
        tree.attrs["reference_product"] = index.reference.model_dump(mode="json")
    if len(targets) != 1:
        msg = "Commit target interfaces must belong to one raw acquisition"
        raise ValueError(msg)
    data.attrs["reference_product"] = index.reference.model_dump(mode="json")
    return data


def make_drive_fit_commit_product_index(data: xr.DataTree) -> DataProdIndex:
    """Describe a target commit and its pinned scientific inputs.

    Parameters
    ----------
    data : xarray.DataTree
        Prepared target collection with a pinned reference product.

    Returns
    -------
    DataProdIndex
        Proposed commit product; target acquisition is primary and the fitted
        realization is an explicit namespaced dependency.
    """
    tree = next(iter(data["interfaces"].children.values()))
    inputs = tree["inputs"].to_dataset()
    target = str(inputs.uid_raw_obs.values[0])
    uid = f"{target}_drive_fit_commit"
    ref = data.attrs["reference_product"]
    return DataProdIndex(
        uid=uid,
        meta={
            "tag": "drive_fit_commit",
            "data_prod_type": "dp_drive_fit_commit",
            "name": uid,
            "master": inputs.attrs["master"],
            **{
                name: int(inputs[name].values[0])
                for name in ("obsnum", "subobsnum", "scannum")
            },
        },
        assocs=[
            DataProdIndexAssoc(
                uid=target,
                data_prod_assoc_type="dpa_drive_fit_commit_raw_obs",
                is_primary=True,
            ),
            DataProdIndexAssoc(
                **ref, data_prod_assoc_type="dpa_drive_fit_commit_drivefit"
            ),
        ],
    )


class DriveFitCommit(
    SubConfigKeyTransformer[Literal["drive_fit_commit"]],
    ConfigHandler[DriveFitCommitConfig],
):
    """Commit a published fit to target tones using independent runtime settings.

    Parameters
    ----------
    runtime_context : tollan.config.RuntimeContext
        Runtime containing the root drive_fit_commit configuration.
    """

    def __call__(
        self,
        sources: Iterable[str | Path | FileLoc],
        *,
        reference: Path | DataProdIndex | None = None,
    ) -> xr.DataTree:
        """Match, normalize, optionally plot and publish one target acquisition.

        Parameters
        ----------
        sources : iterable of str, pathlib.Path or FileLoc
            One target sweep per interface of the same acquisition.
        reference : pathlib.Path, DataProdIndex or None, optional
            Existing drive-fit product. None uses the configured reference path.

        Returns
        -------
        xarray.DataTree
            Working collection with commit/plot contexts and optional output path.
            Settings files are published as artifacts. Explicitly enabled
            tlaloc_output applies them after successful product output/registration.
        """
        reference = reference if reference is not None else self.config.reference
        if reference is None:
            msg = "Supply an existing drive-fit reference product"
            raise ValueError(msg)
        data = prepare_drive_fit_commit_inputs(sources, reference)
        steps, snapshots, tlaloc_steps = {}, {}, {}
        for interface, tree in data["interfaces"].children.items():
            cfg = type(self)(
                self._rc.with_config(
                    validation_context={
                        str(k): v for k, v in tree["inputs"].attrs.items()
                    }
                )
            ).config
            if cfg.output != self.config.output:
                msg = "Output settings must agree across target interfaces"
                raise ValueError(msg)
            steps[interface] = [
                DriveFitCommitStep(cfg.commit),
                DriveFitCommitPlotStep(cfg.plot),
            ]
            snapshots[interface] = cfg.model_dump(mode="json")
            tlaloc_steps[interface] = TlalocOutput(cfg.tlaloc_output)
        data.attrs["config"] = snapshots
        _execute_workflow(
            data,
            self._rc,
            self.config.output,
            make_drive_fit_commit_product_index(data),
            steps,
        )
        for interface, step in tlaloc_steps.items():
            if step.config.enabled:
                step(data[f"interfaces/{interface}"])
        return data


def run(runtime: RuntimeContext) -> Iterator[xr.DataTree]:
    """Run the commit workflow on one discovered target acquisition.

    Parameters
    ----------
    runtime : RuntimeContext
        Root drive_fit_commit settings, reference product and input_discovery.

    Yields
    ------
    xarray.DataTree
        The target collection with completed commit and optional plot/output.
    """
    handler = DriveFitCommit(runtime)
    selected = _discover(runtime, handler.config.select)
    yield handler(selected["source"])


def open_drive_fit_commit_product(
    path: Path | DataProdIndex,
    *,
    interface: str | None = None,
    location: str | None = None,
    cache: Path | None = None,
) -> xr.DataTree:
    """Open saved commit settings and matching diagnostics.

    Parameters
    ----------
    path : pathlib.Path or DataProdIndex
        Product directory, pinned index or descriptor from DPDB.
    interface : str or None, optional
        Required when the product contains several readout interfaces.
    location : str or None, optional
        Registered source copy to select when several exist.
    cache : pathlib.Path or None, optional
        Persistent cache for immutable remote arrays.

    Returns
    -------
    xarray.DataTree
        Lazy scientific interface tree; close after reading its values.
    """
    index = _product_index(path)
    if index.meta["data_prod_type"] != "dp_drive_fit_commit":
        msg = "Expected a dp_drive_fit_commit product"
        raise ValueError(msg)
    return _open_interface(index, interface=interface, location=location, cache=cache)
