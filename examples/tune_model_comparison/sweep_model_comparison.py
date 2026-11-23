"""Compare saved legacy and comprehensive fits on identical measured sweeps.

The paired unit is a group/channel response, not an independently recovered
optical detector. All APT-good channels stay in the coverage ledger, including
those without a selected response. No refitting, gain rescaling or best-model
selection is performed here.
"""

# ruff: noqa: INP001 -- standalone comparison recipe helpers
# ruff: noqa: T201 -- network progress during the local audit

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from astropy.table import Table

from tolteca_datamodels.data_prod import DataProdIndex
from tolteca_kids.model_fit import ModelFit
from tolteca_kids.product import open_kids_product
from tolteca_kidsproc.model_flags import ModelFitFlag
from tolteca_kidsproc.models import KidsSweepGainWithLinTrend


def residual_metrics(measured, predicted):
    """Describe residuals against measured excursion, with no fitted rescaling."""
    if len(measured) < 2 or not np.isfinite(predicted).all():  # noqa: PLR2004
        return {"rms": np.nan, "fraction": np.nan}
    rms = float(np.sqrt(np.mean(abs(measured - predicted) ** 2)))
    excursion = float(np.sqrt(np.mean(abs(measured - np.mean(measured)) ** 2)))
    return {"rms": rms, "fraction": rms / excursion if excursion > 0 else np.nan}


def compare_response(samples, old_prediction):
    """Use every finite data-valid sample in the saved response window."""
    measured = samples.I.values + 1j * samples.Q.values
    new = samples.I_model.values + 1j * samples.Q_model.values
    keep = samples.data_valid.values & np.isfinite(measured)
    result = {
        "sample_count": int(keep.sum()),
        "fit_sample_count": int(sum(keep & samples.fit_selected.values)),
        "legacy_invalid_predictions": int(sum(keep & ~np.isfinite(old_prediction))),
        "v3_invalid_predictions": int(sum(keep & ~np.isfinite(new))),
    }
    for name, values in (("legacy", old_prediction), ("v3", new)):
        result.update(
            {
                f"{name}_{k}": v
                for k, v in residual_metrics(measured[keep], values[keep]).items()
            }
        )
    return result


def exclusion_reason(flag, fit_status, probe_status, recovery_mode):
    """Distinguish optimizer failure from binding and recovery state."""
    if not flag & int(ModelFitFlag.fit_unusable):
        return (
            "joint_output_not_mapped"
            if flag & int(ModelFitFlag.fit_joint_required)
            else "map_eligible"
        )
    if flag & int(ModelFitFlag.fit_no_model):
        return "no_model_at_probe"
    if probe_status != "bound" or fit_status == "not_selected":
        return "probe_binding_unresolved"
    if fit_status != "optimizer_converged":
        return "optimizer_not_converged"
    if recovery_mode in {"invalid", "unresolved"}:
        return "recovery_unresolved"
    return "other_unusable"


def paired_summary(rows):
    """Summarize identical-window residuals without discarding missing models."""
    old, new = np.asarray(rows["legacy_fraction"]), np.asarray(rows["v3_fraction"])
    finite = np.isfinite(old) & np.isfinite(new) & (old > 0)
    return {
        "responses": len(rows),
        "finite_pairs": int(finite.sum()),
        "v3_smaller_residual": int(sum(new[finite] < old[finite])),
        "legacy_median_fraction": float(np.median(old[finite]))
        if finite.any()
        else None,
        "v3_median_fraction": float(np.median(new[finite])) if finite.any() else None,
        "median_paired_ratio": float(np.median(new[finite] / old[finite]))
        if finite.any()
        else None,
    }


def summarize(channels, responses):
    """Keep sweep prediction, fixed-probe selection and map gates separate."""
    arrays = {}
    for array in np.unique(channels["array"]):
        c = channels[(channels["array"] == array) & (channels["apt_flag"] == 0)]
        r = responses[(responses["array"] == array) & (responses["apt_flag"] == 0)]
        selected = r[r["selected_for_probe"]]
        reason, counts = np.unique(c["exclusion_reason"], return_counts=True)
        arrays[str(int(array))] = {
            "apt_good": len(c),
            "has_fitted_response": int(sum(c["candidate_responses"] > 0)),
            "has_converged_response": int(sum(c["converged_responses"] > 0)),
            "has_selected_response": int(sum(c["selected_response_exists"])),
            "selected_optimizer_converged": int(
                sum(
                    c["selected_response_exists"]
                    & (c["fit_status"] == "optimizer_converged")
                )
            ),
            "map_eligible": int(sum(c["eligible"])),
            "exclusion_reason": dict(
                zip(map(str, reason), map(int, counts), strict=True)
            ),
            "selected_responses": paired_summary(selected),
            "all_responses": paired_summary(r),
            "selected_by_map_status": {
                str(status): paired_summary(selected[selected["status"] == status])
                for status in np.unique(selected["status"])
            },
        }
    return arrays


def build(prepared, legacy_manifest, output):  # noqa: PLR0915 -- saved-data audit
    """Read saved final-block fits and original tune.txt files; write an audit."""
    receipt = json.loads((prepared / "prepared.json").read_text())
    legacy = json.loads(legacy_manifest.read_text())
    old_sources = {r["interface"]: r for r in legacy["model_equivalence"]}
    channels = Table.read(prepared / "channel-coverage.ecsv")
    model = DataProdIndex.read(receipt["model_index"])
    all_rows, channel_rows, sources = [], [], []
    for item in receipt["context"]["data_items"]:
        interface = item["interface"]
        nw = int(interface.removeprefix("toltec"))
        [sweep] = [c for c in item["cal_items"] if c["type"] == "sweep"]
        [cal] = sweep["data_items"]
        source = old_sources[interface]
        path = Path(source["original"])
        old = Table.read(path, format="ascii.ecsv")
        labeled = Table.read(source["labeled"], format="ascii.ecsv")
        for name in old.colnames:
            np.testing.assert_array_equal(old[name], labeled[name])
        old["chan"] = labeled["chan"]
        old_lookup = {int(c): i for i, c in enumerate(old["chan"])}
        with open_kids_product(model, interface=interface) as tree:
            [block] = [
                b for b in tree.toltec_kids.blocks if b.meta.block == cal["block"]
            ]
            view = ModelFit.view(block.tree)
            samples, fit, props = (
                view.samples.load(),
                view.fit.load(),
                view.chan_prop.load(),
            )
            raw = block.tree["sweep"].to_dataset(inherit=False).load()
            np.testing.assert_equal(
                old.meta["Header.Toltec.AccumLen"], raw.attrs["iq_accumlen"]
            )
            raw_lookup = {int(c): i for i, c in enumerate(raw.iqlen.values)}
            indices = np.asarray([raw_lookup[int(c)] for c in samples.chan.values])
            sample_indices = samples.sample_index.values
            for name in ("I", "Q", "frequency"):
                np.testing.assert_array_equal(
                    samples[name], raw[name].values[indices, sample_indices]
                )
            # Evaluate the historical equation once for every saved sample.
            old_indices = [old_lookup[int(c)] for c in samples.chan.values]
            prediction = KidsSweepGainWithLinTrend.from_table(old[old_indices]).eval(
                samples.frequency.values
            )
            pairs = list(
                dict.fromkeys(
                    zip(samples.group_id.values, samples.chan.values, strict=True)
                )
            )
            membership = {}
            for gid in np.unique(fit.group_id):
                members = fit.isel(row=np.flatnonzero(fit.group_id.values == gid))
                membership[str(gid)] = (
                    len(np.unique(members.chan)),
                    len(np.unique(members.resonator_id)),
                    str(members.fit_status.values[0]),
                )
            ledger = {int(r["channel"]): r for r in channels[channels["nw"] == nw]}
            local = []
            for gid, chan in pairs:
                where = np.flatnonzero(
                    (samples.group_id.values == gid) & (samples.chan.values == chan)
                )
                measured = samples.isel(sample=where)
                assert len(np.unique(measured.sample_index)) == len(where)
                prop = props.sel(chan=chan)
                m, n, state = membership[str(gid)]
                row = dict(
                    **{k: ledger[int(chan)][k] for k in channels.colnames},
                    group_id=str(gid),
                    fit_channels=m,
                    fit_resonators=n,
                    fit_status=state,
                    selected_for_probe=str(prop.group_id.item()) == str(gid),
                    legacy_flag=int(old["flag"][old_lookup[int(chan)]]),
                    **compare_response(measured, prediction[where]),
                )
                local.append(row)
            all_rows.extend(local)
            for chan, base in ledger.items():
                prop = props.sel(chan=chan)
                candidates = [r for r in local if r["channel"] == chan]
                selected = [r for r in candidates if r["selected_for_probe"]]
                assert len(selected) <= 1
                state, probe, recovery = (
                    str(prop[k].item())
                    for k in ("fit_status", "probe_status", "recovery_mode")
                )
                channel_rows.append(
                    dict(
                        **{k: base[k] for k in channels.colnames},
                        fit_status=state,
                        probe_status=probe,
                        recovery_mode=recovery,
                        recovery_reason=str(prop.recovery_reason.item()),
                        candidate_responses=len(candidates),
                        converged_responses=sum(
                            r["fit_status"] == "optimizer_converged" for r in candidates
                        ),
                        selected_response_exists=bool(selected),
                        exclusion_reason=exclusion_reason(
                            int(base["model_flag"]), state, probe, recovery
                        ),
                    )
                )
        sources.append(
            {
                "interface": interface,
                "block": cal["block"],
                "legacy": str(path),
                "legacy_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "accumlen": int(old.meta["Header.Toltec.AccumLen"]),
                "measured_samples_match_saved_sweep": True,
            }
        )
        print(f"{interface}: audited {len(local)} group/channel responses", flush=True)
    responses, channel_table = Table(rows=all_rows), Table(rows=channel_rows)
    output.mkdir(parents=True, exist_ok=True)
    meta = {
        "sample_domain": "finite data-valid points in each saved response window",
        "normalization": (
            "RMS measured complex IQ excursion about its mean, no baseline rescaling"
        ),
        "scope": (
            "in-sample sweep description, not held-out performance "
            "or timestream identifiability"
        ),
        "model_index": receipt["model_index"],
    }
    responses.meta.update(meta)
    responses.write(output / "sweep-model-responses.ecsv", overwrite=True)
    channel_table.write(output / "sweep-model-channels.ecsv", overwrite=True)
    result = {
        "method": meta,
        "sources": sources,
        "arrays": summarize(channel_table, responses),
    }
    (output / "sweep-model-quality.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("prepared", "legacy_manifest", "output"):
        parser.add_argument(key, type=Path)
    build(**vars(parser.parse_args()))
