# Configured sweep reduction

The reducer-only KIDs pipeline converts raw TolTEC sweep samples into per-step
means and standard deviations. Results use the existing schema, mapper, and
view APIs and can be reopened independently of the raw file.

## Command line

To inspect inputs and validate configuration without reduction, see
[Inspect KIDs inputs and check a run](kids-inputs.md).

```console
tolteca_kids --config_paths recipes/ --kids.output.enabled true --dataprod_toltec_path products -- sweep.nc
tolteca_kids --config_paths recipes/ --show_config -- sweep.nc
tolteca_kids --kids.output.enabled true --dataprod_toltec_path products --kids.sweep_reduce.block 0 -- tune.nc
tolteca_kids --show_inputs -- '/data/lmt/toltec/tcs/toltec*/*_tune.nc'
tolteca_kids --data_lmt_path /data/lmt --kids.output.enabled true --dataprod_toltec_path products -- 123456/0
tolteca_kids --log_level DEBUG --data_lmt_path /data/lmt --dry_run -- 'tcs-123456/0'
```

`--config_paths` accepts YAML/dotenv files or directories of numbered files.
`--env_paths` accepts dotenv files or directories of numbered env files.
Both accept multiple values and repeated occurrences, in increasing precedence.
Place positional sources after `--` to delimit the lists.
Dotted `--kids.*` options are runtime overrides. Automatically derived option
names preserve their Python spelling, including underscores.
`--show_config` prints the effective, validated configuration without loading
I/Q samples or executing reduction. Enable the final output step with
`--kids.output.enabled true`; `--dataprod_toltec_path` supplies its parent
directory. The default product name is the input file stem;
existing product directories are never overwritten. Without output configuration,
reduction runs in memory and publishes no files.

Multiple files, quoted globs, and [observation selectors](obsspec.md) are accepted.
The local resolver and database API share the same typed selector language.
Local selectors require `--data_lmt_path`; they do not require a database.

`--dry_run` resolves inputs and validates each input's effective configuration,
then executes a real no-op step through `SequentialPipeline`. The step attaches
its live context and marks it completed, without reading signal arrays or writing
products—even when output is enabled in the configuration. This checks the
input/configuration/execution path, not scientific processing or export. It also
works for NetCDF timestream inputs whose processing pipeline is not implemented.

Use `tolteca_kids --log_level DEBUG ...` to see discovery counts, normalized
selectors, scalar input metadata, config-source conditions and their decisions,
effective configuration, selected steps, completion state, failures, and timings.
Debug output includes paths and configuration values; treat saved logs accordingly.
Array contents are not logged. `--show_inputs` and `--show_config` remain inspection
commands that exit before running the pipeline.

`--show_inputs` prints the metadata catalog. `--select` filters rows with a
pandas expression, for example `"roach < 7 & file_suffix in ['tune', 'targsweep']"`.
This selects **inputs**; config-source `enable_if` selects **configuration** for
each retained input. Both expressions are trusted local configuration, not an
interface for evaluating untrusted user submissions.

Only raw-sweep reduction and its product-output stage are supported here.
Timestream and processed-file kinds are recognized, but their execution raises
an explicit unsupported-workflow error; they are not fed into the raw reducer.
Sweep checking, detection, fitting,
and instrument output are not invoked. Unknown settings and unsupported stage
sections are rejected; full legacy KIDs recipes cannot yet be used unchanged.

## Configuration

```yaml
kids:
  sweep_reduce:
    enabled: true
    block: -1
    force: false
  output:
    enabled: true
    path: products
    name: '{stem}'
```

The stage always detects blocks and computes standard deviations. All detected
blocks are reduced and retained. `block=-1` selects the last block for consumption, matching
the legacy reader default; nonnegative values select an explicit block.
`force` requests recomputation of an existing result in a working tree.
`output.name` can use source fields, for example
`'{obsnum:06d}-{subobsnum:03d}-{scannum:04d}-{roach}-{file_suffix}'`.
It must expand to one directory component. Explicit `kids.output.path` settings
take precedence over the shared product-root default. The shared root alone
does not enable output.

Source-list YAML supports conditional configuration, for example:

```yaml
- source:
    kids:
      sweep_reduce:
        block: -1
- enable_if: roach >= 7 & file_suffix in ['targsweep', 'tune']
  source:
    kids:
      sweep_reduce:
        block: 0
```

Shared `tolteca_datamodels.file` records describe source paths, identifiers,
master, timestamp, suffix, and data kind. Filename discovery does not open files.
Catalog preparation enriches records with available scalar KIDs header metadata;
header identifiers and kinds correct filename guesses. Original-filename
consistency checking is opt-in.
Missing fields remain absent, not fabricated zeros. Each input receives its
own runtime context variant. The resulting handler keeps its validated config
snapshot; subsequent runtime changes do not alter it.

Included source filenames are relative to their containing YAML file. Ordinary
path-valued config data is resolved during validation against the common
`rootpath`, not separately against each included file's directory.

## Python execution and ownership

`Kids` owns configured steps and assembles the pipeline appropriate for the
input kind. The runner owns input-file lifetimes; `DataProdOutput` owns
publication policy and delegates serialization to the product writer.
The numerical reducer remains usable without any of these layers.

```python
from pathlib import Path

import xarray as xr
from tollan.accessor.xarray import ensure_datatree
from tollan.config import RuntimeContext
from tolteca_datamodels.file import SourceInfoModel
from tolteca_datamodels.toltec.kids.source import identify_source
from tolteca_kids.runner import create_kids
from tolteca_kids.sweep_reduce import SweepReduce

source = Path("sweep.nc")
rc = RuntimeContext.from_cli(config_paths=[Path("recipes")])
with xr.open_dataset(source) as raw:
    input_info = identify_source(raw, SourceInfoModel(source=str(source)))
    context_input = input_info.model_dump(
        mode="json", context={"flat": True},
        exclude={"filepath_info", "filepath_info_orig"},
    )
    kids = create_kids(rc, context_input=context_input)
    working = ensure_datatree(raw)
    working['tolteca_kids'] = xr.DataTree()
    working['tolteca_kids'].attrs['input'] = input_info.model_dump(
        mode='json', round_trip=True,
    )
    kids.pipeline_for(input_info.data_kind)(working)
    result = SweepReduce.view(working)
    selected = result.selected_block
    I = selected.I
    frequencies = selected.frequency  # Hz, with the same dimensions as I
```

The pipeline returns the same working tree. Steps add or replace their own
result nodes and preserve unrelated siblings. Array buffers shared with the
raw input are not modified. Keep the input open while using raw-backed metadata
in the working tree, or export the result before closing it.

Each step attaches a fresh live context before it runs. During execution,
`SweepReduce.get_context(working)` returns that invocation. Successful execution
stores its JSON-compatible metadata beside the results. If reduction raises,
the current context stays incomplete; existing array buffers may remain for
inspection, but the product writer refuses to publish them. Check `completed`
before treating a retained result as the outcome of the latest invocation.

The output step follows the same rule: a failed publication has an incomplete
context and no new `product_path`. An earlier published product is not changed.

Standalone execution uses the same contract on an open working tree:

```python
step = SweepReduce(block=0)
working, context = step(working, return_context=True)
assert context.completed
```

Internally, `run(data, context)` returns a Boolean completion indicator; the
public step call returns the original container. The pipeline skips disabled
steps, whereas calling a step directly executes it. An enabled output step
requires a completed reduction; disabling reduction without providing an
existing result does not yield a valid product.

For numerical-only use:

```python
from tolteca_datamodels.toltec.kids import ReducedSweepView, SweepReducer

with xr.open_dataset(source) as raw:
    result = SweepReducer(compute_uncertainty=True)(raw)
    sweep = ReducedSweepView(result)
```

Inspect the complex sweep response with the reduced view:

```python
import numpy as np
from tolteca_datamodels.toltec import open_toltec

with open_toltec("toltec0_145606_000_0001_2025_10_31_09_22_14_targsweep.nc") as ds:
    tree = SweepReducer()(ds)
    sweep = tree.toltec_kids.sweep
    S21 = sweep.I + 1j * sweep.Q
    S21_db = 20 * np.log10(np.abs(S21))
    frequency = sweep.frequency
```

Reduction reads detector data. KIDs accessors expose typed metadata and native
arrays such as `f_tone` and `tone_mask` before reduction. A flat tune's full
metadata scans LO frequency to identify recorded sweep blocks; filename/catalog
identification does not. Inspect one block's metadata explicitly with:

```python
with open_toltec("toltec0_145609_000_0001_2025_10_31_09_26_59_tune.nc") as ds:
    meta = ds.toltec_kids.mapper.get_metadata(ds, block=1)
    print(meta.n_times, meta.t_exp)
```

The sample count and integration time describe that block. Acquisition start/end
timestamps still describe the containing file. `n_blocks_max` is the allocated
capacity, not the count of recorded blocks. An explicitly block-indexed dataset
returns a list of metadata objects.

## Products and result views

```python
from tolteca_kids.product import open_sweep_product
from tolteca_datamodels.toltec.filestore import build_input_catalog
from tolteca_kids.runner import run_source

catalog = build_input_catalog([source])
inputs = catalog.toltec_file.to_info_list("roach == 0")
rc_output = rc.with_config(overrides={
    "kids": {"output": {"enabled": True, "path": "products"}}
})
path = run_source(inputs[0], rc_output)
assert path is not None
with open_sweep_product(path) as tree:
    result = SweepReduce.view(tree)
    config = result.context.config
    I = result.selected_block.I.load()
```

A product directory contains:

- `arrays.zarr`: the reduced result node, its required coordinates and copied
  non-time metadata. It does not contain the raw I/Q root.
- `manifest.json`: the product format, structured input record, array-store
  location, creation time, effective pipeline configuration, and software
  versions. It is written last, after a
  metadata reopen check.

Signal and uncertainty arrays retain a declared input unit; bare inputs remain
bare. Frequency coordinates carry `Hz`. These attributes work with the shared
`.u` accessor after reopening.

The step's effective config, selected block, and completion state are stored as
JSON-compatible metadata on the reduced node. `SweepReduceView.context`
reconstructs the typed context; arrays remain in the dataset, not duplicated
in a `StepContextData` payload. `selected_block` returns a `ReducedSweepView`;
its `sweep` property remains the frequency-offset coordinate.

On the live tree, `DataProdOutput.view(working).context` records the output
configuration, completion state, and product path. The manifest records the
publication settings durably; no open runtime or file objects are serialized.

Read product metadata without opening its arrays:

```python
from tolteca_kids.product import read_sweep_manifest

manifest = read_sweep_manifest(path)
source_info = manifest.input
config = manifest.config
```

For a pipeline-produced product, save the effective settings as a recipe and
replay with the original input. This does not re-evaluate the original recipe's
conditions:

```python
import json
from tolteca_kids.core import Kids

assert manifest.config is not None
Path('replay.yaml').write_text(json.dumps(Kids.prepare_runtime_config_data(
    manifest.config.model_dump(mode='json', round_trip=True),
)))
```

```console
tolteca_kids --config_paths replay.yaml --kids.output.name replayed -- sweep.nc
```

Choose a new output name or directory because existing products are not
overwritten. Direct `write_sweep_product()` calls can export in-memory results
without an input record or pipeline output config; those missing fields remain
`None` and cannot provide a complete pipeline replay recipe.

The writer uses Zarr format 2 to preserve fixed-byte netCDF headers
without an unstable dtype extension. Readers reject missing manifests and
unsupported product formats. A failed write can leave an incomplete directory
for inspection; it is not a published product. The manifest is a completion
marker, not a checksum or a distributed transaction guarantee.

Database registration and Web catalog projection are separate consumers. They
must only publish completed products. Reduction itself does not require a
database or orchestration service.
