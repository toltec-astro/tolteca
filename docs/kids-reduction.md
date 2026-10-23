# Configured KIDs reduction

The KIDs pipeline averages raw sweep repetitions with `SweepReduce`, and
analyses raw timestreams with the Python `TimestreamReduce`. The latter exposes
time signals and Welch spectra without invoking C++. Both use the schema,
mapper, and view APIs and write products that reopen independently.

## Command line

To inspect inputs and validate configuration without reduction, see
[Inspect KIDs inputs and check a run](kids-inputs.md).

```console
tolteca_kids --config_paths recipes/ --kids.output.enabled true --dataprod_toltec_path products -- sweep.nc
tolteca_kids --config_paths recipes/ --show_config -- sweep.nc
tolteca_kids --kids.output.enabled true --dataprod_toltec_path products --kids.sweep_check.enabled true -- tune.nc
tolteca_kids --show_inputs -- '/_data/lmt/toltec/tcs/toltec*/*_tune.nc'
tolteca_kids --data_lmt_path /_data/lmt --kids.output.enabled true --dataprod_toltec_path products -- 123456/0
tolteca_kids --log_level DEBUG --data_lmt_path /_data/lmt --dry_run -- 'tcs-123456/0'
```

`--config_paths` accepts YAML/dotenv files or directories of numbered files.
`--env_paths` accepts dotenv files or directories of numbered env files.
Both accept multiple values and repeated occurrences, in increasing precedence.
Place positional sources after `--` to delimit the lists.
Dotted `--kids.*` options are runtime overrides. Automatically derived option
names preserve their Python spelling, including underscores.
`--show_config` prints the effective, validated configuration without loading
I/Q samples or executing reduction. Enable the final output step with
`--kids.output.enabled true`; configure its parent directory with
`kids.output.path`. `--dataprod_toltec_path` supplies a default for that setting;
explicit recipe settings and `--kids.output.path` take precedence.
Products use `<root>/<namespace>/<acquisition>/dp_raw_obs_proc/`.
The default namespace is `scratch`; TCS is omitted from acquisition directory
labels. Selected interfaces of an acquisition are published together. Existing
run artifacts are preserved; subsequent invocations generate a new run UID. Without output configuration,
reduction runs in memory and publishes no files. If an enabled output step
records a failure, the CLI reports it, continues the selected inputs, and exits
with status 1.

Multiple files, quoted globs, and [observation selectors](obs_spec.md) are accepted.
The local resolver and database API share the same typed selector language.
Local selectors require `--data_lmt_path`; they do not require a database.

`--dry_run` resolves inputs and validates each input's effective configuration,
then executes a real no-op step through `SequentialPipeline`. The step attaches
its live context and marks it completed, without reading signal arrays or writing
products—even when output is enabled in the configuration. This checks the
input/configuration/execution path, not scientific processing or export. It also
works for both raw sweep and raw timestream inputs.

Use `tolteca_kids --log_level DEBUG ...` to see discovery counts, normalized
selectors, scalar input metadata, config-source conditions and their decisions,
effective configuration, selected steps, completion state, failures, and timings.
Debug output includes paths and configuration values; treat saved logs accordingly.
Array contents are not logged. `--show_inputs` and `--show_config` remain inspection
commands that exit before running the pipeline.

`--show_inputs` prints the input table. `--select` filters rows with a
pandas expression, for example `"roach < 7 & file_suffix in ['tune', 'targsweep']"`.
This selects **inputs**; config-source `enable_if` selects **configuration** for
each retained input. Both expressions are trusted local configuration, not an
interface for evaluating untrusted user submissions.

Raw-sweep averaging and Python raw-timestream spectral reduction are supported.
Optional sweep checks, diagnostic figures, KIDs finding, tone preparation and
Tlaloc file output are also supported. Processed-file execution, resonator fitting
and model-based timestream solving are not invoked by the configured pipeline.
Unknown settings and unsupported stage sections are rejected; full legacy KIDs recipes cannot yet be used unchanged.

## Configuration

```yaml
kids:
  sweep_reduce:
    enabled: true
    force: false
    compute_uncertainty: true
  output:
    enabled: true
    path: products
    namespace: scratch
    tag: main
    run: {}
```

The handler prepares all blocks and invokes reduction on each. Standard
deviations are included by default; set `compute_uncertainty=false` to omit them.
Select a block explicitly to consume its result or execution record. When calling
a reducer directly on an acquisition, its `block` setting selects the view in
that invocation context; -1 selects the last block. A block-scoped pipeline uses
the supplied block.
`force` requests recomputation of an existing result in a working tree. It does
not overwrite a saved run or allow a reserved run UID to be reused.

### Prepare the next Tlaloc tone list

Enable finding and tone preparation to include the four v2 tone files in an
ordinary sweep product. Enable `tlaloc_output` to also install them in an explicit
Tlaloc `etc` directory after that acquisition's product publication and configured
registration succeed:

```yaml
kids:
  sweep_check:
    enabled: true
  kids_find:
    enabled: true
  kids_find_plot:
    enabled: true
  tone_prepare:
    enabled: true
    tone_phases:
      method: random
      seed: 42
  output:
    enabled: true
    path: products
  tlaloc_output:
    enabled: true
    path: run/tlaloc-example/etc
```

The writer creates `toltecN/targ_freqs.dat`, `default_targ_amps.dat`,
`random_phases.dat` and `default_targ_masks.dat`. They share one tone order.
Amplitudes transfer from recorded input probes through finding's tone match;
phases are in radians. Standalone VNA and target sweeps can change the tone count.
Tune preserves each block's input count by retaining unmatched probes as needed.
Only the last sweep block's files are installed; products retain every prepared
block. These steps do not upload tones or change `atten_drive.dat`. Ordinary
tone upload retains the current global drive attenuation; changing the tone
count or relative amplitudes redistributes the available comb power.

Finding also publishes `kids_find.ecsv` (one row per detection) and
`chan_prop.ecsv` (one row per enabled input probe), including match distances,
quality flags, amplitudes and detection statistics. Set `kids_find_plot.enabled`
to include seven interactive HTML diagnostics for peaks, detection statistics
and matches. Disable that step to omit the figures while retaining the tables
and prepared tone files.

Set `tone_prepare.fix_n_chans` to explicitly override the observing-mode default.
Use `tone_prepare.tone_phases: {method: file, path: /path/to/phases.dat}` to supply
headerless phases instead of generating them. Keep `tlaloc_output.enabled` false
to publish files without installing them. See [drive-fit settings](drive-fit.md#write-tlaloc-settings)
for the amplitude/attenuation-only commit handoff.

Prepared files can also be installed from a saved ordinary product, with the raw
acquisition closed:

```python
from pathlib import Path
from tolteca_kids.product import open_sweep_product
from tolteca_kids.tlaloc_output import TlalocOutput

with open_sweep_product(product_path) as saved:
    blocks = saved.toltec_kids.blocks
    block = blocks[len(blocks) - 1]
    TlalocOutput(enabled=True, path=Path("run/tlaloc-example/etc"))(block.tree)
```

The selected product must have tone preparation enabled. Installation reuses its
saved tone table, without finding, matching or generating phases again.

#### Fill ambiguous amplitudes from the calibration-group LUT

Amplitude policies live under `kids.tlaloc_output`, including when file
installation is disabled. `tone_amps_method: match` transfers input amplitudes;
`ones` explicitly assigns unity to detected tones. The former
`kids.tone_prepare.tone_amps_method` key is no longer accepted.
Regenerate ordinary-sweep products whose saved configuration uses that former
key before reopening them with this revision.

Leave `tone_amps_fallback: null` to reproduce v2, including its ambiguous matches.
Use `median` to fill ambiguous and unavailable matches with the median enabled
input amplitude. Use `lut` to use the calibration-group LUT:

```yaml
kids:
  sweep_check: {enabled: true}
  kids_find: {enabled: true}
  tone_prepare: {enabled: true}
  tlaloc_output:
    enabled: false
    path: /path/to/tlaloc/etc
    tone_amps_method: match
    tone_amps_fallback: lut
  output:
    enabled: true
    path: products
```

Place the headerless two-column RF-frequency (Hz), amplitude file at
`/path/to/tlaloc/etc/toltecN/amps_correction_lut.csv`. Use the LUT from the VNA
sweep's calibration group. The reduction uses unambiguous positive input probes
to estimate one median input/LUT ratio in dB, and applies that scale to the LUT
values at all ambiguous detections. No member of an ambiguous group inherits
the probe amplitude. Unique matches keep their relative coefficients. If any
coefficient exceeds one, one common divisor rescales the entire output comb.
Missing LUTs or absent usable scale anchors stop preparation.

The saved `tlaloc` dataset includes `amp_tone_matched`, `amp_tone_ambiguous`,
`amp_tone_filled` and `tone_power_fraction`. Its attributes record the LUT hash,
source, scale, anchor count and common divisor. The power fractions use the
same fixed-total approximation as drive-fit commit, including duplicate DAC
bins. They are not calibrated detector-input powers. Changes in tone count or
fallback amplitudes can shift the power of otherwise unchanged tones because
ordinary upload retains the existing global attenuation. Reopening and
installing the product uses its saved amplitudes even if the LUT later changes.

### Check sweeps and publish diagnostic plots

Enable checks after sweep reduction, and enable plots to include standalone HTML
in the product:

```yaml
kids:
  sweep_check:
    enabled: true
    despike: true
  sweep_check_plot:
    enabled: true
  output:
    enabled: true
    path: products
```

`SweepCheck` requires I/Q uncertainties from `SweepReduce`. The configured
workflow invokes it on each reduced sweep block. The original reduced sweep remains available
at `block.sweep`. The corrected sweep is `block.sweep_check.sweep`.

| Saved result | Contents |
| --- | --- |
| `sweep_check/sweep` | Despiked I/Q, original uncertainties and frequency axes |
| `sweep_check/samples` | Original complex samples, removed spikes, sample flags and baseline masks |
| `sweep_check/channels` | Channel flags, level, range and uncertainty statistics |
| `sweep_check/chunks` | Overlapping-window statistics, flags, baseline masks and sample bounds |
| `sweep_check/d21` | Unified D21, baseline, baseline RMS and detrended D21 |
| `sweep_check/d21_chunks` | D21-window statistics and frequency/sample bounds |

These paths are relative to each block, for example
`toltec0/blocks/0/sweep_check/channels` in the product Zarr. The check namespace
also retains its effective configuration. Readout-power modeling and raw-sample
noise PSD are not computed by this check.

To compute and inspect a check in Python, set `raw_path` to a raw sweep or tune
NetCDF file:

```python
import xarray as xr
from tollan.accessor.xarray import ensure_datatree
from tolteca_kids import SweepCheck, SweepCheckPlot, SweepReduce
from tolteca_kids.sweep_check_viz import S21FrequencyGridViz

with xr.open_dataset(raw_path) as ds:
    tree = ensure_datatree(ds)
    SweepReduce()(tree)
    block = tree["blocks/0"]
    SweepCheck()(block)
    check = SweepCheck.view(block)
    mask_bad = check.mask_chan_bad
    figure = S21FrequencyGridViz()(check, channels=[0, 1])
```

Pass the acquisition tree to `SweepCheck()(tree)` to check every reduced sweep
block. Pass a block node, as above, to check only that block. In an acquisition
invocation, `SweepCheck(block=0)` selects the view exposed in the context; it
still checks every reduced sweep block.

Numeric check properties load their arrays when accessed. For lazy selections,
use `check.mapper.get_arr(check.data_source, check.mapper.schema.bitmask_chan)`
and select channels before computing. Keep the source open until the requested
arrays or figure have loaded. `figure` owns its plotted values and can be used
after the source closes, including in a Dash `dcc.Graph`.

The same `SweepCheck.view(tree, block=0)` call reads checks from an
`open_kids_product(...)` tree. It does not require a live step context or rerun
analysis. `tree.toltec_kids.blocks[0].sweep_check` provides the block accessor.

| Viz | Display |
| --- | --- |
| `ChannelSummaryViz` | Channel quality flags and channel/chunk statistics |
| `S21FrequencyViz` | Corrected sweep amplitudes, uncertainty bars, D21 and baseline diagnostics |
| `S21FrequencyGridViz` | Amplitude and uncertainty panels; `page_size=25`, `columns=5` |
| `IQGridViz` | Overlaid IQ samples and uncertainties; `page_size=25` |

Construct one figure directly, or run the plot step on a checked block to retain
all four. Retrieve the resulting object and its effective configuration by Viz
class (or configured instance for an aliased figure):

```python
from tolteca_kids.sweep_check_viz import IQGridViz

SweepCheckPlot()(block)
context = SweepCheckPlot.get_context(block)
artifact = context.artifacts[IQGridViz]
figure = artifact.data
```

`context.config.artifacts` configures the four figures. The following settings
export HTML and Plotly JSON for each figure, with ten channels per IQ page:

```yaml
kids:
  sweep_check: {enabled: true}
  sweep_check_plot:
    enabled: true
    artifacts:
      save_formats: [html, json]
      i_q_grid: {page_size: 10, height: 600}
      s21_f_grid: {page_size: 10, columns: 2}
```

`DataProdOutput` collects completed steps' artifacts and writes them under
`artifacts/<interface>/blocks/<block>/<step key>/`. Each artifact appears in the
run index. HTML embeds Plotly and supports offline controls and channel paging;
JSON can be reopened with `plotly.io.read_json`. Scientific check data stays in
`array.zarr`. Disabling `sweep_check_plot` keeps those scientific results without
constructing figures. Setting `artifacts.save=False` retains live figures without
exporting them; an individual Viz's `save=False` excludes only that figure.

For immediate output, supply `save_path=Path("preview")` to `SweepCheckPlot`.
Use a distinct path per block. `show=True` displays the retained figures together
in a local Dash app; `artifacts.show_plotly` supplies its host and port.
Alternatively, call `artifact.spec.write(artifact.data, path=Path("preview/iq"))`
or `artifact.spec.show(artifact.data)` for one retained figure. These operations
use the loaded object without repeating the calculation. Product publication
still owns its own destination and atomic commit.

To replace a check in a working tree, invoke it with `force=True` or changed
calculation settings. Reinvoke `SweepCheckPlot` afterwards to replace its live
figures before publication. Existing saved runs remain immutable.

### Choose a namespace, tag and run name

| Setting | Use |
| --- | --- |
| `output.namespace` | Store workspace, such as `scratch`, `ql` or `taco` |
| `output.tag` | Selection of a completed product, such as `main` |
| `output.run.uid` | Stable invocation UUID, optionally supplied by an external driver |
| `output.run.name` | Namespace-unique name used for the run directory |

Every invocation has one UUID shared across its output products. With no run
settings, a UUID is generated and its text is also used as the directory name.
Naming choices do not change the run's identity:

| Configuration | Directory name |
| --- | --- |
| `run: {}` (default) | The generated run UUID |
| `run: {name: nightly}` | Exactly `nightly`; must be unused in the namespace |
| `run: {name: {generator: counter, key: nightly}}` | Next available value, such as `0000-nightly` |
| `run: {name: {generator: counter, value: 42, key: nightly}}` | Exactly `0042-nightly`; value 42 must be unused |

To retain an external invocation identity, set `run.uid` to its UUID alongside
any of these naming choices. The counter format defaults to `"{value:04d}-{key}"`.
An omitted key uses the run UUID. Counter values belong to the namespace across
all products and tags; literal names do not consume counter values. An unused
lower explicit value is allowed but never rewinds allocation. Run UIDs, names
and counter reservations remain consumed after failure.

`output.tag` defaults to `main`. Set another tag for a separate reduction flavor,
or `null` for an untagged run. The CLI and runners acquire that product's
tag before block processing; a busy tag records a `tag_busy` output failure.
Other tags and other products can process concurrently.
Placement uses acquisition metadata, including when processing an already-open
Dataset. Explicit `kids.output.path` settings take precedence over the shared
product-root default. The shared root alone does not enable output.

The CLI and `run_inputs` group selected interfaces by acquisition before writing.
Their output settings must agree; reduction settings can differ by interface.
Each publication contains the selected interfaces and computed results. It does
not merge with earlier runs or resume them. Products commit independently, even
when they share a run UID. One invocation can therefore publish
some acquisitions successfully while others fail.

### Combine configuration sources

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

File tables collect filename guesses and header values for selection.
Filename discovery does not open files. Header identifiers and data kinds
correct guesses before execution; missing required raw headers fail metadata
validation. The CLI skips or raises for affected selected inputs according to
the [input error policy](kids-inputs.md).
Each input receives
its own runtime context variant. The resulting handler keeps its validated
config snapshot; subsequent runtime changes do not alter it.

Included source filenames are relative to their containing YAML file. Ordinary
path-valued config data is resolved during validation against the common
`root_path`, not separately against each included file's directory.

## Python execution and ownership

`Kids` owns configured steps and assembles the pipeline appropriate for the
input kind. Use `run_source()` to open and close a local file, or `run_data()`
to process an already-open Dataset or DataTree and inspect the live results.
`DataProdOutput` owns publication policy and delegates serialization to the writer.
The numerical reducer remains usable without any of these layers.

```python
from pathlib import Path

import xarray as xr
from tollan.config import RuntimeContext
from tolteca_kids.runner import run_data
from tolteca_kids.sweep_reduce import SweepReduce

source = Path("sweep.nc")
rc = RuntimeContext.from_cli(config_paths=[Path("recipes")])
with xr.open_dataset(source) as raw:
    working = run_data(raw, source, rc)
    block = working["blocks/0"]
    context = SweepReduce.get_context(block)
    result = context.data
    assert result is not None
    I = result.I
    frequencies = result.frequency  # Hz, with the same dimensions as I
```

`run_data()` reads current headers for identification and conditional config;
it does not open the source location or close the supplied data. A Dataset is
wrapped without copying its arrays. A supplied DataTree is updated and returned
as-is, including its unrelated nodes. Steps add or replace their own results.
Keep the input open while using raw-backed metadata in the working tree, or
export the result before closing it. Raw array buffers are not modified.

`run_data()` identifies and creates the acquisition's blocks, runs the configured
pipeline for each block, then publishes the acquisition when output is enabled.
Use `run_inputs()` with a prepared open input table to process several interfaces
and publish them together. The CLI uses this batch runner.

```text
acquisition: BlockIdentify → BlockCreate
  each sweep block:      SweepReduce → [SweepCheck] → [SweepCheckPlot]
  each timestream block: TimestreamReduce
publication: DataProdOutput
```

Bracketed steps are optional and disabled by default. `kids.sweep_pipeline(block)`
and `kids.timestream_pipeline(block)` process a previously created single block;
they do not prepare or publish an acquisition.

The working tree has this layout. Select a block to invoke
further analyses or detach it for independent use:

```text
/                          acquisition attrs and live contexts
  raw                      native variables
  tolteca_kids              source identity and prepared input row
  blocks
    index, chan, probe     shared columnar tables (separate leaves)
    <block ID>
      sweep                reduced sweep, or
      timestream           time-domain projection
      sweep_check          optional corrected sweep, flags and statistics
      <analysis namespace> analysis results in separate leaves
```

Each processing context belongs to its input block. The acquisition root and
block scopes have no arrays, so analyses can use their own coordinates.

### Step results and execution state

| Operation | Live `context.data` | Other result fields |
| --- | --- | --- |
| `BlockIdentify` | Native `ToltecBlockIndex`, or None for an existing block structure | — |
| `BlockCreate` | Block collection or selected-block view | — |
| `SweepReduce` | `ReducedSweepView` of the selected block | `block`: acquisition block ID |
| `SweepCheck` | `SweepCheckView` of the selected block | `block`: acquisition block ID |
| `SweepCheckPlot` | None | Typed figures in `artifacts`; immediate export receipts in `files` |
| `TimestreamReduce` | `TimestreamView` of the selected block | `block`: acquisition block ID |
| `DataProdOutput` | Written product `Path` | Producing output settings in `config` |
| `NoOp` | None | Completion state only |

`Step.get_context(block)` returns a block processing invocation. Preparation and
product output contexts are attached to the acquisition tree. An aliased
invocation uses `Step.get_context(block, alias="repeat")`. Each run attaches a
fresh context before computation. Retained result views continue to expose that
invocation's arrays when later runs replace results; obtain the current context
for the latest run.

A normal return with no recorded failures marks a step complete. The sequence
continues after recorded failures; downstream steps decide whether the available
results are usable. Product output requires a completed reduction with a live
payload. Otherwise it records a `missing_input` failure and publishes no
scientific result.
An escaping exception is recorded and re-raised, stopping the sequence.
A completed reduction may contain lazy spectra; errors during export belong to
the output step that evaluates them.

### Save and restore a working tree

Pickle preserves live contexts and their views. This is separate from exporting a
Zarr product, which saves selected arrays and JSON execution metadata.

```python
import pickle

with xr.open_dataset(source) as raw:
    working = run_data(raw, source, rc)
    checkpoint = pickle.dumps(working)

restored = pickle.loads(checkpoint)
try:
    context = SweepReduce.get_context(restored["blocks/0"])
    assert context.data is not None
    I = context.data.I
    assert context.make_step().config == context.config
finally:
    restored.close()
```

A file-backed checkpoint may reopen its source lazily, so source files must remain
available at their recorded locations. No open file handle is required across
the round trip. For source-independent sharing, write a product with the provided
export functions.

### Run on borrowed inputs

`run_data` also accepts data borrowed from an open file table:

```python
from tolteca_kids.noop import NoOp
from tolteca_datamodels.toltec.file import guess_info_from_sources

table = guess_info_from_sources([source])
with table.toltec_file.open() as opened:
    selected_inputs = opened.toltec_file.query("roach == 0")
    for row in selected_inputs.itertuples(index=False):
        checked = run_data(row.io_obj, row.source, rc, dry_run=True)
        assert NoOp.get_context(checked).completed
```

The table scope owns these files. `run_data()` reuses them and applies the
configuration to each input without opening another handle. Header identification
still runs for each invocation; the table's earlier metadata is not an execution
cache. For many local files, `run_source()` processes one at a time without
keeping the whole input table open.

Each step attaches a fresh live context before it runs. During execution,
`SweepReduce.get_context(working)` returns that invocation, including after
completion. Product export writes a JSON snapshot without changing
the live context. If reduction raises,
the current context stays incomplete; existing array buffers may remain for
inspection, but the product writer refuses to publish them. Check `completed`
before treating a retained result as the outcome of the latest invocation.

The output step follows the same rule: a failed publication has an incomplete
context and `data=None`. An earlier published product is not changed.
Products retain the reduction's JSON metadata but omit Tollan's reserved live
context store. That store remains available on the working tree. Other exported
attributes must be JSON-compatible; unsupported objects cause publication to fail.

For an aliased step, use `Step.get_context(working, alias="repeat")` to read
its live context. Calling `make_step()` on it reconstructs the same alias and configuration.
Unforced calls reuse that alias's existing numerical result; a first invocation
may reuse the primary result. Set `force=True` to recompute, including after
changing raw input values on an existing working tree.

### Run steps directly

Standalone execution uses the same contract on an open working tree:

```python
step = SweepReduce(block=0)
working, context = step(working, return_context=True)
assert context.completed
```

A step call returns the original container. The pipeline skips disabled
steps, whereas calling a step directly executes it. An enabled output step
requires a completed reduction; disabling reduction without providing an
existing result does not yield a valid product.

### Reduce an open acquisition

Wrap the Dataset with Tollan and invoke the reduction step:

```python
from tollan.accessor.xarray import ensure_datatree
from tolteca_kids import BlockCreate, SweepReduce

with xr.open_dataset(source) as raw:
    result = ensure_datatree(raw)
    SweepReduce(compute_uncertainty=True)(result)
    sweep = result.toltec_kids.blocks[0].sweep
```

Inspect the complex sweep response with the reduced view:

```python
import numpy as np
from tolteca_datamodels.toltec import open_toltec_kids

with open_toltec_kids("toltec0_145606_000_0001_2025_10_31_09_22_14_targsweep.nc") as ds:
    tree = ensure_datatree(ds)
    BlockCreate()(tree)
    SweepReduce()(tree)
    sweep = tree.toltec_kids.blocks[0].sweep
    S21 = sweep.I + 1j * sweep.Q
    S21_db = 20 * np.log10(np.abs(S21))
    frequency = sweep.frequency
```

`SweepReduce` reduces every sweep block in an acquisition or collection. To
reduce one block, pass `tree["blocks/0"]` to the step. Existing trees update in
place; matching settings reuse results unless `SweepReduce(force=True)` is used.
Both reducers invoke `BlockCreate` when the working tree has no block structure.
Creation invokes `BlockIdentify` if an index has not been published yet.
Existing structures are reused. Identification reads LO samples once. Averaging
uses the identified probe bounds and eagerly reads each selected block's I/Q.

For arrays without acquisition metadata, provide the repetition intervals:

```python
I = xr.DataArray([1., 3., 5., 7.], dims="sample").u.set("adu")
Q = xr.DataArray([2., 4., 6., 8.], dims="sample").u.set("adu")
from tolteca_kidsproc.reduction import reduce_sweep_samples

reduced = reduce_sweep_samples(
    I, Q, sample_dim="sample",
    sample_slices=[slice(0, 2), slice(2, 4)],
)
assert reduced.I.values.tolist() == [2., 6.]
```

`reduce_sweep_samples` returns I/Q means and optional population standard deviations
on a `sweep` dimension. The caller supplies frequency coordinates when needed.
Input units are retained; NaNs propagate. Configure the same calculation within
the pipeline with `kids.sweep_reduce.compute_uncertainty`.

## Acquisition, blocks, and analysis subtrees

An opened native Dataset contains all file variables. Its
`ds.toltec_kids.meta` reads only acquisition headers and dimensions.
`ds.toltec_kids.block_index` inspects block structure on first access. Its
`.meta` includes the acquisition metadata and discovered block count. Inspection
reads tone headers and LO samples, leaving detector I/Q lazy.
`BlockIdentify` publishes this index in its context while native
variables stay in place. `BlockCreate` uses it to install the block structure
on the working DataTree.

The identified metadata exposes `block_index` as a Dataset and
`block_index_table` as a QTable, alongside `chan_axis_table` and
`probe_axis_table`. The QTables are materialized on first access and reused.
These properties do not require `BlockCreate`.

Import `tolteca_kids` for processing-tree access:

| Object | Metadata | Data access |
| --- | --- | --- |
| Native Dataset | `.toltec_kids.meta` | `.toltec_kids.block_index` inspects blocks |
| Acquisition tree | `.toltec_kids.meta` | `.toltec_kids.blocks[i]` selects a block |
| `blocks` subtree | `.toltec_kids.meta` | `.toltec_kids.blocks.block_index_table` lists blocks |
| Block subtree | `.toltec_kids.meta` | `.toltec_kids.block.raw`, `.sweep`, `.timestream` |
| Sweep leaf | `.toltec_kids.meta` | `.toltec_kids.sweep`; D21 results belong to the block scope |
| Timestream subtree | `.toltec_kids.meta` | `.toltec_kids.timestream` |
| D21 subtree | `.kids.d21.d21_analysis` (producing analysis) | `.kids.d21`, `.toltec_kids.d21` |

Reduced views require the corresponding result at the selected scope. An
acquisition can contain several sweep results, so select a block explicitly.

```python
from tolteca_datamodels.toltec import open_toltec_kids
from tollan.accessor.xarray import ensure_datatree
from tolteca_kids import BlockCreate, BlockIdentify, SweepReduce
from tolteca_kidsproc.analysis import D21Analysis

with open_toltec_kids("toltec0_145609_000_0001_2025_10_31_09_26_59_tune.nc") as ds:
    tree = ensure_datatree(ds)
    BlockIdentify()(tree)
    identified = BlockIdentify.get_context(tree).data
    acquisition = identified.meta
    bounds = acquisition.block_index       # still no blocks child
    index_table = acquisition.block_index_table
    channels = acquisition.chan_axis_table # n_blocks * n_chans rows
    probes = acquisition.probe_axis_table  # all recorded LO dwells
    BlockCreate()(tree)
    blocks = tree.toltec_kids.blocks
    index = blocks.block_index_table       # small, eager QTable
    second = blocks[1]
    meta = second.meta                    # complete inherited acquisition fields
    block_channels = meta.chan_axis_table
    block_probes = meta.probe_axis_table
    raw = second.raw                      # lazy native slice
    params = second.model_params          # lazy columnar Dataset, or None
    params_table = second.model_params_table  # QTable, loaded on demand

    SweepReduce()(tree["blocks/1"])
    tree = D21Analysis()(
        tree["blocks/1"].kids.sweep, output=tree, parent_namespace="blocks/1"
    )
    sweep = tree["blocks/1"].toltec_kids.sweep
    saved_settings = SweepReduce.get_context(tree["blocks/1"]).config
    derivative = tree["blocks/1"].toltec_kids.d21.d21
```

The tree has attributes-only containers. Array dimensions belong to independent
leaves, allowing different blocks and analyses to have different sample counts:

```text
/                          acquisition attrs: meta
  raw                      native file variables
  blocks                   collection attrs: acquisition meta + n_blocks
    index                  block, data_kind, sample/chan/probe start/end, sweep settings
    chan                   complete channel columns
    probe                  complete probe columns
    0                      block attrs: complete meta, data kind
    1                      block attrs: complete meta, data kind
      sweep                reduced Dataset with meta and reducer configuration
      tolteca_kidsproc.analysis.d21  attrs: namespaced analysis configuration
        d21                matched derivative Dataset
        d21_unified        unified derivative Dataset
```

`blocks/index` contains one row per block with exclusive bounds for native
samples and shared axis rows. On reopening, `.blocks` eagerly loads this small
index and its collection metadata; it does not load axes, detector arrays or all
block records. Selecting a block parses only that block's metadata. Its axes
use indexed slices, so selecting one block does not scan the full axis tables.
QTable properties materialize their selected columns on demand; columnar
`second.chan_axis.data_source` and `second.probe_axis.data_source` remain lazy.
The generic QTable conversion lives in Tollan and preserves numeric columns,
`units` attributes, and boolean masks without a special table serialization.

`ToltecRawObsBlocksMetadata` extends acquisition metadata with the discovered
block count and access to the index and shared axes. Each block's
`ToltecRawObsBlockMetadata` inherits those fields;
`ToltecSweepMetadata` and `ToltecTimestreamMetadata` describe its data kind.
`n_times` and `t_int` describe the block; acquisition timestamps and `t_elapsed`
describe the file. `n_blocks_max` is allocated capacity, while `n_blocks` counts
recorded blocks. `n_sweepreps` is the programmed acquisition setting;
`probe_axis_table["n_samples"]` records actual repetitions, including partial
steps. Probe `step` is local to a block; sample start/end bounds refer to the
acquisition. Channel count is fixed across blocks. Tone amplitudes come from
`Header.Toltec.ToneAmp`, as recorded in `tolteca_ref_data`; block-axis construction
requires that header and does not substitute unity amplitudes.

Use `ds.toltec_kids.select_channels(...)` to slice readout samples and all
channel headers together. It seeds original channel IDs before selecting and
retains them through repeated selections, reductions and exports. Slices,
integer lists and boolean arrays select positions within the current subset;
`"tone_mask"` selects the named boolean table column. For example,
`select_channels([10, 0, 5])` produces channel IDs `[10, 0, 5]`, not `[0, 1, 2]`.
Directly slicing an unlabelled native dimension with xarray before this step
cannot preserve source identities that have already been discarded.

Recorded model parameters belong to each block. `block.model_params` is a lazy
Dataset with one variable per recorded parameter and original IDs in its `chan`
coordinate. `block.model_params_table` adds `chan` as the first QTable column,
followed by the names in `Header.Toltec.ModelParamsHeader` order. These are the
recorded values; as in v2, no units or model fits are inferred. Both properties
return `None` when there are no model headers (for example, the reference VNA
sweep). Incomplete or malformed headers raise `ValueError`.

On a block tree the same properties are available through
`block_tree.toltec_kids.model_params` and `.model_params_table`. Select the block
explicitly from an acquisition; there is no implicit first-block model table.
Column names are read when constructing the view; parameter values remain lazy
until requested. QTable conversion uses Tollan's shared columnar utility.

Use `ensure_datatree(ds)` to start a working tree that shares the Dataset's
arrays. `BlockIdentify()(tree)` publishes its index and shared axes without
moving native variables or creating children. `BlockCreate()(tree)` consumes
that result, moves native variables to `raw` and installs the
recorded block index, axes and scopes. It returns the same tree and publishes
the block collection through `BlockCreate.get_context(tree).data`.
On a selected block, its context instead exposes that block's view.
Existing structures are reused, including reopened products without live
contexts; identification on these scopes completes without publishing a new
index. Retain the tree to accumulate results and keep the source file open
while reading borrowed arrays. On a processing node, `.toltec_kids.tree`
returns the node itself.

A block selected from the collection uses virtual raw/channel/probe slices.
`block.detach()` returns an independent block tree with local raw, channel and
probe leaves, complete metadata and existing analysis results. Recorded model
parameters remain accessible from the local raw headers. It leaves the
original acquisition untouched. The arrays remain lazy and share storage where
possible; detaching does not load them or close their source file. When raw
is omitted, detaching reads model column names to retain a lazy `model_params`
leaf, without loading parameter or detector values.

```python
with open_toltec_kids("toltec0_145609_000_0001_2025_10_31_09_26_59_tune.nc") as ds:
    tree = ensure_datatree(ds)
    BlockCreate()(tree)
    blocks = tree.toltec_kids.blocks
    assert blocks.tree is tree["blocks"]

    standalone = blocks[1].detach()
    # Equivalent when starting from a block node:
    standalone = tree["blocks/1"].toltec_kids.detach()
    standalone = SweepReduce()(standalone)
    standalone = D21Analysis()(standalone.kids.sweep, output=standalone)
    from tolteca_kids.dataprod_output import DataProdOutput
    DataProdOutput(path="products", namespace="standalone")(standalone)
    standalone_path = DataProdOutput.get_context(standalone).data
```

Detaching an acquisition copies its tree structure with all existing children;
it does not compact them. Detaching a collection keeps the collection as root,
with shared axes and local raw selections for its blocks. An analysis subtree
can likewise be detached without bringing its acquisition or block parent.
Use `include_raw=False` to omit raw leaves while retaining metadata, axes,
recorded model parameters and results.

`BlockCreate`, reduction steps and configured processing with `run_data()`
use this same block layout. The block Zarr exporter requires that
layout; `detach()` can copy any existing tree while retaining its results.

D21 accepts a sweep view and uses `output=` to select the result tree. PSD accepts
a timestream view.
Both use `parent_namespace=` to select an existing attributes-only output scope
and return the supplied tree. A selected block node may itself be the output tree. For TolTEC reduction steps, pass the selected block node directly.

`DataProdOutput` stores shared axes once and omits reconstructible block copies.
Sweep-only products omit raw samples while retaining recorded model parameters
in columnar leaves. Timestream products retain raw samples and derive block views
from them. A detached block publishes its own axis selections and analysis results.
The product writers save execution metadata; read it with `read_sweep_context`
or `read_timestream_context` after reopening.

```python
import xarray as xr

from tolteca_kids.product import open_sweep_product

with open_sweep_product(standalone_path) as block:
    I_subset = block.toltec_kids.sweep.I.isel(iqlen=slice(0, 8)).load()
    meta = block.toltec_kids.meta
    # The sweep scope has everything needed to recompute D21 independently.
    analyzed = D21Analysis()(block.kids.sweep, output=block, force=True)
    derivative = analyzed.toltec_kids.d21.d21
```

A constant-LO timestream has one probe row. For `nominal` and `timestream`
acquisitions, preparation checks the first and last 64 LO samples; short arrays
are read once. Matching endpoints imply a constant interior, while disagreement
triggers a full scan. This heuristic cannot detect excursions confined to the
interior. Sweep acquisitions use a full LO scan. Mixed-file boundary detection
remains a TODO, independent of this tree layout.

## Products and result views

Run the transform steps and publish directly from an open dataset:

```python
import xarray as xr
from tollan.accessor.xarray import ensure_datatree
from tollan.pipeline import SequentialPipeline
from tolteca_kids import BlockIdentify, BlockCreate, SweepReduce
from tolteca_kids.dataprod_output import DataProdOutput
from tolteca_kids.product import open_sweep_product

pipeline = SequentialPipeline(steps=[
    BlockIdentify(),
    BlockCreate(),
    SweepReduce(),
    DataProdOutput(
        enabled=True, path="products",
        run={"name": {"generator": "counter", "key": "transformed"}},
    ),
])
with xr.open_dataset(source) as ds:
    working = pipeline(ensure_datatree(ds))
    path = DataProdOutput.get_context(working).data

assert path is not None
with open_sweep_product(path) as transformed:
    acquisition_meta = transformed.toltec_kids.meta
    block_index = transformed.toltec_kids.blocks.meta.block_index_table
    sweep = transformed.toltec_kids.blocks[0].sweep
    transformed_I = sweep.I.load()
```

This publishes a run such as `0000-transformed` beneath the acquisition’s
`dp_raw_obs_proc` directory. Keep the input open until publication finishes; the
saved sweeps can then be read independently. Use `run_data` or `run_source` to
also record the input location and acquire the tag before computation. Calling
the output step directly acquires its tag only when that step starts.

Run the same configured sequence from a file:

```python
from tolteca_kids.product import open_sweep_product, read_sweep_context
from tolteca_datamodels.toltec.file import read_info_from_sources
from tolteca_kids.runner import run_source

table = read_info_from_sources([source])
inputs = table.toltec_file.query("roach == 0").toltec_file.source
rc_output = rc.with_config(overrides={
    "kids": {"output": {"enabled": True, "path": "products"}}
})
path = run_source(inputs.iloc[0], rc_output)
assert path is not None
with open_sweep_product(path) as tree:
    context = read_sweep_context(tree)
    result = SweepReduce.view(tree, block=context.block)
    config = context.config
    I = result.I.load()
```

A product directory contains:

```text
dp_raw_obs_proc/
  index.yaml
  0000-transformed/
    index.yaml
    array.zarr/
    artifacts/             retained artifacts from completed steps
```

| File | Contents |
| --- | --- |
| Namespace `namespace.yaml` | Stable namespace identity, counter, actors and invocation outcomes |
| Group `index.yaml` | Version-group metadata, membership, realized product records and current tags |
| Run `index.yaml` | Run UID and name, allocation/completion metadata, artifact sources, raw-product association and `provenance`: selected inputs, resolved workflow settings, runtime and software versions |
| `array.zarr` | Acquisition/block metadata, block index, shared axes, model parameters, transformed data, analyses and their configuration snapshots |
| `artifacts/<interface>/blocks/<block>/<step key>/<figure>.<format>` | Retained presentations, such as standalone sweep HTML and Plotly JSON |

Each realized `dp_raw_obs_proc` has UID `<uid_raw_obs>_proc/<run_uid>` in its
namespace. Its global version group has UID `<uid_raw_obs>_proc` and can contain
versions from different namespaces. Each DPDB has one `main` per group; publishers
can choose explicit labels such as `ql-main` without changing namespace ownership.
Their primary `dpa_raw_obs_proc_raw_obs` association targets `<uid_raw_obs>`.
A run directory carries relative payload locations and can be copied and
reopened independently.

### Reopen a selected run

The writer returns the pinned run directory. Passing the product directory
to a reader selects `main`. Resolve a tag explicitly:

```python
from tolteca_datamodels.data_prod_store import DataProdStore

index_path = DataProdStore.resolve_index(path.parent, tag="main")
with open_sweep_product(index_path, interface="toltec0") as tree:
    signal = tree.toltec_kids.blocks[0].sweep.I.load()
```

Readers also accept a `DataProdIndex` obtained from a database or metadata
service. No YAML file is read on that path. Use `location` to select among
registered copies and `cache=Path("product-cache")` to persist requested remote
Zarr objects across opens. Local sources are read directly. Cache files can be
removed when unused; they are not registered data products.

Separate Python processes can share a local HTTP cache. Retry a read after a
failed transfer or interrupted reader; incomplete downloads are not reused.
Previously cached selections remain readable while the origin is unavailable.
Concurrent cold reads may download the same object more than once.

The interface argument can be omitted for single-interface products. Sweep-only
publication omits raw I/Q samples; timestream publication retains them.
Each interface tree occupies `array.zarr/<interface>/`, including a product with
only one interface. The reader returns the selected interface's scientific tree.
Signal and uncertainty arrays retain a declared input unit; bare inputs remain
bare. Frequency coordinates carry `Hz`. These attributes work with the shared
`.u` accessor after reopening.

### Inspect versions and select a saved result

Read the group inventory without opening Zarr arrays:

```python
from tolteca_datamodels.data_prod import DataProdIndex

inventory = DataProdIndex.read(path.parent)
current_product = inventory.tags["main"].current  # DataProdRef or None
completed_run_uids = [product.run.uid for product in inventory.products]
```

Incomplete and failed requests are in `DataProdStore.read_namespace().runs`.
A batch reserves one run UID for all its acquisitions.
`current` is None until a completed product is selected. Resolving that tag raises
`KeyError`; it cannot yet supply scientific data.

Use `set_tag` to select a completed run without reprocessing. This example adds
the saved run to `review`; use `tag="main"` to promote it for default readers:

```python
store = DataProdStore("products", namespace="scratch")
pinned_index = store.set_tag(
    acquisition_meta.uid_raw_obs, "dp_raw_obs_proc", path.name, tag="review",
)
assert DataProdStore.resolve_index(path.parent, tag="review") == pinned_index
```

Promotion updates the current selection and its modifier attribution; the run's
files remain unchanged. Selection history is not recorded. An older
completed run can be selected the same way. A busy tag raises
`DataProdTagBusyError`; an unknown or incomplete run raises `KeyError`. Untagged
runs can be opened by their pinned path or selected with `set_tag` later.

### Recover processing settings and step records

The invocation records for the executed steps, including `BlockIdentify`,
`BlockCreate`, the reducer and enabled check/plot steps, are stored as JSON strings
in `tolteca_kids` attributes, including aliases when run.
They retain configuration, completion state and failures; the reduction record
also retains the selected block. `read_sweep_context(tree)` loads the reduction
metadata through `SweepReduce.load_context()`; its `data` remains None and
no live step context is attached. Read the saved arrays with
`SweepReduce.view(tree, block=context.block)`. The view's `sweep` property
is the frequency-offset coordinate.

On the live tree, `DataProdOutput.get_context(working)` records the output
configuration and completion state; its `data` is the written product path.
The run index retains the publication settings without serializing live resources.

Read the workflow settings and selected inputs without opening Zarr:

```python
from tolteca_kids.product import read_product_provenance

provenance = read_product_provenance(path)
selected_input = provenance.inputs["toltec0"]
config = selected_input.config
source = selected_input.source
publication_config = provenance.output
```

Each interface has its own resolved configuration, including defaults, overrides
and conditional recipe settings. Defaulted values are saved explicitly. If a
default changes from `1` to `2`, replaying a snapshot that contains `1` still uses
`1`. Whether the original recipe specified that value has no effect after
resolution. Read acquisition metadata through the opened
tree's `toltec_kids.meta` and actual result settings through its analysis views.

| Record | Use |
| --- | --- |
| `provenance.inputs[interface].config` | Replay the resolved workflow for that input, without re-evaluating recipe conditions |
| Saved step context | Inspect execution settings, completion and failures |
| Analysis view's configuration snapshot | Inspect the settings that produced that stored result, including reused results |
| `provenance.output` | Inspect the settings used to publish this run |

A workflow snapshot describes the invocation. Later independently invoked steps
or reused results can have different settings; inspect their saved records and
result snapshots when tracing those arrays. Publication does not reconstruct a
workflow from step records. Trees produced solely by individual steps have
`config=None` unless a known workflow is supplied to `write_kids_product`.

For a workflow-produced product, save one input's resolved settings as a recipe:

```python
import json
from tolteca_kids.core import Kids

assert config is not None
Path('replay.yaml').write_text(json.dumps(Kids.prepare_runtime_config_data(
    config.model_dump(mode='json', round_trip=True),
)))
```

Replay using the recorded input, with a new output run:

```console
tolteca_kids --config_paths replay.yaml --kids.output.run.uid null --kids.output.run.name null -- sweep.nc
```

Here `sweep.nc` must refer to the recorded input. Clearing both output run fields
creates a new UID and name; existing artifacts remain unchanged.
For several interfaces with different settings, save and apply each interface's
snapshot to its recorded input. Original recipe files need not remain available.

Reproducing numerical results also requires the same input contents and compatible
software. Input UIDs/locations and installed package versions are recorded; input
content checksums, environment locks and uncommitted code changes are not. Python
preprocessing performed before `run_data` is not captured by `KidsConfig`.

### Size chunks for partial reads

New products use at most 16 channels per chunk for reduced sweeps and retained
raw time arrays. Reduced sweeps retain the complete probe axis in each chunk,
including uncertainties and frequency coordinates. Raw time arrays use at most
1,024 samples per chunk. Small channel coordinates and block indexes remain
compact.

Use larger channel chunks when consumers usually read many channels together:

```yaml
kids:
  output:
    n_chans_per_chunk: 64
    n_samples_per_chunk: 1024
```

Both settings must be positive integers. They change storage granularity, not
values, dimensions or units. Publication records the settings in its output
configuration. Existing products keep their saved chunks; publish a new run to
use a different layout.

### Publication outcomes

The writer uses Zarr format 2, consolidates metadata and verifies a metadata
reopen before publishing completed artifacts. Starting a publication creates a
reservation in the namespace's `namespace.yaml`. Per-output publication requests
record status and failures while `tags[label].current` continues to select the
previous completed product. Successful publication
adds the artifact sources and advances `current` atomically. Existing run
artifacts remain available.

An interrupted run retains its reservation with no completion time. Its ID and
counter value cannot be reused. A crash during final publication can leave a run
directory that is not committed in the product inventory; readers following tags
continue to select the previous completed result. Re-run with a new ID.

A store without consolidated metadata remains readable through xarray's slower
fallback, which emits a performance warning. Readers reject missing indexes or
provenance and unsupported product formats.

Shared mounts must support cross-machine file locks and atomic rename. File locks
on independent copies do not coordinate publication: transfer into the destination
and publish there. Database registration is a separate, retryable operation;
publication does not provide a filesystem/database transaction or content checksum.
Database registration and Web product-index projection are separate consumers. They
must only publish completed products. Reduction itself does not require a
database or orchestration service.


## Python timestream analysis

A raw timestream view is available before computing spectra:

```python
import xarray as xr
from astropy import units as u
from tolteca_datamodels.toltec import open_toltec_kids
from tolteca_kids import TimestreamReduce, TimestreamView
from tolteca_kidsproc.analysis import PSDAnalysis

with open_toltec_kids("toltec0_145609_000_0002_2025_10_31_09_27_57.nc", chunks={}) as ds:
    selected = ds.toltec_kids.select_channels([10, 0, 5])
    acquisition = ensure_datatree(selected)
    BlockCreate()(acquisition)
    block = acquisition.toltec_kids.blocks[0]
    view = block.timestream
    signals = view.data_source             # I/Q on (chan, time), lazy
    seconds = view.time                    # seconds within this block
    samples = view.sample                  # acquisition sample IDs
    iq = view.S21                          # I + j Q; no model normalization

    reducer = TimestreamReduce(
        psd=PSDAnalysis(
            nperseg=256, stat_f_lims=(10 << u.Hz, 50 << u.Hz),
            compute_common=True, batch_size=32
        ),
    )
    tree = reducer(block.tree)
    view = tree.toltec_kids.timestream
    samples_I = view.I                     # original recorded samples
    psd_I = view.I_psd                      # spectrum of the same signal
    magnitude_psd = view.amplitude_psd
    common_I = view.psd.psd_common.I
    saved_settings = TimestreamReduce.get_context(tree).config
    assert tree.kids.psd.psd_analysis == reducer.config.psd
    from tolteca_kids.dataprod_output import DataProdOutput
    from tolteca_kids.product import open_timestream_product
    DataProdOutput(path="products", namespace="timestream-block")(tree)
    saved_path = DataProdOutput.get_context(tree).data

with open_timestream_product(saved_path) as tree:
    view = tree.toltec_kids.timestream
    one_channel = view.I.sel(chan=10)       # original readout ID
    spectrum = view.I_psd.sel(chan=10)
```

The TolTEC `TimestreamView` provides both signal and spectral access.
Use `.I`, `.Q`, `.S21`, `.amplitude`, `.phase`, `.time` and `.sample` for
signals, and `.I_psd`, `.Q_psd`, `.amplitude_psd`, `.phase_psd` and `.f_psd`
for spectra. There are no separate domain views. Its `.data_source` is one
virtual Dataset containing the available columns; it also supports the generic
`.kids.multi_timestream` signal view from `tolteca_kidsproc`. The tree owns raw
samples once on disk. The working block stores prepared samples at `timestream`
and spectral results at `tolteca_kidsproc.analysis.psd/{psd,psd_common}`.
`view.psd` and `block.kids.psd` expose the same results, including common spectra
when `compute_common=True`. Export omits reconstructible time-sample projections;
reopened TolTEC views rebuild them lazily from the native raw leaf.
In a spectra-only export made with `include_raw=False`, `I`/`Q` and `time`
return `None`. Metadata, axes, recorded model parameters and spectra
remain accessible.

Magnitude uses floating-point I/Q arithmetic, avoiding integer overflow.
`phase` is wrapped raw I/Q phase; `phase_psd` uses phase unwrapped along time.
Neither is model-derotated phase. Spectra of supplied numerical `r`/`x` columns
are supported, but the reducer does not derive them from raw I/Q or recorded
model parameters. No hard-coded sweep/timestream normalization is applied.

To analyse a short time interval, select before reduction:

```python
with open_timestream_product(saved_path) as tree:
    segment = tree.toltec_kids.timestream.data_source.isel(time=slice(100, 700))
    segment_result = PSDAnalysis(nperseg=128)(segment.kids.timestream)
    segment_psd = segment_result.kids.psd.psd.I.compute()
```

Standalone `PSDAnalysis` retains the supplied view's Dataset at `_input` and
installs spectra in its analysis namespace. Use `.kids.psd` to access those
results. Native acquisition and block reduction use the block tree. Supply complete,
uniformly sampled data with `f_smp` matching the selected samples. Check continuity
and handle missing samples in the surrounding workflow before reduction. Update
the rate after decimation. Welch tasks process channel batches, each with
the selected time span. The default `psd.batch_size` is 32 channels. Large files
should be sliced to the intended analysis interval first.

Sampling rate comes from the acquisition header. `time` is elapsed seconds,
not telescope UTC. Raw `Data.Toltec.Ts` and `Data.Toltec.RecvTime` remain available
through `block.raw`; clock synchronization and packet-gap handling require an
explicit analysis. PSD units follow the input units: squared units per Hz for
`density`, squared units for `spectrum`. Headers with no declared I/Q unit remain
unitless; no ADU scaling is invented. Frequency-band median and MAD statistics
use all bins when no band is configured. Set `psd=None` to skip spectral analysis.

## Reusing PSD analysis

`PSDAnalysis` provides a high-level workflow for KIDs timestream views and
low-level calculations for arbitrary numerical time series.

| Call | Input | Result |
| --- | --- | --- |
| `analysis(timestream_view)` | `TimestreamView` or `MultiTimestreamView` | DataTree with I/Q spectra, statistics, and configuration |
| `analysis.make_psd(data, f_smp=..., dim=...)` | DataArray or Dataset with an explicit sampling-frequency quantity | Dataset of spectra under the original series names |
| `tollan.utils.xarray.reduce_median_mad(data, dim=...)` | Selected numerical values | Pair of median and scaled-MAD Datasets |

Start with a timestream view to compute I/Q spectra and frequency-band statistics:

```python
import numpy as np
import xarray as xr
from astropy import units as u
from tolteca_kidsproc.accessors import make_kids_dataset
from tolteca_kidsproc.analysis import PSDAnalysis, PSDView
from tollan.utils.xarray import reduce_median_mad

time = np.arange(1024) / 128.0
signal = np.exp(2j * np.pi * 8 * time)
source = make_kids_dataset(
    signal.real, signal.imag, time=time << u.s, f_smp=128 << u.Hz,
)
source["I"] = source.I.u.set(u.V)
source["Q"] = source.Q.u.set(u.V)
timestream = source.kids.timestream
analysis = PSDAnalysis(nperseg=256, stat_f_lims=(2 << u.Hz, 20 << u.Hz))
result = analysis(timestream)
view = result.kids.psd
power = view.psd.I.compute()
median = view.psd_median.I
saved_config = view.psd_analysis
assert view.psd.I.u.unit == u.V**2 / u.Hz
assert result.kids.timestream.data_source.identical(source)
```

The view supplies mapped I, Q, and time arrays; other variables in its Dataset
are not analysed. Its `.f_smp` is a frequency quantity read from the required
sampling-rate attribute (stored in Hz). `make_kids_dataset` prepares both time
and rate:

| Supplied arguments | Construction |
| --- | --- |
| `f_smp` only | Construct elapsed time starting at zero with spacing `1 / f_smp`. |
| `time` only | Verify regular spacing, then derive and store the rate. |
| Both | Verify that time is regularly sampled at the supplied rate. |

For data prepared elsewhere, check data integrity and handle missing samples
before analysis. PSD assumes complete, uniformly sampled input; it does not
evaluate timestamps or infer a rate.
The sample dimension comes from the view's shared one-dimensional time coordinate.

Use frequency quantities for `stat_f_lims`; configuration files accept
quantity strings such as `["2 Hz", "20 Hz"]`. `None` uses every frequency bin.

For arbitrary signals, call `make_psd` with a frequency quantity and the sample
dimension. It does not infer sampling from attributes or timestamps. Signal
units come from `.u.unit`; set them with `.u.set(...)` before calling. Signals
without a declared unit are treated as dimensionless, producing `1 / Hz` under
`density` scaling and dimensionless results under `spectrum` scaling:

```python
temperature = xr.DataArray(
    np.sin(2 * np.pi * 8 * time), dims="sample", name="temperature",
).u.set(u.K)
individual = analysis.make_psd(temperature, f_smp=128 << u.Hz, dim="sample")
band = individual.sel(f_psd=slice(2, 20))  # f_psd is in Hz
band_median, band_mad = reduce_median_mad(band, dim="f_psd")
assert list(individual.data_vars) == ["temperature"]
assert band_median.temperature.ndim == 0
assert individual.temperature.u.unit == u.K**2 / u.Hz
```

Dataset variables carrying the supplied dimension are processed independently,
retaining their names and non-time dimensions. Different variables can have
different units. An unnamed DataArray produces a variable named `data`.
Complex signals use a shared two-sided frequency grid in NumPy FFT order;
real signals use a one-sided grid by default. Spectral computations remain lazy,
with a complete time series in each task.

For multiple channels, request a common spectrum with `compute_common=True`.
The input view's mapper supplies the channel dimension through `chan_dim`:

```python
signals = np.exp(2j * np.pi * np.array([4, 8, 12])[:, None] * time)
multi = make_kids_dataset(
    signals.real, signals.imag, time=time << u.s, f_smp=128 << u.Hz,
)
timestream_view = multi.kids.multi_timestream
multi_analysis = PSDAnalysis(nperseg=256, compute_common=True)
multi_view = multi_analysis(timestream_view).kids.psd
assert multi_view.psd.I.dims == ("chan", "f_psd")
assert multi_view.psd_median.I.dims == ("chan",)
assert multi_view.psd_common.I.dims == ("f_psd",)

chan_dim = timestream_view.mapper.get_name(timestream_view.mapper.schema.chan_dim)
common, common_mad = reduce_median_mad(multi_view.psd, dim=chan_dim)
xr.testing.assert_identical(common, multi_view.psd_common)
```

Tollan’s `reduce_median_mad` reduces any supplied dimension. Select or normalize spectra
first when the comparison needs different channel selections or relative scales.
The high-level analysis selects the frequency band for per-channel statistics
and the mapped channel dimension for the common spectrum. Both scatter estimates
use MAD multiplied by 1.4826 and propagate NaNs. These are scatter measures,
not Welch uncertainties. A common median spectrum is not the PSD of a combined
time signal. Aggregation combines the selected dimension in each task.

The high-level result retains the view's source at `_input`. Individual spectra
and band statistics occupy `tolteca_kidsproc.analysis.psd/psd`; an optional
`psd_common` leaf contains the common spectra. The enclosing namespace retains
the producing configuration. `.kids.psd` groups results by their semantic signal
names `I` and `Q` and exposes `.psd`, `.psd_median`, `.psd_mad_std`,
`.psd_common`, `.psd_common_mad_std`, `.f_psd`, `.f_smp`, `.n_samples`, `.nperseg`,
and `.psd_analysis`.

D21 and PSD use the same input/output pattern:

| Analysis | Input | Output scope |
| --- | --- | --- |
| D21 | `SweepView` or `MultiSweepView` | New scope retaining the view's Dataset at `_input`, or an existing tree supplied with `output=` |
| PSD | `TimestreamView` or `MultiTimestreamView` | New scope retaining the view's Dataset at `_input`, or an existing tree supplied with `output=` |

To install results in a block, select its view and pass the block as `output`:

```python
tree = xr.DataTree.from_dict({"blocks/0/timestream": source})
block = tree["blocks/0"]
assert analysis(block.kids.timestream, output=block) is block
view = block.kids.psd
assert view.data_source is block["tolteca_kidsproc.analysis.psd"]
```

`parent_namespace` selects a destination within an explicit output tree:

```python
assert analysis(
    block.kids.timestream, output=tree, parent_namespace="blocks/0"
) is tree
```

Existing output scopes must have no variables, coordinates, or dimensions,
including inherited ones. Their attributes and unrelated children are retained.
Signal views select the input: `block.kids.timestream` locates its mapped payload,
while `TimestreamView(block["timestream"])` uses that node's payload directly.
Calling `.kids` alone does not require signal fields.

Accessing `.kids.d21` or `.kids.psd` constructs a fresh analysis view. A previously
obtained view retains its results and producing configuration after replacement;
a new view sees the replacement. Force recomputation when input values change:

```python
saved = block.kids.psd
assert analysis(block.kids.timestream, output=block) is block
assert block.kids.psd.data_source is saved.data_source
assert analysis(block.kids.timestream, output=block, force=True) is block
assert block.kids.psd.data_source is not saved.data_source
view = block.kids.psd
```

Use `from_namespace` to read a detached analysis namespace. Copy or export the
enclosing scope to retain the input as well:

```python
detached = view.data_source.copy(deep=False)
power = PSDView.from_namespace(detached).psd.I
```

Reopened Zarr results use the same view:

```python
from pathlib import Path
from tempfile import TemporaryDirectory

with TemporaryDirectory() as directory:
    target = Path(directory) / "spectra.zarr"
    detached.to_zarr(target, zarr_format=2)
    with xr.open_datatree(target, engine="zarr") as reopened:
        restored = PSDView.from_namespace(reopened)
        assert restored.psd_analysis == analysis
        xr.testing.assert_equal(restored.psd, view.psd)
```

The TolTEC reducer prepares its signals, including magnitude, unwrapped phase,
and supplied r/x, for the shared PSD workflow. All selected observables receive
the configured band statistics and optional common spectra. Callers access
samples and spectra through `.toltec_kids.timestream`, and full spectral results
through its `.psd` property. For numerical inputs on a custom sample dimension,
use `PSDAnalysis.make_psd(data, f_smp=rate, dim="tick")`.

## Configured timestream products

```console
tolteca_kids --kids.timestream_reduce.psd.nperseg 256 --kids.output.enabled true --kids.output.path products -- timestream.nc
```

Equivalent YAML:

```yaml
kids:
  timestream_reduce:
    block: -1
    force: false
    psd:
      nperseg: 256
      stat_f_lims: ["10 Hz", "50 Hz"]
      batch_size: 32
  output:
    enabled: true
    path: products
    namespace: scratch
    tag: main
    run: {}
```

The input kind selects `TimestreamReduce` followed by the shared
`DataProdOutput` step. The product contains `<run>/array.zarr` in the block layout
and `index.yaml` listing its arrays and provenance, with source metadata, settings and
software versions. Raw samples are retained once so the product provides both
time and PSD domains. Existing directories are never overwritten.

```python
from pathlib import Path
from tolteca_kids.product import open_timestream_product, read_timestream_context
from tolteca_kids.timestream_reduce import TimestreamReduce

with open_timestream_product(Path("products/scratch/145609-0-2/dp_raw_obs_proc")) as tree:
    context = read_timestream_context(tree)
    assert context.completed
    assert context.data is None
    view = tree.toltec_kids.timestream
    samples_I = view.I
    spectrum_I = view.I_psd
```

The product writer saves reduction contexts through `TimestreamReduce.dump_context()`.
Use `read_timestream_context(tree, alias=...)` for an exported aliased invocation,
then `TimestreamReduce.view(tree, block=context.block, alias=...)` for its arrays. The saved
fields include the step configuration, selected block, completion state, and
failures; result arrays are accessed through the timestream view. The analysis
configuration remains available through `view.psd.psd_analysis` when spectra
are present.

The reader validates completion and configuration consistency and closes the
opened tree if validation fails. Caller-owned inputs passed through `run_data`
remain open; `run_source` owns and closes the file it opens. `--dry_run` remains
a headers-only no-op even when product output is enabled.

## Acquisition feedback

Enable `kids.quicklook.enabled` alongside `kids.sweep_check.enabled` to publish
a per-interface PNG summary and a portable JSON feedback document. The runner
prepares channel level, RMS and D21/baseline panels from the existing checks.
`kids.quicklook.block: -1` selects the last sweep block by default; missing
checks are shown explicitly. Set a block number to select that block on each
interface. All reduced blocks remain in the saved scientific tree.

`kids.quicklook.summary` configures the `SweepCheckSummaryViz`, including
`row_height` and inherited Plotly export options. PNG export requires Kaleido
and Chrome/Chromium. `BROWSER_PATH` can select the executable. Prepared feedback
is published by DataProdOutput in the same immutable run as the Zarr arrays.
An export failure leaves the previous tag intact.

The standalone `KidsQuicklook` artifact specification accepts acquisition trees
keyed by interface. Keep readers open until preparation returns; its loaded
result can then be exported with `spec.write(result, path=directory)`. The
`tolteca_web quicklook` service serves registered feedback through HTTP.


To publish from an already discovered collection of file locations:

```python
from tolteca_kids.runner import publish_quicklook

# runtime has kids.output.enabled and kids.quicklook.enabled set to True.
indexes = publish_quicklook(sources, runtime)
```

The returned paths identify the namespace descriptor, pinned products and current
version groups. Input files are closed before return. Enable
`kids.output.registration` to register them during publication, or ingest the
saved indexes separately. Use `run_sources(sources, runtime)` to inspect
working trees during processing, or `run_inputs(selected, runtime)` when you
already own an opened, header-populated input table.


### Register output in DPDB

Set `kids.output.registration: true` and configure `data_prod.index.db_bind` to
register published products. The raw acquisition records must already be in DPDB.
The runner updates the final run status after the batch finishes. A registration
failure preserves the pinned publication; retry its saved indexes with
`tolteca_db index ingest`, without reducing the files again. Leave registration
False (the default) for processing without a DB connection.
