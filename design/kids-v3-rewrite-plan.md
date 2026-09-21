# KIDs v3 rewrite: scientific contract and proposed architecture

Status: reducer-only local slice implemented, 2026-09-15. The established
SweepReducer remains the numerical object. Pipeline steps enrich a stable live
tree and retain the Boolean internal run contract. See
[current slice status](kids-reducer-slice-status.md) for executable evidence,
DB/Web probe boundaries, and unresolved friction. Later scientific stages and
production DB/Web integration remain work in progress.

Scope: the entire owned stack, including Tollan, Tolteca's component packages,
Tolteca DB, and Tolteca Web. Existing APIs are changeable. The constraint is
preserving v2 scientific output values, not preserving its classes or file layout.

This reassessment supersedes the **recommendations**, not the recorded
observations, in the earlier
[pipeline/provenance audit](/Users/ma/Codes/toltec/dev_toltec/python/tollan/design/pipeline-provenance-audit-2026-09.md).
In particular, an identity-keyed weak context store and a generic execution-record
framework are not prerequisites for this rewrite.

## 1. Recommendation

Build the pipeline around the **existing SweepReducer contract**: a configured
processor returns a DataTree containing data-backed results, accessed through
schema → mapper → accessor/view. Extend that pattern rather than replacing the
reducer with a Dataset-returning API or requiring a new product container:

1. One authoritative raw-reading and sweep-reduction path, exposed through the
   existing schema → mapper → accessor/view architecture.
2. TolTEC pipeline stages return an enriched DataTree, preserving source and
   previous result nodes. Generic analyses such as D21Analysis retain independent
   Python APIs with natural Dataset/DataArray results. Stages select the relevant
   views, invoke those analyses, and place the results. Detection explicitly uses
   the checked-sweep view.
3. Keep scientific results in native result nodes. Represent ordinary tabular
   results as column variables where appropriate, with Polars/Parquet projections
   for querying. Evaluate genuinely awkward table structures when encountered;
   a mixed KidsProducts container is not a prerequisite.
4. Keep typed configuration and small serializable processing metadata. Make
   runtime/debug context an optional, explicitly owned execution concern.
5. Export a versioned product containing Zarr arrays, Parquet tables, and a small
   manifest. Domain views work after reopening it, without restoring Python
   algorithm objects or rerunning reduction.
6. Let DB/orchestration register and locate products. Let Web select and visualize
   their contents. Shared scientific interpretation belongs below both packages.
7. Deliver a complete **raw sweep → ingestion → SweepReducer → persisted reduced
   view → DB discovery → Web visualization** slice first. Establish reducer-level
   v2 comparisons inside this slice before changing arithmetic. Extend the
   comparison harness as subsequent scientific stages are ported.

This slice must demonstrate the architecture through actual consumers, not stop
at a working reducer in memory. SweepCheck, KidsFind, and generic Step changes
follow after the reduced sweep can be ingested, reopened, discovered, and plotted.
The public reduction CLI is also part of this slice. Orchestrators invoke that
CLI, preserving the ability to rerun outside toltecpipe/Dagster; its implementation
calls shared library code rather than owning a separate reduction/config path.

## 2. What was inspected, and the strength of the evidence

Current source snapshots: `tollan 81be60e`, `tolteca 6ef2c16`,
`tolteca_db 4ea8804`, `tolteca_web d51c56c`. Legacy source was inspected through
the checkout used by the `tolteca_v2` pyenv. These working checkouts, rather than
April status tables, determine what currently exists.

Inputs to this assessment include:

- The April Tolteca architecture and implementation records, archived DataTree
  accessor design, and relevant Tollan, DB, and Web architecture records.
- The complete eight-page
  [SPIE paper](/Users/ma/Codes/toltec/dev_toltec/python/refs/zma_spie2026_v2.pdf),
  including its architecture, product graph, ingestion/reduction flow, and web
  consumption discussion. Sections 2–6 support the package boundaries and
  array/table products, but are not proof of implementation completeness.
- Legacy reader, pipeline, SweepCheck, KidsFind, peak/matching result types,
  DataProdOutput, Tlaloc output, tone helpers, dispatcher, and recipe overrides.
- Current raw/reduced accessors, reducers, D21, KIDs pipeline and plotting, Web
  data service/catalog, DB metadata/ORM/hashing, and Dagster asset definitions.
- Current official xarray, Zarr, Astropy, Arrow, Dask, and Pydantic documentation.

### Verified here

Source-level differences below were rechecked. The executable
[layout probe](/Users/ma/Codes/toltec/dev_toltec/python/tolteca/design/probes/kids_result_layout.py)
passed eight checks in the development environment:

- Conflicting parent/child channel lengths are rejected by DataTree.
- A coordinate-free root permits sibling datasets with independent dimensions.
- Replacing a computed variable leaves its input unchanged.
- Shallow copying still shares unmodified array buffers.
- Exact coordinate alignment rejects mismatched channel IDs.
- Building the tested lazy result tree executes no Dask tasks.
- Zarr v3 round-trip preserves the tested sibling shapes and integer flags,
  including zero-valued flags.
- Parquet round-trip preserves tested unsigned IDs, null versus NaN, and Arrow
  field unit metadata. Polars does not thereby acquire a unit-aware column API.

Environment: xarray 2026.7.0, Zarr 3.3.0, Dask 2026.8.0, Astropy 8.0.1,
Polars 1.44.1, PyArrow 25.0.1, Pydantic 2.13.5. Probe products were temporary.
An additional serializer check confirmed signed-infinite SweepCheck defaults and
Pydantic's supported string encoding of `Infinity`, `-Infinity`, and `NaN`.

### Not yet established

Earlier investigation reported exact reducer agreement for selected target/tune
I/Q, uncertainties, and sweep offsets. That is useful preliminary evidence, but
it is **not full-pipeline parity**, nor a durable golden-test suite. It was not
rerun as part of this layout probe. No claim is made here that the current v3
pipeline reproduces the final v2 tables or that the proposed storage layout has
been benchmarked over the production network.

## 3. Concrete gaps that affect the design

| Finding | Evidence | Required consequence |
| --- | --- | --- |
| Despiking changes downstream science in v2 | [v2 SweepCheck](/Users/ma/Codes/toltec/toltec_astro/tolteca/src/tolteca_kids/sweep_check.py:302) mutates `swp.S21` and resets derived caches. [v3 SweepCheck](/Users/ma/Codes/toltec/dev_toltec/python/tolteca/src/tolteca_kids/sweep_check.py:455) changes a local array; [KidsFind](/Users/ma/Codes/toltec/dev_toltec/python/tolteca/src/tolteca_kids/kids_find.py:264) extracts original I/Q again. | Return and explicitly consume the checked signal. Preserve the relevant post-despike derived calculations. |
| The uncertainty formula changed | [v2 S21Mixin](/Users/ma/Codes/toltec/toltec_astro/tolteca/src/tolteca_kidsproc/kidsdata/sweep.py:270) uses the magnitude of complex I/Q uncertainty. [v3 extraction](/Users/ma/Codes/toltec/dev_toltec/python/tolteca/src/tolteca_kids/sweep_check.py:327) uses radial error propagation. | Restore numerical semantics through the canonical view/kernel contract; a scientifically plausible replacement is still a parity failure. |
| Zero amplitude behavior changed | The same v3 helper replaces zero amplitude with a tiny positive value before logarithms; v2 directly applies the logarithm. | Compare finite values, infinities, NaNs, masks, and downstream branch outcomes. Do not silently regularize input. |
| Two reduction paths disagree in meaning | [SweepReducer](/Users/ma/Codes/toltec/dev_toltec/python/tolteca/src/tolteca_datamodels/toltec/kids/sweep.py:432) handles blocks and standard deviations. [Zarr pipeline builder](/Users/ma/Codes/toltec/dev_toltec/python/tolteca/src/tolteca_kids/pipeline.py:187) performs separate accumulation, omits uncertainties, and gives `f_lo` a different meaning. | Route raw formats through one semantic reader and one reducer. A format adapter must not also implement a different scientific reduction. |
| Axis roles are inferred positionally | [Raw metadata](/Users/ma/Codes/toltec/dev_toltec/python/tolteca/src/tolteca_datamodels/toltec/kids/core.py:310) treats the first I dimension as channel and last as time. The raw fixture layout is time × channel. | Resolve dimension roles from the input schema and explicitly transpose at the numerical boundary. |
| Real channel metadata is replaced by defaults | [KidsFind](/Users/ma/Codes/toltec/dev_toltec/python/tolteca/src/tolteca_kids/kids_find.py:276) synthesizes tone masks/amplitudes of one, with frequency fallbacks. | Carry actual block-selected tone information from the source. Missing required science metadata must not become invented data. |
| Noise/power functionality is missing from the current SweepCheck port | [v2 tone-power calculation](/Users/ma/Codes/toltec/toltec_astro/tolteca/src/tolteca_kids/sweep_check.py:359) runs during the check; [optional PSD](/Users/ma/Codes/toltec/toltec_astro/tolteca/src/tolteca_kids/sweep_check.py:607) uses unreduced I/Q and sample rate. The current v3 class lacks those result fields and the PSD config options. | Preserve these capabilities as explicit analysis branches with the actual required inputs, rather than limiting the product to the existing v3 field inventory. |
| Tree rebuilding can discard results | [SweepReducer return](/Users/ma/Codes/toltec/dev_toltec/python/tolteca/src/tolteca_datamodels/toltec/kids/sweep.py:327) constructs a root and just its reduced child. | Define whether a stage returns a new result or enriches a collection. Collection assembly must preserve unrelated results deliberately. |
| Durable output is much smaller than usable output | [Pipeline export](/Users/ma/Codes/toltec/dev_toltec/python/tolteca/src/tolteca_kids/pipeline.py:449) persists a few flags/detection fields; plotting consumes richer peak, match, and signal data. | Define output by the scientific and consumer contracts, not by whichever fields are easiest to serialize. |
| Web reconstructs missing outputs with a new calculation | [get_kids_diag_data](/Users/ma/Codes/toltec/dev_toltec/python/tolteca_web/src/tolteca_web/obs/data_service.py:495) runs SweepCheck/KidsFind with default configuration. | Plot saved results with their effective configuration. Missing diagnostics should be reported as unavailable. |
| Product reuse and replacement are unsafe | Pipeline skips on group existence and deletes existing output groups before rewriting. | Distinguish recipe identity, execution attempt, output identity, and publication completeness. Never overwrite a source as the default reduction operation. |
| Hash behavior is not stable across all environments/configs | Pipeline hashes `model_dump_json()` with non-finite limits; [DB hashing](/Users/ma/Codes/toltec/dev_toltec/python/tolteca_db/src/tolteca_db/utils/hashing.py:18) changes algorithm according to optional `blake3` availability. | Adopt one explicit versioned digest contract for new processing identities, with tested non-finite serialization. |
| Current DB is less complete than some refs describe | [Dagster assets](/Users/ma/Codes/toltec/dev_toltec/python/tolteca_db/src/tolteca_db/dagster/assets.py:29) expose `process_quartet` and `association_groups`. The described `export.ZarrExporter` / `reduce_obs` implementation is absent from the inspected source tree. | Treat reduction/product publication integration as work to implement, not an available service. |
| Remote lookup still depends on local files | [get_obs_data](/Users/ma/Codes/toltec/dev_toltec/python/tolteca_web/src/tolteca_web/obs/data_service.py:336) tests local existence before constructing its HTTP URL. | Resolve a product location independently of a local mirror; preserve the existing catalog/service separation. |

These findings do not invalidate schema/mapper/accessor. They show where the
prototype bypassed it or failed to carry enough information between layers.

## 4. The v2 scientific contract

### Actual execution and configuration

[Kids.pipeline](/Users/ma/Codes/toltec/toltec_astro/tolteca/src/tolteca_kids/core.py:89)
orders SweepCheck, SweepCheckPlot, KidsFind, KidsFindPlot, TlalocOutput, and
DataProdOutput. Plotting and writing are separate responsibilities even though
they are expressed as steps.

The [dispatcher](/Users/ma/Codes/toltec/toltec_astro/taco_recipes/scripts/kids/dispatch_tolteca_kids.py:11)
loads through `NcFileIO`, resolves configuration with `roach` and `file_suffix`,
then executes. The base YAML disables plots and Tlaloc, but the dispatcher's CLI
setup enables Tlaloc. Conditional YAML changes thresholds for roach ≥ 7 and
changes fixed-channel tone allocation for tune versus other sweeps. Consequently
neither library defaults nor the base YAML alone define the production baseline.

Record the effective configuration after all conditionals and CLI overrides,
the selected input block, input identities, and software versions. Fix a seed
or use a specified phase input for reproducible Tlaloc comparisons. Its legacy
generator uses `np.random.default_rng(seed)`; seeding NumPy's global RNG alone
does not control this generator.

Also verify that every effective v2 science setting has an explicit v3 counterpart
or an explained removal. Passing an old configuration into a Pydantic model that
ignores unknown fields must not silently drop features such as `noise_psd`.

### Outputs to inventory and compare

[DataProdOutput](/Users/ma/Codes/toltec/toltec_astro/tolteca/src/tolteca_kids/dataprod_output.py:76)
exports two specific projections: `detected_matched` and `chan_matched`, with:

`idx_det`, `bitmask_det`, `f_det`, `Qr`, `dist`, `d_phi`, `idx_chan`, `f_chan`,
`amp_tone`, `snr_d21`, `snr_s21`, `height_d21`, `height_db_s21`.

Both carry observation identifiers, roach identity, and raw observation/file UIDs.
The optional context pickle and selected-object pickles are debug/local-inspection
outputs. They do not mean every runtime object is a public product.

[TlalocOutput](/Users/ma/Codes/toltec/toltec_astro/tolteca/src/tolteca_kids/tlaloc_output.py:228)
also defines scientifically consequential selection, trimming, placeholder reuse,
amplitude assignment, phases, and ordering. Generate and compare these products
in temporary directories; do not invoke any live instrument deployment.

The contract should include four layers:

| Layer | What must agree |
| --- | --- |
| Input interpretation | Selected block, channel identity/order, LO/tone/sweep frequencies, tone masks/amplitudes, units, raw flags. |
| Numerical checkpoints | Reduced I/Q and uncertainties; checked S21; D21, baseline, peak locations; matching indices. These localize failures. |
| Final products | Both legacy table projections, all flags/masks/units, and tone-generation products for enabled supported profiles. |
| Diagnostic product values | The numerical inputs of supported diagnostic plots. Plotly serialization, object identities, and image pixels need not reproduce v2. |

Float equality should initially be exact, with NaN locations and signed infinities
checked explicitly. Also compare masks, dtypes where semantically significant,
row identities, and ordering where meaningful. Different record organization may
require an explicit ID-based correspondence; indiscriminately sorting tables
would hide channel-order or tie-breaking changes.

If dependency/architecture changes make exact values diverge, report the first
divergent checkpoint and its cause. Do not introduce tolerances or accept merely
equal detector counts without a separate scientific decision. A fixed seed alone
also does not prove equal output across changed RNG implementations or draw order.

### Fixture coverage

Use the read-only fixtures through
`toltec_astro_dev/run/data_lmt/toltec/tcs/toltec0/`:

- `145605 ... vnasweep.nc`: broad sweep coverage.
- `145606 ... targsweep.nc`: target sweep coverage.
- `145609 ... tune.nc`: multi-block behavior; legacy defaults select the last block.

Network 0 does not establish the behavior of the roach ≥ 7 conditional profiles.
Add explicit tests of configuration selection and synthetic boundary cases now;
obtain additional real reference observations before claiming all-network parity.
Also cover despike on/off, no/one/many detections, disabled/placeholder tones,
matching ties and duplicates, empty tables, zero amplitude, missing required
metadata, and non-default block selection. Cases that v2 rejects must be recorded
as such rather than silently advertised as new successful parity cases.
Include the optional noise-PSD profile and tone-power diagnostics. PSD must be
compared using the unreduced samples, with the same Welch parameters/sample rate;
averaged sweep points are not an equivalent substitute.

## 5. Native results and the role of StepData

### The established template

[SweepReducer](/Users/ma/Codes/toltec/dev_toltec/python/tolteca/src/tolteca_datamodels/toltec/kids/sweep.py:192)
is a frozen Pydantic-configured callable. It accepts Dataset/DataTree and returns
a DataTree with raw data at the root and a namespaced reduced child. It records
configuration on that child. This return contract is retained.

There are two complementary examples, not one mandatory signature for all classes.
[D21Analysis](/Users/ma/Codes/toltec/dev_toltec/python/tolteca/src/tolteca_kidsproc/analysis/d21.py:318)
is a **generic** analysis of sweep data. `tolteca_kidsproc` owns reusable sweep
handling/analysis without TolTEC observation, instrument-file, DB, or dispatcher
knowledge. `tolteca_kids` connects those capabilities to TolTEC data and its
processing workflow; `tolteca_datamodels` supplies instrument schemas/views and
the existing SweepReducer.

D21Analysis is constructed with ordinary Python keyword arguments, for example
`D21Analysis(method="gradient", smooth=5)`. It is itself a frozen validated
parameter object, not a consumer of RuntimeContext. Its `make_matched(view)` and
`make_unified(view, ...)` methods return DataArrays; its current `__call__(ds)`
adds results to a Dataset and returns it. D21Schema/Mapper/View provide data-backed
access, and the analysis serializes its parameters into result attrs. This is
the independence we should retain for generic algorithms.

Useful details to carry forward:

- Numerical defaults and parameter validation live on the analysis class once.
  Step/domain configuration can contain that class as a typed field rather than
  repeating all its parameters. SweepCheckConfig already does this for D21Analysis.
- A directly constructed analysis can run without CLI, runtime configuration,
  StepContext, Dagster, DB, or TolTEC-specific source files.
- Data-backed views recover the recorded analysis settings after reopening.
- Reuse matched D21 when building unified D21, as the current high-level method
  already does; do not recompute shared work merely to cross a wrapper boundary.

Do not copy every current implementation detail blindly. D21Analysis currently
mutates its supplied Dataset, while its view can resolve a tree node. Run that
method on an explicitly owned result Dataset or use the DataArray-returning
methods when assembling a stage result. Its config-only skip check assumes stable
input; source revision and complete outputs still matter. Generic analyses do
not all need to return a complete TolTEC working tree.

| Existing component | Role | Corresponding later analysis component |
| --- | --- | --- |
| SweepReducer | Configured computation and result-node creation | SweepCheck / KidsFind configured processor |
| ToltecSweepSchema | Semantic fields of reduced data | SweepCheckSchema / KidsFindSchema |
| ToltecSweepMapper | Resolve stored fields | SweepCheckMapper / KidsFindMapper |
| ReducedSweepView | Typed properties; locate the result within a tree or use a standalone Dataset | SweepCheckView / KidsFindView |
| `dt.toltec_kids.sweep` | Convenient access to the view | Accessor conveniences as needed, without making algorithms depend on UI syntax |

The schema stays flat. A DataTree group is a data namespace; it does not require
nested Mapping/Schema classes. Output nodes replace the scientific-field
inventories in SweepCheckData/KidsFindData, and views supply their typed interface
without holding another copy of the data.

### Proposed composition

```python
# First two lines use the established reducer/view API.
dt = SweepReducer()(raw)
sweep = dt.toltec_kids.sweep

# Proposed extensions using the same processor/result/view pattern.
dt = SweepCheck(...)(dt)
dt = KidsFind(...)(dt)
check = SweepCheckView(dt)
found = KidsFindView(dt)
```

Each processor resolves its inputs through views and adds its own result nodes
to the returned tree. It preserves existing nodes and avoids mutating shared
input buffers. A small KidsPipeline composes these calls. No replacement generic
Step framework or KidsProducts wrapper is required to establish this behavior.

SweepCheck produces a checked-sweep node using the reduced-sweep schema, plus
quality/D21 results. `SweepCheckView.sweep` returns a ReducedSweepView over that
checked node. KidsFind explicitly reads `SweepCheckView(dt).sweep` and its
associated quality/D21 results. It does not silently select the original
`dt.toltec_kids.sweep`, nor rely on a runtime context's cached arrays.

The tree also retains access to raw data when needed: noise PSD reads unreduced
samples/sample rate through a raw view; tone-power analysis reads checked I/Q and
the required readout metadata/ADC snapshot through the appropriate views. These
analyses can follow the same result-node pattern, while their numerical functions
still receive explicit arrays/parameters. A common working tree does not imply
that every analysis has the same scientific input.

### Tables and output formats

Start with native result datasets. An ordinary detection table can use one
detection dimension with typed column variables; membership relations can use
explicit row IDs instead of Python object columns. Polars and Parquet are useful
query/export projections of those values. Do not maintain two independently
mutable authoritative copies.

Masked/ragged/rich table representations need focused tests when detection and
matching are ported. A separate native table component remains an option if a
concrete result proves awkward in xarray, not an assumed new foundation for the
pipeline. Quantities/QTables may remain local numerical tools; conversions must
preserve the explicit unit/mask contract.

Multi-block selection is explicit in the view/product contract, with the legacy
last-block default verified in the first slice. Retain the existing block
dimension where grids align. For unequal grids/channel counts, verify whether
separate groups are necessary rather than forcing padding or redesigning all
block handling in advance. Reference matching/named passes, when actually used,
identify their input result nodes/products explicitly; no recursive runtime
`refdata` object graph is needed.

## 6. Tensor layout, views, and ownership

Separate the **working tree** from the **exported product**. The working tree
retains SweepReducer's established raw root and namespaced reduced child:

```text
/                                             raw source Dataset
/tolteca_datamodels.toltec.kids.sweep            existing reduced result
/<sweep-check namespace>/sweep                 checked reduced-sweep result
/<sweep-check namespace>/quality               flags, masks, statistics
/<sweep-check namespace>/d21                   derivative/baseline results
/<kids-find namespace>/...                     detection/matching results
```

Namespace details for new stages remain to be chosen. Do not rename established
groups/variables just to implement the vertical slice. Preserve sibling nodes
when a processor runs or reruns; the present reducer rebuilds only its own child,
which needs correction for safe composition.

Export selects scientific result nodes and required metadata; it need not copy
the raw root. A metadata-only export root can avoid unrelated inherited dimensions.
For the first single reduced-sweep product, a standalone reduced Dataset is also
valid: ReducedSweepView already supports it. Test the reader with the actual
chosen representation, including metadata that must survive without the raw file.
The following is an illustrative later full-product layout, not a mandatory
namespace migration or the first-slice storage specification:

```text
arrays.zarr/
  /                         small format/observation metadata; no array dimensions
  /sweep/reduced             I, Q, unc_I, unc_Q; channel and sweep-step axes
  /sweep/checked             corrected I/Q and necessary sweep metadata
  /quality/samples           sample flags, spike mask, baseline mask
  /quality/channels          per-channel flags and statistics
  /quality/chunks            chunk bounds and statistics
  /d21/matched               per-channel derivative data
  /d21/unified               unified grid, coverage, baseline and uncertainties
  /detection/d21             detection traces, thresholds, labels
  /detection/s21             detection traces, thresholds, labels
  /matching/diagnostics      optional alignment/correlation arrays
  /noise                    optional raw-sample PSD and associated summaries
  /readout                  requested ADC/power diagnostic arrays
```

New export groups can describe scientific meaning. Namespacing is a storage
choice hidden by the mapper/view, not a reason to revise the settled flat
Mapping/Schema mechanism or duplicate raw arrays in published products.

DataTree enforces alignment with ancestors, while siblings may represent
independent grids. Its structure is containment; multi-input derivation belongs
in processing metadata/DB edges. These are consequences of the
[xarray hierarchical model](https://docs.xarray.dev/en/stable/user-guide/hierarchical-data.html),
and the dimension conflict was reproduced locally.

### Frequency and axis semantics

Distinguish `f_tone(channel)`, scalar `f_lo_center`, `f_chan(channel)`,
`f_sweep(sweep_step)`, and absolute `frequency(channel, sweep_step)`.
Use one documented formula with the v2 arithmetic order: channel reference
frequency is tone plus center, sweep offset is LO minus center, and the grid is
sweep offset plus channel reference. Algebraic reassociation can change the last
bits and is unnecessary for this port. Do not use `f_lo` for channel reference
frequency in one adapter and actual LO in another.

Keep original channel IDs, independent of frequency sorting. Express array
orientation with dimension names, then perform an explicit transpose before a
kernel that expects channel × sweep. Use exact alignment at stage input boundaries
where coordinates must match; avoid silent intersection or reindexing in science
arithmetic. [xarray.align](https://docs.xarray.dev/en/stable/generated/xarray.align.html)
provides that check without a custom alignment framework.

The established mapper remains the broker between source layouts and semantic
views. Preserve Dataset/DataArray/DataTree support and reusable resolved mapping
plans. Give the reducer the semantic inputs it needs; remove competing adapter
implementations of frequency/uncertainty calculation. Share the same sweep view
with D21 and plotting instead of `_D21SweepAdapter` and Web-local reconstruction.

### Mutation and caching

Treat caller inputs as read-only by API contract. Steps create new variables for
changed values and may share unchanged buffers. Xarray is mutable: a shallow copy
is **not** immutable protection, as the probe demonstrates. Do not modify a shared
`.values` array in place. Own a fresh work buffer when a numerical kernel mutates.

Publish the checked signal explicitly. Retaining both reduced and checked signals
supports diagnostics. Initially favor this simple complete representation over
delta encoding. Do not store a second full complex signal and amplitudes merely
because all were cached properties in v2; views can derive inexpensive expressions.
Conversely, retain expensive algorithm outputs used by several consumers.

Validate small schemas/configuration once at boundaries. Checks of names, units,
dtypes, dimensions, and metadata should not materialize arrays. At an eager
kernel boundary, load a block's required arrays together and reuse them. Cache
static schema/unit/serializer information, not arbitrary numerical input values.
View-derived data caches require an explicit immutable snapshot/ownership rule;
avoid another implicit source cache invalidation mechanism.

## 7. Rich result representation

| Current object/data | Native representation | Reconstituted behavior |
| --- | --- | --- |
| `SweepCheckData` signal flags, chunk windows and statistics | Named array datasets; store slice start/stop/step or explicit bounds rather than Python `slice` objects | Quality view exposes masks/statistics and can reconstruct indexing helpers. |
| `Peaks1DResult` | Trace/threshold/label arrays plus typed peak table; algorithm configuration in processing metadata | A peaks view implements selection masks using saved peak indices/widths. No peak detection rerun. |
| `Match1DResult` | Query/reference arrays, shift/correlation arrays, path indices, and edge/member tables with stable IDs | A match view draws plots or evaluates selections from those values. |
| NetworkX component sets/object table columns | `component_id`, member side, member ID records; relation rows carry query/ref IDs | Grouping is ordinary table selection; no pickled graph or object-dtype array. |
| DTW alignment object | Path indices and required scalar summary; cost matrix as an explicitly requested diagnostic array | Plot functions consume numerical components directly. Cost matrices are potentially quadratic, so storage is a deliberate option. |
| Detected/matched/channel tables | Native column variables with row IDs/order, units, masks, and flag definitions; Polars/Parquet projections, with separate table components only if justified | Shared table/domain views serve queries and plots. |
| Tone/model/power helper objects | Tone table, model identifier/parameters, units, and scientific values actually required by consumers | Reconstruct supported model/view behavior from a known schema, not arbitrary class import paths or linked Python objects. |
| Optional noise PSD | Frequency axis, I/Q PSD, phase-noise spectrum, selected frequency mask and per-channel summaries | Noise view and plots read saved values; no re-estimation from reduced sweep samples. |
| Figure objects/loggers/temporary work buffers | Runtime/debug only; optional rendered files as separate artifacts | Never needed to reopen a scientific result. |

The existing
[Peaks1DResult](/Users/ma/Codes/toltec/dev_toltec/python/tolteca/src/tolteca_kids/peaks1d.py:333)
and [Match1DResult](/Users/ma/Codes/toltec/dev_toltec/python/tolteca/src/tolteca_kids/match1d.py:283)
show that the useful behavior operates on concrete arrays and tables. There is no
requirement to make every runtime Python class serializable.

Quantities are straightforward with the existing unit accessor: numeric arrays
plus explicit units. For tables, use a shared column schema and write unit metadata
through Arrow; also expose it in the product manifest for clients that do not
preserve Arrow metadata. These copies are generated projections of one schema,
not independently maintained definitions.

Astropy now provides `to_df("polars")` through Narwhals, but its
[documented conversion limitations](https://docs.astropy.org/en/stable/table/dataframes.html)
include unit loss and loss of values hidden under masks. Use it only where the
conversion contract allows. Preserve masks separately where needed; do not conflate
null, NaN, zero, and a masked numerical payload. Exact export comparison should
retain hidden payloads with explicit companion fields if they are part of the
golden data contract. Astropy's native
[Parquet writer](https://docs.astropy.org/en/stable/io/unified_table_parquet.html)
is useful for baseline capture; it need not dictate the public v3 table schema.
[Arrow field metadata](https://arrow.apache.org/docs/python/generated/pyarrow.field.html)
provides a standard storage mechanism for per-column annotations.

No generic Polars mapper exists in the inspected Tollan tree. Start with the
concrete result table schema/readers. Extract a small reusable adapter into Tollan
only if the implementations demonstrate a common need; do not assume the pandas
mapper already gives Polars its semantics.

## 8. Configuration, analyses, steps, and execution entrypoints

### 8.1 Separate parameter objects from runtime configuration resolution

"Config" currently refers to several different responsibilities. Keep their
ownership explicit rather than adding nested generic Step machinery:

| Layer | Owns | Must not own |
| --- | --- | --- |
| Analysis/reducer object | Numerical parameters, defaults, validation, calculation; accepts ordinary Python arguments | YAML discovery, CLI parsing, runtime context, DB, scheduler |
| TolTEC step/pipeline | Typed stage config, enabled/disabled stages, input-view selection, invoking analyses, result placement | Another implementation of generic numerical parameters or per-entrypoint recipe merging |
| Domain config handler/resolver | Ordered sources, per-observation conditions, typed effective Kids config | Numerical reduction or mutable cross-observation science state |
| Public CLI/application runner | Input selection, configuration-resolution call, invoking pipeline, explicit output/publication, exit status | Alternative science/config implementations |
| toltecpipe/Dagster adapter | Triggering, resources, subprocess invocation, retries/cancellation, result registration coordination | Scientific defaults, recipe interpretation, another reduction pipeline |

The same operation can be used directly, inside a scientific step, and from an
orchestrated job without each use becoming a new kind of scientific Step.

The existing
[SweepCheckConfig.d21_analysis](/Users/ma/Codes/toltec/dev_toltec/python/tolteca/src/tolteca_kids/sweep_check.py:178)
already embeds `D21Analysis` as a typed parameter object. Apply the same idea to
the reducer stage: operational stage settings such as `enabled` belong to its
StepConfig; a nested SweepReducer owns the numerical parameters. For example,
conceptually `stage_config.reducer(raw)` invokes an ordinary configured Python
object. Do not reconstruct it from a serialized dict for every invocation or
duplicate its fields/defaults in several config classes.

The existing Step constructor accepts a validated StepConfig or validates keyword
arguments. It does **not** discover/merge all YAML sources. That happens above it
in RuntimeContext/ConfigHandler. Keep this division even if Step's result/context
handling is simplified later. The earlier recommendation to avoid a framework
rewrite does not mean removing the useful stage/config layer.

### 8.2 Reuse current Tollan configuration; restore the domain application layer

In v2, [tolteca_config](/Users/ma/Codes/toltec/toltec_astro/tolteca/src/tolteca_config/core.py:23)
adds TolTECA runtime information and handlers over legacy Tollan config machinery.
[Kids](/Users/ma/Codes/toltec/toltec_astro/tolteca/src/tolteca_kids/core.py:35)
extracts the `kids` subtree, validates KidsConfig, and constructs the configured
steps. The recipe dispatcher then supplies source context and invocation overrides.

V3 already has
[RuntimeContext](/Users/ma/Codes/toltec/dev_toltec/python/tollan/src/tollan/config/runtime_context.py:69),
[ConfigHandler](/Users/ma/Codes/toltec/dev_toltec/python/tollan/src/tollan/config/handler.py:38),
SubConfigKeyTransformer, ordered source lists, YAML/dict/env-file sources, a CLI
factory, and conditional loading. There is no current `tolteca_config` source
package. The reducer-only [tolteca.cli](/Users/ma/Codes/toltec/dev_toltec/python/tolteca/src/tolteca/cli/__init__.py:1)
now uses a small TolTECA domain composition/CLI layer over Tollan's generic
infrastructure; do not copy the entire legacy config
backend back in or move instrument defaults into Tollan. The exact domain module
name is less important than this ownership boundary.

Before each observation/interface/block execution:

1. Resolve input/location selection and read only the metadata needed to identify
   the source and evaluate conditions. Bootstrap input selection must not require
   running scientific algorithms to obtain their own configuration.
2. Build the condition context from typed source metadata (`roach`, data kind,
   the legacy `file_suffix` meaning, selected block where relevant).
3. Resolve ordered defaults/site/user files, optional explicit env-file inputs,
   and invocation overrides; validate the domain config within that context.
4. Capture a standalone effective config for that work item and leave the mutable
   resolution context behind. Stage execution uses those typed objects.
5. Record the effective scientific settings and source selection with the output.

One resolver is used regardless of CLI, direct application invocation, or job
trigger. Source precedence, deep-merge/list behavior, relative path bases, unknown
keys, disabled stages, and environment inputs need explicit tests against the
actual v2 recipes. CLI/job overrides must have the same precedence; do not infer
defaults independently from scheduler configuration.

Configuration contract:

- Retain the established pandas-style condition language: `&`, `|`, `~`, `in`,
  `not in`, and comparison-before-boolean precedence. For example,
  `roach >= 7 & file_suffix in ["targsweep", "tune"]` needs no extra parentheses
  or syntax migration. Keep evaluation confined to `is_enabled_for`, using the
  v2 `DataFrame.query(parser="pandas")` approach. A replacement evaluator needs
  a demonstrated benefit and tests preserving the public recipe language.
- Recipes are trusted operator inputs. Validate the metadata needed by the
  selected workflow before evaluating conditions. Pandas does not guarantee
  logical short-circuiting: absent context or an undefined variable disables a
  conditional source, while other query errors propagate. Short-circuiting is
  a reasonable future scalar-condition behavior, not a reason to introduce a
  custom parser now. Use independent runtimes for concurrent observations.
- Load YAML mappings and conditional source-list recipes through the same
  RuntimeContext path. Nested lists have local ordering; included file paths are
  relative to their recipe. Reject malformed source documents explicitly.
- Cache parsed source definitions, not context-dependent merged results. Merge
  dict/list containers without modifying cached sources or copying opaque
  analysis objects. Handlers validate their domain config eagerly and retain it
  for their lifetime. New settings produce a new handler, with independent
  resource caches. Temporary variants own their tiers and validation context,
  while reusing user-source reads. Retain the resolved typed config for stage
  execution.

Implementation status (2026-09-14): the focused handler/runtime-variant changes
are committed in Tollan as `92af938`. The
[configuration guide](/Users/ma/Codes/toltec/dev_toltec/python/tollan/docs/configuration.md)
describes their public use. Tests cover handler lifetime, independent tiers and
contexts, nested conditional sources, shared file reads, and context-aware paths.
The subsequent focused pandas-condition patch is committed as `ec87e0a`.
A fresh cross-environment check compared 468 results from the actual v2
ConfigSource (pandas 2.3.3) and current v3 (pandas 3.0.5): all matched. It covers
the six conditions in the production KIDs override recipe and six language
combinations across 13 networks and three sweep suffixes. V3 retains its
single-observation dictionary context API.

The focused YAML source-list patch is committed in Tollan as `22fd695`. Lists
are parsed into the existing ConfigSourceList, with local ordering and relative
file references; cached definitions are evaluated for each current context.
A fresh comparison loaded the complete production
`10_kids_override.yaml` in both v2 and v3 across 13 networks and three sweep
suffixes: all 39 merged configuration dictionaries matched. This establishes
parity for that recipe, not all legacy configuration features or the full CLI.
The reducer-only shared runner and CLI now use these source/handler facilities.
`Kids` owns configured `SweepReduce` and `DataProdOutput` steps and assembles
the pipeline; it has no `reduce_sweep` execution method. Shared datamodels
provide source records, filename discovery, a metadata-only pandas catalog,
and local observation selectors. The runner enriches scalar header metadata,
constructs an input-specific handler, wraps a working DataTree, and invokes
`kids.pipeline_for(info.data_kind)`. The final configured output step writes
the product. `SweepReduceConfig.reducer` is the numerical
object itself, and direct reducer use remains independent of runtime machinery.

The working tree's identity is stable. A step places the independent reducer's
result node into that tree and returns a Boolean internally. Tollan's final
`set_context` hook allows the step to persist JSON invocation metadata beside
native results. This retains standalone step calls and the established
pipeline contract without a second authoritative StepData payload.

`ensure_datatree` is the only wrapping operation: it does not identify sources,
resolve configuration, or alter numerical data. Source metadata is attached
separately to a `tolteca_kids` execution node. Each workflow uses an explicit
step list and the same SequentialPipeline engine, not repeated type branches
inside every stage. Only the raw sweep list is implemented; timestream and
processed inputs are recognized and rejected explicitly until their scientific
workflows are ported. The existing TimestreamReducer is not yet a verified
replacement for the production noise-reduction workflow.

`KidsConfig.output` owns publication path/name/enabled policy. Output is opt-in
for in-memory use; `--output` enables it for CLI execution, with explicit dotted
settings taking precedence. Database registration remains outside this step.
The DB scanner now adapts the shared source parser. Its existing query DTOs and
enum definitions remain review work. The contained follow-up consolidates the
selector language into datamodels ObsSpec, consumed by local catalogs and both
DB query APIs. Its acceptance target is CLI -> ObsSpec/catalog -> per-input
configuration -> wrapped DataTree -> real no-op StepContext completion, with
rich debug logging and no numerical computation or publication. `--dry_run`
selects this path, including for NetCDF timestreams. It does not establish a
DB-backed CLI or settle the pending reducer/view/metadata/persistence review.

The old Zarr-oriented `KidsPipeline`, with its separate sample averaging, is
untouched; consumers should migrate to the shared view rather than extending
that duplicate reduction path. Unsupported later-stage sections remain rejected.

Current validation and actual DB/Web integration boundaries are recorded in
[the slice status](kids-reducer-slice-status.md). The local probe passes through
the real ingestion, ORM, catalog, and a new shared-view figure consumer. It does
not yet replace the existing Web page or provide a production publication API.
Saved-config CLI replay, recorded argv/input fingerprints, broader metadata and
irregular-input coverage, and the remaining A4 criteria are still outstanding.

### 8.3 CLI-first orchestration, with independently callable library code

Retain the established toltecpipe → dispatcher/CLI advantage reported by the user.
The inspected
[shell dispatcher](/Users/ma/Codes/toltec/toltec_astro/taco_recipes/scripts/kids/dispatch_tolteca_kids.sh:1)
selects the environment and passes paths/config to the Python dispatcher. In v3,
make the public `tolteca` reduction command the supported execution boundary:

```text
terminal / toltecpipe / Dagster
          ↓ same public reduction CLI (subprocess for an orchestrator)
    TolTECA input/config resolver and application runner
          ↓ resolved typed stage configuration
    TolTEC pipeline / steps → standalone reducer and kidsproc analyses
          ↓ data-backed results
    product writer → optional DB registration → catalog → Web read-only views
```

The command syntax must be finalized and documented; `tolteca kids reduce ...`
is an illustrative direction, not an existing command. Bring the Python
dispatcher's observation selection, context setup, config overrides, and reporting
into the application library/public CLI. taco_recipes can retain site recipe files
and a thin environment wrapper, without owning a second scientific dispatcher.

Contract for the CLI and scheduler adapter:

- Accept explicit files or supported observation selection, config file/directory,
  nested overrides, block selection, and output location. Preserve the useful
  complex-config invocation capability; test existing `--kids.*` override forms
  or provide a documented explicit mapping if names change.
- Provide a resolve/show-config mode that evaluates the selected source's
  conditions without running reduction. Fail early on invalid consumed config.
- Launch the CLI with an argument vector, not shell-interpolated command text.
  Record executable/environment, argv, working directory where relevant, source
  selection, and a reference to the effective config. Keep credentials out of
  copied commands, logs, and replay files.
- Distinguish **rerun with current recipes** from **replay saved effective config**.
  Replay must not reevaluate changed site files, environment-derived settings,
  or conditional rules. Record software/input revisions; this enables diagnosis
  but does not by itself guarantee identical results across different versions.
- Produce human-readable logs, a reliable exit status, and a small machine-readable
  run/result manifest containing output references and per-input outcomes. Do not
  make Dagster scrape progress text or pass live Python contexts across processes.
  Cancellation/failure must not publish incomplete products as available.
- Keep the command useful without Dagster/toltecpipe. Standalone reduction and
  output writing must also work without a DB connection; registration is an
  explicit integration action. The vertical-slice workflow enables it so Web
  discovers real registered products. Choose one registration owner per run to
  prevent CLI and adapter from registering duplicates.
- DB ingestion may trigger/enqueue this same command after source readiness. The
  metadata ingestor itself does not gain a hidden numerical pipeline. Retries and
  resource limits belong to the orchestrator; scientific config belongs to the
  saved domain request. Use a process per sensible job/batch, not per array getter
  or D21 subroutine.

This is CLI-first for operational execution, not CLI-only for Python users.
`SweepReducer(...)`, `D21Analysis(...)`, and the resolved in-process pipeline stay
directly callable. They do not read sys.argv or require a subprocess. The CLI
calls that same library implementation, so notebooks and tests do not need to
launch a process just to analyze a dataset.

### 8.4 Runtime context and data-backed processing metadata

The v2 design had useful properties: frozen typed configuration, straightforward
ordered execution, typed cross-step access, and a deliberate boundary around
potential outputs. The new native result model can retain those benefits without
making runtime context the scientific data bus.

### Proposed responsibilities

| Concept | Responsibility |
| --- | --- |
| Configured analysis/reducer | Validate numerical parameters and run independently on supplied data/views. |
| TolTEC step/pipeline | Use resolved stage config, select views, invoke analyses, and assemble result nodes. Exceptions indicate execution failure. |
| Scientific result | Own/reference the native arrays and tables necessary for later stages and consumers. |
| Processing metadata | Record effective configuration, algorithm/schema version, selected inputs/block, and result identity. Small and serializable. |
| Optional runtime context | Logging, timing, cancellation hooks if needed, and requested debug capture. Explicitly owned by the caller/run; not necessary for science correctness. |
| DB/orchestrator | Product/source registration, dependency edges, retry/scheduling, and durable operational history. |

An execution context may contain scratch references during a run, but later
science stages resolve required result nodes through explicit views of the working
tree; low-level numerical functions receive those values as arguments. Debug capture can
optionally return a run-owned object; it must not leak into array attributes or
become necessary to draw saved diagnostic products.

For convenient inspection, expose processing metadata on the result/view. Small
serialized per-stage records may live in group attrs. This preserves the intended
ability to inspect how a data container was produced. Do not store live
`StepContextDict`, array-bearing Pydantic objects, or a product tree inside attrs.

Prefer this explicit ownership over the old audit's weak sidecar. A sidecar was
motivated by retaining `Step.get_context(data)` verbatim; that compatibility
constraint does not exist. Its identity/copy/lifetime complexity provides little
benefit once science outputs are first-class values.

No generic Step redesign is part of the first reducer/DB/Web slice. SweepReducer
already demonstrates a useful configured callable without Step inheritance.
Generic analyses retain that independence. The domain Step/config layer remains
useful for composition; introducing the reducer stage should require only a thin
adapter, not a new hierarchy of execution engines. When adapting the existing
Step machinery to native outputs, preserve the caller's working tree and place
the numerical result into its own node. Keep the Boolean internal completion
indicator and return the same tree from the public call. Reuse existing bindings
and extract only demonstrated common behavior.

The old audit's universal JSON-string requirement was based on NetCDF transport.
The chosen Zarr/Parquet product need not force that restriction on every attribute:
use backend-appropriate small metadata, with serialization at the storage boundary.
An optional NetCDF exporter can encode its own supported representation. There is
also no need to introduce OpenLineage/W3C PROV runtime dependencies to record a
small ordered list of processing steps.

## 9. Products, DB, and publication

The first slice publishes a **reduced-sweep product**, not a completed KIDs
detection product. It needs the reduced arrays, uncertainties, axes, channel
metadata, observation/interface/block identification, effective reducer config,
source association, and publication metadata. No detection table, KidsFind group,
or derived detection-quality score is required to call this product complete.

The catalog's Parquet projection is distinct from scientific table exports:
the first slice needs product discovery rows even though it has no detection
tables. Generate catalog rows from registered products, not a separately curated
list of paths.

Suggested later full-KIDs durable product layout:

```text
<product-location>/
  manifest.json
  arrays.zarr/
  tables/
    detections.parquet
    channels.parquet
    matches.parquet
    group_members.parquet
    peaks_d21.parquet
    peaks_s21.parquet
    tones.parquet             when tone generation is requested
```

One representation is authoritative for each field. A detection catalog should
not be independently edited in both a Zarr node and Parquet. Index/summary copies
are derived projections carrying their source product identity.

The manifest minimally identifies format/schema version, observation/interface/
block, the effective configuration and implementation, role-labelled input IDs,
component locations/schema IDs, and included diagnostics. Include stable order
keys and flag/unit definitions via the shared result schema. Track:

- **Recipe signature:** deterministic inputs + selected block + effective science
  config + implementation/schema version; used for candidate reuse.
- **Execution ID:** identifies a particular attempt.
- **Product identity/content digest:** identifies a completed result; follows the
  DB's documented product contract, with a declared digest algorithm.
- **Physical source URI:** where one copy resides. It is not the scientific identity.

A single path, config hash, or existing Zarr group cannot substitute for all four.
Where a raw logical ID can refer to changing/incomplete bytes, recipe reuse must
also use a verified immutable input revision/fingerprint. Include input roles and
ordering where significant rather than blindly applying a set hash.

Use one explicit hashing algorithm for the new recipe signature (SHA-256 is a
reasonable stdlib default), not an optional-package-dependent fallback. Preserve
non-finite configuration values distinctly from null. Pydantic provides supported
[serialization controls](https://docs.pydantic.dev/latest/api/config/), including
`ser_json_inf_nan`; test the concrete nested config corpus with a schema-versioned
encoding. Do not claim plain sorted JSON implements RFC 8785. No accessor should
hash entire lazy arrays merely to answer a field request.

### Publication boundary

Write new products separately from read-only inputs. For local storage, stage in a
new directory and publish only after writing and validating all required
components; a same-filesystem rename can be the final local operation. For object
storage/HTTP, use a unique prefix and a final completion manifest/catalog update;
do not assume a directory rename is atomic. The reader must reject incomplete
products, and registration should make them available only after completion.
Use unique attempt paths and explicit catalog conflict handling for concurrent
writers. This is a small publication contract, not a distributed transaction
framework.

`DataTree.to_zarr()` handles array serialization, not an atomic multi-format
product commit. Its
[documented behavior](https://docs.xarray.dev/en/stable/generated/xarray.DataTree.to_zarr.html)
includes eager metadata writes even with `compute=False`, and warns about
overwriting a lazily opened source. These facts rule out treating group existence
as completion.

The [current DB metadata](/Users/ma/Codes/toltec/dev_toltec/python/tolteca_db/src/tolteca_db/models/metadata.py:161)
and [ProcessContext](/Users/ma/Codes/toltec/dev_toltec/python/tolteca_db/src/tolteca_db/models/metadata.py:432)
are useful starting points, but need enough information to identify this product
and its processing inputs. They use dataclasses/Adaptix, which need not be replaced
wholesale by Pydantic. Adapt shared domain metadata at the DB boundary; do not make
the numerical package import ORM models.

The existing DB distinction between a logical product, physical sources, and
directed associations should remain. Preserve its existing edge direction
(`derived/containing product → dependency/member`) unless deliberately migrated.
One per-interface/block computational output need not redefine the identity of the
observation-level raw product. Group related component products through the DB's
association model rather than forcing them into a common xarray dimension.

## 10. Consumer-driven design and performance

Web should resolve a product, inspect its metadata, then request selected array
groups and table columns/rows. Plot functions accept saved-data views. Changing
a tab must not start numerical reduction. An explicit new reduction request can
be a separate workflow, with a new config and result identity.

Keep `ObsCatalogBackend` / `ObsDataService` as the current useful division. Extend
the location result beyond a local `Path` to a URI and product/schema identity.
Move frequency reconstruction and other scientific interpretation out of Web into
the shared model/view package. Bound open-store caches, close on eviction, and
key by immutable product identity/location revision. Avoid indefinite caching of
negative catalog results while new products are arriving.

### Performance rules to verify

- Preserve lazy source access until a numerical stage actually needs values.
  Small reference sweeps can reasonably run eagerly; D21 unification, peak
  detection, and DTW should not be declared chunk-independent without evidence.
- Initially parallelize independent observations/interfaces/blocks, retaining
  the legacy reduction order within a science kernel. Avoid Dask reductions that
  regroup floating-point arithmetic before parity is established.
- At a stage boundary, materialize shared required inputs together and avoid
  repeated `.values`/`.compute()` on the same lazy work. Bound concurrency by peak
  work-buffer memory, not just source array size. This follows
  [Dask best practices](https://docs.dask.org/en/stable/best-practices.html).
- Keep the established unit accessor; do not materialize a whole lazy dataset to
  construct quantities. Local eager kernel conversion is explicit and measurable.
- Large matching cost matrices are optional diagnostic products. The metadata
  records whether they were saved; Web cannot silently rebuild them. Standard
  diagnostics needed for supported plots must otherwise be persisted completely.
- Avoid full expanded frequency grids in storage if the exact reconstruction
  contract suffices; compute selected grids through the shared view. Store a full
  grid when it is genuinely irregular or necessary to preserve exact values.
- Benchmark channel-select and overview reads separately. Single-channel chunks
  are not automatically optimal: request overhead and file counts matter.
  [Zarr's performance guide](https://zarr.readthedocs.io/en/latest/user-guide/performance/)
  recommends choosing chunk/shard shapes for access patterns. Start a measured
  comparison with channel chunks of 1, 16, and 64 over a full small sweep, not a
  universal fixed rule.
- Evaluate Zarr v3 sharding for the served data. Current Zarr 3.3 includes
  [sharded-read coalescing](https://zarr.readthedocs.io/en/latest/user-guide/examples/sharding_coalescing/).
  Whether it helps depends on server range-request support, caching, request
  latency, and selection patterns; the local layout probe is not a benchmark.
- Record wall time, peak RSS, bytes read/written, Dask task execution, and remote
  request counts. Measure first open, one-channel plot, observation overview,
  catalog query, and repeated access. Do not add a specialized data server until
  this simpler architecture shows a measured shortfall.

Choose one tested Zarr format/codec profile for the initial product schema. The
probe supports trying v3; verify deployment/client versions before adopting it.
Do not promise that arbitrary codec choices, complex types, or inherited metadata
will work with every consumer. Storing I/Q separately also gives straightforward
non-Python access without a custom complex decoder.

## 11. Package changes justified by this design

| Package | Work | Why |
| --- | --- | --- |
| `tollan` | Reuse mapping/generic-binding/unit and config-source/RuntimeContext/ConfigHandler infrastructure. Fix demonstrated recipe/caching gaps; keep Step changes minimal. | Shared configuration and analysis support remain independent of TolTEC/DB. |
| `tolteca_datamodels` | Correct raw axis/block/tone metadata, unify reader semantics, define shared product metadata/readers and instrument sweep views. | All producers and consumers need one interpretation of TolTEC data. |
| `tolteca_kidsproc` | Complete reusable sweep/uncertainty/D21 views and tested numerical functions with explicit result inputs. | Generic KIDs calculations should not depend on file layout or operational context. |
| `tolteca_kids` | Explicit SweepCheck → KidsFind data flow; native result datasets/tables; complete product writer; diagnostic views; optional tone-output producer. | This is the domain pipeline and its scientific product contract. |
| `tolteca` application/CLI layer | Compose domain config handling, source selection, public reduction command, saved-config replay, and job reporting over library functions. | Replace dispatcher-owned application logic with one entrypoint shared by terminal and orchestrators. |
| `tolteca_db` | Register completed results and component sources; preserve input associations; add reduction orchestration and product discovery/export integration where absent. | Orchestration/availability belong outside numerical kernels. |
| `tolteca_web` | Read product-backed diagnostics and shared views; URI-aware catalog/service; bounded identity-keyed caches. | Correct remote consumption without recomputing or guessing source semantics. |

Definitions should live at the lowest layer that actually shares them, without
creating import cycles. Generic KIDs result semantics can live in kidsproc/domain
code; TolTEC observation metadata remains in datamodels. DB and Web may depend on
both. No repo reorganization or new umbrella framework is needed in advance.

## 12. Implementation sequence and acceptance gates

### A. First milestone: reduced-sweep vertical slice

This is the agreed first deliverable. It includes Tolteca, Tolteca DB, and Tolteca
Web together, with only necessary Tollan fixes. A passing reducer unit test alone
does not complete it.

```text
read-only reference sweep .nc
  → ingest raw product/source metadata in the local DB
  → public reduction CLI (terminal or orchestrator), shared per-source config resolver
  → open via shared raw schema/mapper/accessor
  → existing SweepReducer → DataTree → ReducedSweepView
  → write and reopen a reduced-sweep Zarr product
  → register completed product, physical location, and raw-source association
  → query/export catalog from those registrations
  → ObsDataService → shared ReducedSweepView → sweep visualization
```

Raw ingestion records what exists; reduction creates a derived scientific product;
publication/registration makes that result discoverable. One local workflow can
expose these operations without conflating identities or placing numerical
reduction in the ingestor/viewer.

#### A0. Connect the public CLI and shared configuration path

Current status: shared runner, domain handler, metadata-based conditions,
public CLI, show-config, and completed-product manifest are implemented locally.
Saved-config CLI replay and recorded invocation arguments remain outstanding.

- Restore a useful public reduction CLI over a shared application runner. The
  first target is reducer-only processing; full KIDs steps follow in B/C.
- Reuse Tollan's source/context/handler machinery and introduce the small domain
  config adapter, with typed reducer parameters nested in its stage config.
  Keep direct SweepReducer construction independent of the runner and CLI.
- Retain the condition language and confined pandas evaluator described in §8.2;
  no expression-syntax migration is part of this phase. Tollan's loading/cache
  fixes are committed; reducer-only application integration is implemented.
- Test recipe file/directory loading, conditional-list handling, expression truth
  tables, precedence, nested overrides, units/paths, and per-observation handler
  lifetime. Validate the settings consumed by this slice; later-stage settings
  must be explicitly deferred, not silently claimed to execute.
- Add show/effective-config output, saved-config replay, exit-status/run-manifest
  behavior, and a recorded argv. Exercise the same command directly and through
  a small subprocess adapter; a live Dagster service is not required to prove
  scheduler independence.
- Source/configuration loading and replay are application concerns. Do not add
  CLI or RuntimeContext requirements to reducer or generic analysis classes.

#### A1. Establish reducer parity and complete the shared view

Current status: exact I/Q, uncertainty, frequency, and unit-meaning comparisons
pass for the target/VNA fixtures and both tune blocks. Broader channel metadata,
input fingerprinting, non-finite/masked and irregular-window coverage remain.

- Build the first part of the local two-environment harness: legacy NcFileIO in
  `tolteca_v2`, current accessors/SweepReducer in the development venv. Capture
  input interpretation, reduced I/Q, uncertainties, frequency grid, channel
  metadata, and block selection. A full KidsFind baseline is not a dependency.
- Start with the target sweep as the end-to-end example; acceptance also includes
  the reference VNA and multi-block tune, the legacy-selected last block, and an
  explicit non-default block. Record input fingerprints, effective configuration,
  environment/revisions, and dirty state. Expected-output regeneration is explicit.
- Retain `SweepReducer.__call__ -> DataTree` and `dt.toltec_kids.sweep`.
  Complete ReducedSweepView's frequency, I/Q, uncertainty, channel, and typed
  metadata access so all consumers use the same semantics.
- Correct raw axis roles, tone metadata selection, sample-window boundaries,
  masking/scaling/dtype behavior, and mean/std conventions. Route raw access
  through the raw mapper/view. Do not create another reducer in DB or Web.
  Verify equal-frequency grouping against legacy sample windows rather than
  assuming equivalence for irregular/multi-block input.
- Preserve unrelated child nodes on rerun and respect source-buffer ownership.
  Required missing metadata must not become fabricated frequencies/amplitudes.

#### A2. Ingest, persist, register, and discover the actual product

Current status: local Zarr products and explicit DB-to-Parquet integration probe
work. The probe uses actual ingestion/ORM classes with disposable SQLite state;
this is not yet a production registration/transaction API.

- Use the existing
  [DataIngestor](/Users/ma/Codes/toltec/dev_toltec/python/tolteca_db/src/tolteca_db/ingest/ingest.py:168)
  and product/source/association models as starting points. Inspect their source
  extraction and make necessary changes for the shared raw sweep semantics;
  do not create a second observation identity convention.
- Keep ingestion metadata-only where possible: shape/header reads are acceptable,
  but cataloging should not load I/Q. Reduction is the intentional numerical
  materialization boundary.
- Write a reduced product separately from the fixture. Persist enough axes,
  channel metadata, units, reducer configuration, and observation/block identity
  to construct the reduced view without reopening the raw file. Test both the
  in-memory tree view and reopened product view.
- Register a completed **reduced sweep**, its location, and its dependence on the
  raw product/source and selected block. Do not require KidsFind results or claim
  detection completeness. Preserve existing association direction.
- Generate Web discovery data from actual DB registrations. Reuse the Parquet
  catalog backend where practical, adjusting the projection for product kind,
  schema, interface/block, configuration/version, location, and availability.
  Hand-authored catalog rows are not end-to-end integration.
- Make the Dagster/toltecpipe trigger an adapter to the same public CLI, with
  recorded arguments, effective config, and machine-readable output references.
  The command works without live telescope services or a running scheduler;
  sensor/partitioning redesign is not required for this slice.

#### A3. Visualize through the shared reduced view

Current status: the Web figure consumer plots saved I/Q and shared-view
frequencies; probe-generated HTML payloads match the saved arrays exactly.
Existing page integration, full interaction, and remote-product access remain.

- Reuse the existing
  [sweep viewer](/Users/ma/Codes/toltec/dev_toltec/python/tolteca_web/src/tolteca_web/sweep/sweep_viewer.py:222),
  [catalog interface](/Users/ma/Codes/toltec/dev_toltec/python/tolteca_web/src/tolteca_web/obs/catalog.py:57),
  and ObsDataService. Select a registered reduced product by observation,
  interface, and block, with explicit result/configuration identity when multiple
  reductions exist.
- Replace this path's use of
  [ZarrSweepDataset](/Users/ma/Codes/toltec/dev_toltec/python/tolteca_web/src/tolteca_web/sweep/zarr_model.py:34)
  and its separate frequency reconstruction with ReducedSweepView. Do not convert
  the result into the old simplified Web schema to preserve that adapter.
- Show reduced I/Q, amplitude/phase or IQ-plane views, uncertainties where
  applicable, channel selection, and observation/block/reducer metadata. Compare
  numerical plot payloads to the reopened shared view. Raw versus reduced identity
  must be visible; the viewer must not silently average samples.
- Plotting calls neither SweepReducer nor SweepCheck/KidsFind. Missing reduction
  is an availability state; any processing action is separate from rendering.
- Make URI-aware lookup work in both data access and the viewer's availability
  matrix, which also currently checks local paths. A remote product must not
  require a local mirror. Exercise a controlled HTTP-served copy locally before
  relying on a production network.

#### A4. Definition of done

| Check | Required evidence |
| --- | --- |
| Scientific parity | Exact reducer/view agreement with v2 for all three reference sweep kinds, including metadata/units, non-finite values, channel order, and block selection. |
| Config/entrypoint parity | The CLI and orchestrator subprocess path resolve the same per-source config and produce identical numerical outputs. Changing recipe files does not change a saved-config replay. No stale handler settings leak between observations. |
| Persistent equivalence | Reopened reduced views match in-memory views; no raw-file dependency, live context, or separate Web frequency calculation. |
| Ingestion/registration | Query raw/reduced products, sources, and their association from the local DB; repeated ingestion does not duplicate identity. |
| Real discovery | A catalog query derived from registrations locates the result; no manually entered filesystem path or fabricated catalog row. |
| Actual visualization | Browser smoke test selecting observation/interface/block and channels; inspect sweep and metadata. Automated plot-payload checks verify values. |
| Isolation | Instrument reduction calls during display: zero. Fixtures remain unchanged; artifacts/DB/catalog/logs stay in test directories or the development run tree. |
| Publication/reuse | Incomplete writes are not offered as usable products; reruns preserve previous products; input revision/config distinguish reuse candidates. |
| Access behavior | Ingestion/metadata opening does not eagerly read I/Q; channel selection precedes loading plotted arrays. Record reads, time, and memory locally and over controlled HTTP. |
| Local QA | Affected tests and applicable `just` QA recipes for every touched repository; report unrelated existing failures separately. No CI addition. |

Deliver the public reducer invocation and a local recipe for fixture ingestion
through publication/catalog generation, plus replay and viewer launch/review
instructions. Reuse the current CLI/config building blocks rather than designing
a new CLI framework. The slice completes only when the command can be rerun
outside orchestration and the actual catalog-backed viewer works.

Out of scope for A: peak finding, despiking, tone output, a generalized result
bundle, Step/StepContext redesign, production service deployment, and exhaustive
remote chunk tuning. These must not delay the reducer vertical slice.
Bounded configuration fixes, the domain stage adapter, and restoring the public
CLI are in scope; "no Step redesign" must not be read as "no application config".

### B. Extend the same slice through SweepCheck

Following the reducer example, add SweepCheck's result schema, mapper, and view.
Return an enriched tree with checked I/Q and quality/D21 nodes. KidsFind's eventual
input is `SweepCheckView(dt).sweep`, not the original reduced node. Extend the v2
baseline and the same publisher/catalog/viewer to this stage, rather than waiting
for the full pipeline to prove persistence and display.

Acceptance: exact SweepCheck checkpoints, preserved raw/reduced input, correct
despike on/off behavior, close/reopen through typed views, and saved quality
diagnostics visible without reduction in Web. Add explicit noise/power branches
and comparison profiles as that functionality is ported; raw PSD uses unreduced
samples. Generic Tollan changes require a demonstrated benefit here.

### C. Extend through detection, matching, and tone products

Use the same processor/result-node/view pattern. Normalize peak/matching objects
into native arrays/columns; test mask/ragged representations before selecting any
separate table component. Extend the baseline to both legacy final table
projections and enabled tone outputs, controlling phase generation and using
temporary instrument-output destinations.

Acceptance: exact final values, selections, masks, units, order, and tone outputs;
data-backed peak/match plots after reopen; no required StepData/DTW Python instance.
Add Parquet science-table projections and discovery as part of this slice. Missing
optional expensive diagnostics are reported, not recomputed by Web.

### D. Consolidate and measure the completed pipeline

Remove obsolete adapters/payload classes, consolidate demonstrated common helpers,
and document the accepted public API. Benchmark full reduction and Web access
before tuning chunks/shards or concurrency. Keep scientific parity gates active
while optimizing. Update design status separately from public docs and run local
QA across all touched repositories.

Each milestone can have small reviewable multi-repo commits, but its acceptance
gate crosses producer, persisted representation, discovery, and consumer.

## 13. Agreed direction and remaining choices

Agreed direction:

- SweepReducer's configured callable, DataTree result, and schema/mapper/view
  pattern are the foundation. Complete the reducer end to end first.
- DB ingestion/product registration and Web visualization are part of that first
  milestone, not deferred final integration tasks.
- Generic kidsproc analyses remain independently callable parameter objects;
  TolTEC steps/configuration select and compose them over instrument data views.
- Operational execution uses the public reduction CLI from terminal or
  toltecpipe/Dagster. Library analyses remain callable without that CLI. Shared
  config resolution and saved effective-config replay prevent entrypoint drift.
- Scientific result nodes replace duplicate StepData inventories; typed views
  make those results usable in memory and after reopening.
- Source fixtures are read-only; derived products are published separately.
- Scientific values match v2, with reducer-level evidence first and broader
  comparisons as stages are added.

Choices to resolve within A: the minimal self-contained reduced product layout,
necessary view metadata fields, catalog product/block selection, and explicit
reuse/publication semantics. Preserve existing conventions when sufficient and
justify changes through the end-to-end tests.

Later choices, not prerequisites: special rich-table storage, generic Step/context
helpers, full matching diagnostics, and network chunk/shard optimization. The
previous mixed KidsProducts proposal and mandatory coordinate-free working root
are superseded; an export projection may still use a metadata-only root.
