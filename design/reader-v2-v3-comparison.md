# KIDs and telescope readers: v2 → current v3 behavior and computation audit

Date: 2026-09-18. Scope: the installed `tolteca_v2` readers versus the current
Python workspace, including its uncommitted reader/reducer changes. This is a
comparison of existing behavior, not a proposal to reproduce v2's architecture.
Sections 1–9 and the summary below retain the **initial audit baseline**, before
reader corrections. No reader implementation was changed for that baseline.
The current implementation progress, decisions, and exceptions awaiting review
are recorded in [section 10](#10-reader-correction-iterations). Its telescope
results supersede the baseline telescope findings where explicitly listed.
The current telescope contract is **T5**, with shared path naming updated in **C3** below. T1/T2 and C1/C2 are historical
iteration records, not competing descriptions of the current API.

## Summary

The reference sweep **numerical reduction path works**, but the surrounding v3
reader interface is **not yet feature-complete or uniformly correct**.

- All four existing persisted-sweep parity cases pass: VNA, target, tune last
  block, and tune first block. They check exact I/Q, standard deviations,
  channel/sweep/full frequencies, and units against the actual v2 interpreter.
- Full reference timestream I/Q values match. All four telescope files have
  matching RA, Dec, azimuth, elevation and hold values. Telescope elapsed time
  matches exactly after explicitly converting v3's absolute timestamps.
- KIDs metadata avoids loading ADC snapshots and avoids LO arrays except for
  tune block detection. This is a real reduction in materialization.
- There are confirmed correctness/performance problems outside that passing
  path: oversized channel tables, wrong-axis channel selection, unavailable raw
  sweep tables, missing model-parameter table access, and multi-block reduction
  that broadcasts and sometimes loads unrelated headers.
- Telescope metadata has deliberate or unresolved semantic differences: time
  origin, interface/component names, header-versus-filename identifiers,
  integration time, goal validation, and omission of the observing program.
- Small-file telescope opening plus metadata is approximately 3× slower here.
  Lazy array access does not mean that every operation is faster.

The immediate work should be reader correctness and explicit metadata/array
contracts—not another redesign of schema/mapper/accessor.

## 1. Evidence and boundaries

### Implementations examined

| Layer | v2 | v3 |
|---|---|---|
| KIDs reader | `tolteca_datamodels/toltec/ncfile.py`, `toltec/core.py` | [`kids/core.py`](../src/tolteca_datamodels/toltec/kids/core.py) |
| Telescope reader | `tolteca_datamodels/lmt/tel.py`, `io/core.py`, `io/ncfile.py` | [`lmt/tel.py`](../src/tolteca_datamodels/lmt/tel.py) |
| Low-level mapping | `tollan/utils/nc.py` | Tollan `accessor/mapper.py`, `accessor/xarray/mapper.py` |
| Sweep computation | KIDs reader `_read_sliced()` and `MultiSweep` construction | [`kids/sweep.py`](../src/tolteca_datamodels/toltec/kids/sweep.py), `ReducedSweepView` |
| Processed/timestream fields | KIDs reader's `data`/`data_extra` mappings | [`kids/timestream.py`](../src/tolteca_datamodels/toltec/kids/timestream.py), generic `tolteca_kidsproc` views |
| Metadata/discovery | Reader dictionaries, filename parsing, telescope model | `toltec/metadata.py`, `lmt/metadata.py`, `lmt_base/metadata.py`, `toltec/file.py`, `kids/source.py` |

v2 source root is `/Users/ma/Codes/toltec/toltec_astro/tolteca`; HEAD
`54f882373bf90d97665c095e11f27f3f082a39d5`, with no tracked modifications.
Its Tollan HEAD is `444ce1491cfcf91bc9485af4616e30c25f01e54a`.
v3 Tolteca HEAD is `6ef2c1617ccb071b74f1142aac6e9c5d6ed7e56c` **plus the current
working tree**. v3 Tollan mapper changes are committed as
`658b7014a4379c8779c1e3e5db36f0325488895f`.

Both interpreters currently have NumPy 2.5.2, Astropy 8.0.1 and Pydantic 2.13.5.
v2 has netCDF4 1.7.3; v3 has netCDF4 1.7.4, xarray 2026.7.0 and Dask 2026.8.0.
This compares v2 **code running now**, not historical package versions or
historical production timings.

### Reference coverage

All input files were read through `toltec_astro_dev/run/data_lmt`; none was
modified. The JSON evidence records paths, file sizes and SHA-256 fingerprints.

| Reference | Samples | Readout channels | Recorded blocks | Frequency steps/block |
|---|---:|---:|---:|---:|
| `145605`, VNA | 4,910 | 1,000 | 1 | 491 |
| `145606`, target | 1,770 | 668 | 1 | 177 |
| `145609/0001`, tune | 3,540 | 632 | 2 | 177 |
| `145609/0002`, timestream | 7,802 | 632 | 1 | — |
| Telescope `145605/0001` | 1,432 | — | — | — |
| Telescope `145606/0001` | 715 | — | — | — |
| Telescope `145609/0001` | 2,837 | — | — | — |
| Telescope `145609/0002` | 3,427 | — | — | — |

There are **no legacy processed-sweep, D21, solved-timestream, tel_toltec2 or
wyatt files in this fixture set**. Those paths are inventoried below from code,
not claimed to have passed numerical parity. Corrupt, incomplete and differently
encoded files also need separate parity cases.

### Measurement method

[`probes/reader_comparison.py`](probes/reader_comparison.py) runs separately in
each environment. Evidence:

- [`reader-comparison-v2.json`](probes/reader-comparison-v2.json)
- [`reader-comparison-v3.json`](probes/reader-comparison-v3.json)

Tracing records native-variable reads through v2's mapper and reads through
xarray's netCDF4 backend. Proxy wrapping is idempotent to avoid double-counting
the same native access. A read's byte count is the **logical returned payload**,
not physical disk bytes, HDF5 cache misses, network traffic or peak resident
memory. File/header inspection itself is not counted as array payload I/O.

Trace phases run sequentially on one open reader so cache effects are visible.
Timing trials are separate, uninstrumented, fresh-reader runs: one warm-up,
three measured trials, median reported. Imports are warm; OS cache is not
flushed. Environments run sequentially. These are local diagnostic timings,
not a controlled benchmark or a forecast for object storage.

v3 is tested with `engine="netcdf4"`, ordinary `chunks=None`, and separately
with `chunks={}` for Dask read traces. Its public open helpers are used.

## 2. Shared reading contract and lifetime

| Operation | v2 behavior | Current v3 behavior / consequence |
|---|---|---|
| Open KIDs | Opens native file and eagerly loads applicable metadata by default; filename identification can avoid a kind-header read | Opens xarray Dataset; first accessor/mapper construction is structural; `.meta` triggers metadata reads |
| Open telescope | Construction validates TelTime presence with a native file open; `.open()` opens again and populates metadata | Open helper opens Dataset; first `.meta` reads headers/endpoints; no equivalent content check at helper return |
| Locate a field | Native variables or dimensions; `get_value` checks variable first; no general attribute namespace | Mapping records name and `FieldLocation`; native variable, attribute and dimension are explicit alternatives |
| Name collision | A variable can implicitly win over a dimension | Unqualified name in multiple locations raises ambiguity; explicit `.var()`/`.dim()` fixes location. `n_sweepsteps` has an explicit ordered var → attr → dimension list |
| Native array handle | `get_var()` returns netCDF4 Variable; slicing reads | `get_arr()` returns DataArray without reading payload; slice before `.load()`/`.to_numpy()` to limit access |
| Scalar | Native scalar variable `.item()`; string handled separately | Scalar var, attr, or dimension count via mapper; non-scalar numeric arrays rejected |
| Text | `get_str`: S1 character arrays; 1-D text or 2-D list of strings through `ncstr` | `get_str`: scalar/singleton strings and 1-D character arrays. It is not a replacement for decoding the 2-D model-header table |
| Defaults | Reader-specific synthesized values in several places | `get_value()` reads source; only `get_value_or_default()` permits configured absence fallback. Reader schemas here do not use eager `resolve_value=True` |
| Metadata reuse | Stored mutable KIDs dictionary / telescope model; no read on repeated `.meta` | Cached typed metadata instance per accessor; no read or validation on repeated `.meta` |
| Direct metadata builder | KIDs generally uses its stored dictionary | `mapper.get_metadata(ds)` deliberately rebuilds. With NumPy-backed caching, already read variables need not hit backend; with Dask they commonly do |
| Mutation / new source | Reader carries file state and cached tables | A cached accessor's metadata/derived values are not a live monitor of mutations. Reusing a resolved mapper does not validate source identity or rebuild its field plan |
| Close / returned data | `read()` outputs are materialized; metadata contains actual ADC/LO arrays | Lazy DataArrays still depend on the backing resource; raw root and child headers can share backing data. A frozen metadata model does not make the Dataset immutable |

xarray documents that `chunks=None` uses lazy indexing but loads NumPy arrays
when computation requires values; its data cache defaults on. Dask-backed opens
default that cache off. Coordinate-index creation can also read coordinate
payloads at open, although these eight files generated no such traced reads.
[xarray.open_dataset](https://docs.xarray.dev/en/stable/generated/xarray.open_dataset.html)

## 3. KIDs: complete raw header/dimension inventory

In the tables, `HT.X` means `Header.Toltec.X`, `DT.X` means `Data.Toltec.X`,
`HK.X` means `Header.Kids.X`, and `DK.X` means `Data.Kids.X`. Each comma-separated
name below is an individual field, not an unspecified family.

For applicable v2 `meta` mappings, fields are read eagerly and generally
required. In v3, “optional” means no mapping leaves the typed field `None`;
invalid present values can still raise. Counts use dimension sizes, not
coordinate values, and do not read array payloads.

| v2 field → v3 field | Physical source | Behavior, computation and completeness |
|---|---|---|
| `obs_type` / identification | `HT.ObsType` | v2 may identify from filename without reading it; v3 reads it for kind and optional metadata. Codes 0/1 timestream, 2 VNA, 3 target, 4 tune |
| `kind_str` / identification | `HK.kind` | Both recognize `d21`, `processed_sweep`, `processed_timestream`, `SolvedTimeStream`. v3 also accepts `kind_str` alias and prefers it over raw ObsType |
| `fsmp` → `f_smp` | `HT.SampleFreq` | v2 plain scalar interpreted as Hz by consumers; v3 required frequency Quantity, converts declared frequency unit to Hz, rejects nonpositive rate |
| `f_lo_center` | `HT.LoCenterFreq` | v2 eagerly reads for all KIDs kinds. v3 sweep metadata optional Hz float; **not included in timestream metadata**. Available separately through mapper/frequency access |
| `atten_drive` | `HT.DriveAtten` | Same scalar values; optional float in v3; no conversion of attenuation units |
| `atten_sense` | `HT.SenseAtten` | Same as drive |
| `roach` | `HT.RoachIndex` | Required scalar; v3 uses canonical interface/array lookup |
| `obsnum` | `HT.ObsNum` | Header integer retained; v3 required typed integer |
| `subobsnum` | `HT.SubObsNum` | Same |
| `scannum` | `HT.ScanNum` | Same |
| `cal_roach` | `HT.RoachIndex` again | Both request this same header twice at the Python layer; v3 xarray cache normally removes second backend read, Dask does not |
| `cal_obsnum` | `HT.TargSweepObsNum` | Same value, optional in v3 |
| `cal_subobsnum` | `HT.TargSweepSubObsNum` | Same value, optional in v3 |
| `cal_scannum` | `HT.TargSweepScanNum` | Same value, optional in v3 |
| `adc_snap` | `HT.AdcSnapData` | v2 materializes full `(2,4096)` int16 array at open, 16,384 bytes. v3 metadata stores **the physical variable-name string**, not data; mapper native array access remains possible |
| `n_kids_design` | dimension `loclen` | Same count; optional in v3. Distinct from actual readout channels |
| `n_chans_max` | `HT.MaxNumTones` | Same count; optional in v3. Not a substitute for `iqlen` |
| `filename_orig` | `HT.Filename` | Both decode fixed-width text; optional in v3. Provenance value, not a promise of a currently accessible path |
| `mastervar` → `mastervar` and `master` | `HT.Master` | v2 raw integer plus integer master/string master_name; v3 raw code represented as string, canonical string-enum master. v3 requires source value |
| `repeatvar`, derived `repeat` | `HT.RepeatLevel` | v2 raw integer and duplicate integer `repeat`; v3 optional string `repeatvar`, **no `repeat` field** |
| `n_times` | raw `time`, solved `ntimes` | v2 kind-specific dimension mapping; v3 ordered `time`/`ntimes` dimension mapping. Reduced sweep without time uses step count |
| `n_chans` | raw `iqlen`, reduced `n_chans` | v3 candidates `iqlen`, `n_chans`, `chan`, `channel`; no fallback to design count `loclen` |
| `n_sweepreps` | `HT.NumSamplesPerSweepStep` | Same header; optional in v3. Not necessarily the number actually acquired at every frequency |
| `n_sweepsteps` | raw `HT.NumSweepSteps`; reduced `nsweeps` dimension | v3 explicitly tries header var, header attr, then `sweep`, `nsweeps`, `sweeps` dimensions. Header wins. Reduction's actual unique-frequency count can differ |
| `n_sweeps_max` → `n_blocks_max` | dimension `numSweeps` | Allocated capacity, not acquired count. v3 no longer exposes duplicate `n_sweeps_max`; absent capacity falls back to actual count |

Additional v3 raw schema fields:

| Field | Source | Behavior / cost |
|---|---|---|
| `obs_start_time` → `t0` | `HT.ObsStartTime` | Required scalar Unix time → Astropy Time; not read by v2 KIDs metadata |
| `obs_end_time` → `t1` | `HT.ObsEndTime` | Same; preserves header time independently from sample-count integration |
| `n_blocks` | dimension `block` | Explicit block-indexed data only; not `numSweeps` allocation |
| `atten_in` | `HT.AttenIn` or `atten_in` | Declared mapping only; not consumed by `get_metadata()` or a public attenuation property |
| `atten_out` | `HT.AttenOut` or `atten_out` | Same; does not replace drive/sense interpretation |
| raw `I`, `Q` aliases | `I`, `Q` | Accepted alongside physical DT names; legacy `DK.Is/Qs` are **not** in this raw schema |

### Kind/master verification and unknown cases

Read-only inspection of `toltecdb_slice.sqlite` confirms master rows 0 TCS,
1 ICS, 2 CLIP, **3 REPEAT**, and ObsType rows 0 Nominal, 1 Timestream, 2 VNA,
3 TARG, 4 TUNE. v3 maps 0–2 masters and all five ObsTypes correctly. All four
raw reference files are TCS; they do not exercise other masters. Code 3 is not
supported by the v3 master enum; its interpretation must not be guessed from
an unrelated header convention. The examined v2 master enum also supports only
0–2, so this is an existing coverage gap, not a newly introduced code-3 regression.

v3 structural inference (no kind header but sweep/time dimensions present) is
more permissive than v2 identification. An unrecognized kind string returns
Unknown rather than necessarily rejecting the file, and metadata can proceed
down the non-sweep branch. Conversely, required v3 acquisition-time/master
headers can reject files v2 accepted. Filename/header conflict checks live in
`kids.source.identify_source`, not automatically in `open_toltec().toltec_kids`.

### Derived metadata and metadata moved out of the reader

| v2 output | Current v3 counterpart / difference |
|---|---|
| `instru="toltec"`, `interface="toltecN"`, `roach`, `nw` | Same values on fixtures; adds `array_name` from canonical lookup |
| `instru_component="roach"` | v3 inherited default is **None** in the raw KIDs metadata |
| `master`/`master_name` | Canonical `master="tcs"/"ics"/"clip"`; no duplicate `master_name` |
| `source`, `filepath`, `file_timestamp`, `file_suffix`, `file_ext` | Not members of the v3 raw metadata dataclass. Discovery has a separate SourceInfoModel (`timestamp` rather than `file_timestamp`) |
| `uid_obs`, `uid_raw_obs`, `uid_raw_obs_file` | No equivalent properties in the current raw metadata or SourceInfoModel; not silently interchangeable with ORM identity |
| filename timestamp | v2 timezone-aware UTC datetime. Current discovery parser produces a naive datetime; raw `t0` instead uses header Unix time. These are distinct timestamps |
| `data_kind` in metadata | v3 separate accessor property / discovery field, not raw metadata member |
| `n_timespersweep`, `n_timesperblock` | No stored equivalents; nominal product `n_sweepsteps*n_sweepreps` remains derivable |
| `f_los`, `sweep_slices`, `block_slices` | v2 stores LO values and detected slices in metadata. v3 metadata does not expose these; tune detection and reducer each derive boundaries |
| `n_sweeps`, `n_blocks` | Flat tune: both report actual reset count. Explicit-block v3 metadata returns a list by default; selected metadata has `n_sweeps=1` but retains total `n_blocks` |
| default block | v2 axis/model/read default is **last acquired block**; explicit negative indices supported. v3 raw metadata default is whole acquisition; mapper block selection rejects negative indices. Pipeline selection is a separate layer |
| new `t_exp` | `n_times/f_smp`, seconds; **not** `t1-t0`. Target fixture: 3.62496 s sampled integration versus 9 s header span |

Flat VNA/target metadata no longer scans or validates LO samples. Flat tune
does: positive, finite, one-dimensional LO; decreases split blocks. v2 scanned
all raw sweeps, rejected nonpositive/NaN LO, rejected multiple blocks for
non-tune kinds, and warned about incomplete tune blocks. v3's reducer separately
detects resets for any sweep kind and does not preserve that warning/rejection
policy. This is a validation-timing/behavior difference, not merely a cache change.

## 4. KIDs: axes, model parameters and native data

| Physical field / v2 output | v2 access/computation | Current v3 access/computation |
|---|---|---|
| `HT.ToneFreq` / `f_tone` | Reads actual block rows; QTable one row per channel; Hz | `.f_tone` returns lazy native `(numSweeps,toneFreqLen)` DataArray; preserves allocated block axis |
| derived `f_chan` | Per selected block, tone + scalar LO center, Hz | Cached `.f_chan` adds center to entire tone header. NumPy-backed input loads tone data on first arithmetic; Dask tone arithmetic stays lazy, but scalar center is read eagerly |
| `HT.ToneMask` / `mask_tone` | Reads blocks, **casts bool**; missing mask → all True | `.tone_mask` native dtype, or None; fixture dtype is float64. No cast/default; table column renamed `tone_mask` |
| `HT.ToneAmp` / `amp_tone` | With mask present, requires modern amp field | `.tone_amp` ordered modern/deprecated field, or None; no synthetic ones; table name `tone_amp` |
| `HT.ToneAmps` / deprecated amplitude | Used only when mask absent; otherwise legacy field ignored; absent → ones | Fallback candidate regardless of mask; different behavior on partly populated old headers |
| `HT.TonePhase` / `phase_tone` | With mask present requires phase, radians; absent mask → zeros | `.tone_phase` lazy native array, or None; table name `tone_phase`; no generated zeros |
| `DT.LoFreq` / `f_los`, axis `f_lo` | Full LO read at raw sweep open; retained Quantity; per-block unique values later | `.f_lo` native DataArray; metadata reads only for tune; reducer reads full vector and block vectors |
| raw sweep table: `id`, `f_lo`, `f_sweep`, `n_samples`, `sample_start`, `sample_end` | Sorted unique LO + first indices/counts; absolute sample offsets; cached tables for every block | Raw `get_sweep_axis_data()` fails on all 3 reference sweeps: expects a sweep dimension/frequency coordinate instead of raw time/LO. Its implemented table has only `sweep_id`, `f_sweep`, `f_lo` |
| reduced coordinate `sweeps` | Literal native variable read, interpreted as sweep offset; one sample/step | v3 helper looks at `sweep`/`nsweeps` dimensions or `frequency` coordinate, and uses a 10 MHz magnitude heuristic for relative vs absolute frequency. Not a demonstrated equivalent |
| `HT.ModelParamsHeader` | Reads 2-D character table into column names | No schema field/table decoder in current TolTEC KIDs mapper |
| `HT.ModelParams` | Reads each actual block, transposes logically into per-channel QTable columns; cached; last block default | Native Dataset still exposes it, but **no corresponding typed/table reader**. Copying it into reduced output is not equivalent to interpreting it |
| `DT.Is`, `DT.Qs` | Requested sample/channel slice, then transpose to channel-first | Native arrays retain disk `(time,iqlen)` layout; explicit mapper slicing is lazy. Reducer produces channel-first means/stds |

The reference model tables have columns `f_centered`, `f_out`, `f_in`, `flag`,
`fr`, `Qr`, `normI`, `normQ`, `fp`, `slopeI`, `slopeQ`, `interceptI`,
`interceptQ`, `Qc`, `A`. They exist in target, tune and timestream files.
The VNA fixture lacks the model header; v2's explicit model-table request
raises KeyError there. These columns must not be inferred from array positions
without decoding the stored header.

### Confirmed channel-table and selection defects

1. `get_chan_axis_data()` assigns an unsqueezed `(1,C)` header column to a
   QTable already containing C channel rows. Astropy broadcasts it to `(C,C)`.
   Each of the five tone/frequency columns is oversized. VNA's table occupies
   **40,008,000 bytes of column data**, rather than v2's 41,000 bytes; target
   occupies **17,854,304 bytes**, rather than 27,388. These are actual column
   buffer sizes, not peak-process memory estimates.
2. For tune `(2,632)` does not broadcast to 632 rows; the table raises
   `ValueError: Inconsistent data column lengths`.
3. Tables are rebuilt each call. Even when xarray prevents further backend I/O,
   the oversized allocations/computation repeat. With Dask, table construction
   also rereads headers and reconstructs metadata.
4. `select_channels([0,1])` uses `I.dims[0]`; on every raw fixture this is
   **time**, not iqlen. For target the result is `(time=2,iqlen=668)`, not
   `(time=1770,iqlen=2)`. String handling also lacks v2's full table-expression
   selection behavior. Native named `.isel()` works, but does not excuse the
   incorrect convenience method.

### Frequency semantics still have two paths

Typed sweep metadata normalizes a declared LO-center unit to Hz. The separate
`_get_f_lo_center()` used for axes instead reads raw numbers, permits averaging
a multi-element center header, or computes mean LO if the center is absent.
The reducer has a third fallback: median unique LO. Tone arithmetic also uses
numeric values without normalizing their declared units. Reference files use
Hz so parity passes; mixed units/missing center headers are not covered by that
result. This should be unified before promising unit-general reader behavior.

## 5. KIDs: scientific computations, selection and processed files

### Sweep computation comparison

v2's `.read()` is not a raw-array getter: for raw sweeps it reduces samples,
constructs complex S21 and complex standard-deviation uncertainty, and carries
the selected raw I/Q arrays in result metadata. v3 puts reduction in the
standalone `SweepReducer`, with separate I/Q and uncertainty DataArrays and
computed views. That separation is appropriate; numerical equivalence remains
the requirement.

| Operation | v2 | v3 | Finding |
|---|---|---|---|
| Block detection | Cached LO decreases; rejects non-tune multi-block raw files | Fresh LO decrease detection in reducer; independent of metadata detection | Same boundaries on fixtures; validation policies differ |
| Group samples | `np.unique` first index/count, contiguous slices | `np.unique` inverse indices/count, Boolean mask for every frequency | Same on monotonic reference blocks |
| Means | NumPy mean, float64 output | NumPy mean, float64 output | Exact persisted equality in four parity cases |
| Uncertainty | Population std, `ddof=0`, not standard error | Same by default; can disable uncertainty | Exact persisted equality with default config |
| Output axes | `(channel, sweep)` per selected block | `(channel, sweep)`, or `(block, channel, sweep)` | Intentional organization change |
| Channel frequency | Selected block's tone + LO center | Reduced view selects/alines block tone headers and channel dimension | Exact equality in tested blocks |
| Full frequency grid | Channel frequency + sweep offset | Broadcast sum, transpose to I/Q dimensions | Exact equality; allocation occurs when requested, not metadata-only |
| I/Q units | Bare inputs yield dimensionless values; S21 constructed | Retains input `units` attrs; bare remains bare | Exact tested units; no artificial ADU assignment for bare I/Q |
| Channel/sweep/sample/time selection | Reader slicer supports axis/table queries before reading | Named xarray selection available; no equivalent complete reader slicer; convenience channel method wrong | Must preserve ability to limit reads, not necessarily old syntax |
| Tune default work | Reads last block unless requested otherwise | Standalone reducer processes **all blocks** | A default-call timing comparison is not equivalent work |
| Repeat reduction | Reuses metadata/axis tables but reads and computes data again | Passing a previously reduced tree with matching config returns it; reusing only raw Dataset computes again | Configuration marker does not verify that raw data was not mutated |

The v3 loop performs an `N`-element equality comparison for each of `S` sweep
frequencies: O(SN) mask work in addition to O(NC) statistics. Boolean selection
copies group samples; v2's contiguous slices can be views. v3 may nevertheless
be faster here due to surrounding operations and array layout; the local
timings below do not isolate the cause. Avoid replacing this with an assumed
optimization before checking exact means/std behavior, memory order and missing
samples.

Both implementations use ordinary mean/std rather than nanmean/nanstd. Missing
or packed data need explicit tests: native netCDF4 can return masked arrays,
whereas xarray's CF decoding represents fill values as missing values. NumPy
masked reductions and reductions over NaNs are not an established parity
contract. Other untested cases include incomplete/unequal blocks, nonfinite LO,
unit conversions, coordinate collisions and alternate dimension order.

### Extra materialization in multi-block output

The reducer copies **every non-time data variable** into each reduced block;
it then concatenates the blocks with default data-variable selection. For the
ordinary non-Dask tune open, the trace includes model parameters, ADC snapshot,
tone mask/amp/phase, design positions and otherwise unused headers. In the
traced sequence, 40 backend reads return 18,136,440 bytes during reduction/view
extraction; I/Q account for 17,898,240 bytes.

The other 18 unique names in that phase are `HT.CompileTime`, `HT.HwpInstalled`,
`HT.RoachKatcpMacAddr`, `HT.SynthSerialNum`, `HT.AttenSerialNum`,
`HT.NumKidsActual`, `HT.SelectedMask`, `HT.FpgaFreq`, `HT.AccumLen`,
`HT.Xloc`, `HT.Yloc`, `HT.ToneAmp`, `HT.ToneMask`, `HT.TonePhase`,
`HT.ToneAmpsMode`, `HT.ModelParamsHeader`, `HT.ModelParams` and
`HT.AdcSnapData`. Each is requested twice. This list describes the measured
sequence after metadata/frequency access; starting from a fresh Dataset also
needs the scalar headers not already cached. The implementation's copy rule
includes arbitrary additional non-time variables, not just this finite list.

For example, output ADC snapshot becomes `(block=2,adcSnapDim=2,4096)` and is
in memory: an observation-wide array was duplicated across blocks. Xloc and
scalar headers are broadcast too. This is not necessary to compute sweep
means/stds. With Dask, large ancillary arrays can stay lazy, but scalar headers
and requested frequency computations still trigger reads; exporting the whole
child would subsequently compute its ancillary graph.

This broadcast also breaks subsequent typed metadata reading on the unselected
tune child: scalar header mappings see block-valued headers.
`child.toltec_kids.mapper.get_metadata(child, block=0)` still fails because kind
and common headers are read before block selection. The current error is
`Field 'Header.Toltec.ObsType' is not a scalar (ndim=0)`; the error's ndim text is
itself misleading. Raw-root metadata and numerical reduced views can work while
reduced-child metadata fails.

xarray's concatenation options control which variables are concatenated and
how other values are compared. That is the relevant implementation boundary,
not an argument for copying more raw data or caching it globally.
[xarray.concat](https://docs.xarray.dev/en/stable/generated/xarray.concat.html)

### Complete legacy processed-field inventory

These are actual v2 reader fields, even though no corresponding processed file
is present in this reference set. Namespaced v3 analysis outputs are **not**
automatically aliases for these legacy storage names.

| Physical field | v2 behavior | Current v3 status |
|---|---|---|
| `DK.Is`, `DK.Qs` | Reduced-sweep arrays, channel-sliced/transposed; no raw-step averaging | Raw schema does not map them; reduced sweep schema maps its own namespaced I/Q |
| `DK.rs`, `DK.xs` | Solved timestream arrays → MultiTimeStream r/x | No direct legacy mapping in raw schema or namespaced reduced view |
| `DK.d21_fs` | Unified D21 frequency, Hz; whole array | No direct legacy mapping |
| `DK.d21_adiqs` | Unified D21, interpreted ADU/Hz; whole array | No direct legacy mapping; separate D21Analysis is computation, not file decoding |
| `DK.d21_adiqscov` | Covariance attached to unified sweep | No direct legacy mapping |
| `DK.d21_adiqsmean` | Mean, ADU/Hz | No direct legacy mapping |
| `DK.d21_adiqsstd` | Std, ADU/Hz | No direct legacy mapping |
| `HK.candidates` | Reduced-VNA candidates, Hz | No direct legacy mapping |
| `HK.PsdFreq` | PSD frequency vector read whole | No direct legacy mapping to namespaced `f_psd` |
| `DK.ispsd` | Channel-sliced/transposed I PSD → metadata | No direct legacy mapping |
| `DK.qspsd` | Channel-sliced/transposed Q PSD → metadata | No direct legacy mapping |
| `DK.phspsd` | Channel-sliced/transposed phase PSD → metadata | No equivalent field in current TolTEC timestream schema |
| `DK.rspsd` | Channel-sliced/transposed r PSD → metadata | Namespaced r PSD exists, not a legacy-reader mapping |
| `DK.xspsd` | Channel-sliced/transposed x PSD → metadata | Same for x |

v2 checks existence before reading diagnostic extras, but its unified-D21 object
maker subsequently assumes other D21/candidate fields exist. Its reduced-sweep
axis reader directly requests the literal `sweeps` variable. Those legacy paths
need real fixtures rather than being treated as universally robust baselines.

Raw timestream `.read()` in v2 also constructs sample index/f_smp and channel
frequencies; current v3 native reading does not generate an equivalent time
coordinate automatically. v2's index allocation uses whole-file n_times even
for selected reads, which warrants a sliced-read regression case rather than
blind reproduction.

`TimestreamReducer` is new PSD computation (SciPy Welch, optional summaries),
not equivalent to v2 merely reading stored PSDs. It currently requires literal
`I`/`Q`; applying it directly to the reference raw timestream raises
`Dataset must contain 'I' and 'Q' data variables`. Thus raw-array parity does
not establish an end-to-end timestream processing path.

Generic `S21View` can construct `I + 1j*Q` after fields are mapped. Its computed
S21/amplitude properties are not result-cached: NumPy arithmetic allocates on
each access; Dask builds expressions whose later evaluation does the work.
That is different from v2 storing a constructed complex S21 in MultiSweep.

## 6. Telescope: every v2 field and current additions

`HD.X` means `Header.Dcs.X`, `HS.X` means `Header.Source.X`, and `TB.X` means
`Data.TelescopeBackend.X`.

| v2 field/output | Physical source | v2 behavior | Current v3 behavior / consequence |
|---|---|---|---|
| `mapping_type` | `HD.ObsPgm` | Required text read at open; e.g. `TargSweep` | **Not mapped or retained**; ObsGoal is not a substitute for observing program |
| `obs_goal` | `HD.ObsGoal` | Optional, lowercase or None; post-open direct `__dict__` update bypasses model validation | Optional typed enum; empty/missing → unspecified; invalid string raises |
| `target_ra`, `target_dec` → `target`, `target_off` | `HS.Ra`, `HS.Dec` | Requires two values per field, radians → ICRS SkyCoord | Both absent allowed; otherwise paired matching 1-D arrays of length 1 or 2; unit-aware, rad fallback. Main/reference values exactly match fixtures |
| `target_az`, `target_alt` | `HS.Az`, `HS.El` | When az exists, reads both pairs but **discards them** | Not mapped/read. This removes unused reads, not an exported target-coordinate feature |
| `mastervar` | `Header.Toltec.Master` | Declared in mapper but **not read by telescope metadata updater** | Not mapped; master fixed canonical `tcs` |
| `repeatvar` | `Header.Toltec.RepeatLevel` | Also declared but unused | Not mapped |
| `t0` | `TB.TelTime[0]` | One sample → Time Unix, display format switched to isot | Same instant; keeps Time object in default Unix representation |
| new `t1` | `TB.TelTime[-1]` | Not read | Reads one last sample. Requires nonempty 1-D time array |
| data `time` | `TB.TelTime[:]` | Full read, subtract first, Quantity seconds | Lazy native **absolute Unix seconds**, no subtraction. Exact elapsed equivalence when explicitly normalized |
| data `az` | `TB.TelAzAct` | Full array → Quantity rad | Native lazy DataArray, stored attrs/values; exact numerical match |
| data `alt` | `TB.TelElAct` | Same | Same |
| data `ra` | `TB.SourceRaAct` | Same | Same |
| data `dec` | `TB.SourceDecAct` | Same | Same |
| data `holdflag` | `TB.Hold` | Full array then `astype(int)` | Native dtype retained: **float64 in all fixtures**, despite current docstring saying integer representation. Values match 0/1, dtype differs |
| data `target`, `t0`, `ref_frame="icrs"` | Derived from metadata / constant | Bundled in LmtTelData | Targets/t0 in typed metadata; target frame remains ICRS, no separate bundled ref_frame field |
| data `meta` | Model dump of reader metadata | Copies declared model output; dynamically inserted keys such as mapping_type are not necessarily part of model serialization | Frozen typed metadata; explicit fields only |
| `obsnum` | filename → now `HD.ObsNum` | Filename integer | Required header integer. Fixtures agree; conflicts not checked in this accessor |
| `subobsnum` | filename → now `HD.SubObsNum` | Same | Same |
| `scannum` | filename → now `HD.ScanNum` | Same | Same |
| `interface` | filename / header filename or receiver | Filename parsed, then overwritten with `"lmt"` at open | Header `Header.File.Name` or source filename → `tel_toltec`/`tel_toltec2`; otherwise `HD.Receiver` fallback. Arbitrary prefixes/wyatt need policy |
| `instru`, `instru_component` | constants | `lmt`, **tcs** | `lmt`, **tel** |
| `master`, `master_name` | constants | integer 0 plus `"tcs"` inserted dynamically | Canonical `master="tcs"`; no duplicate name |
| `source`, `file_timestamp`, `file_suffix`, `file_ext` | filename / source | Reader metadata; timestamp UTC midnight from date-only filename | Not raw metadata members; separate discovery. Current telescope discovery pattern does not accept all optional suffixes accepted by v2 |
| new `t_exp` | `HD.IntegrationTime` | Not read | Required scalar Quantity seconds, including zero. Three sweep-associated tel files have 0; timestream tel has 60.320000087609515 s |
| new `az_des`, `alt_des` | `TB.TelAzDes`, `TB.TelElDes` | Not mapped | Native variable mappings exist; access through mapper, no convenience accessor properties |
| new `az_map`, `alt_map` | `TB.TelAzMap`, `TB.TelElMap` | Not mapped | Same |

Stored reference angle attrs are `units="rad"`, time is `units="sec"`, hold
is `units="boolean"`. V3 preserves these rather than assigning Astropy units
to every return value. Hold is categorical metadata, not a physical quantity.
This needs an explicit type/units contract before downstream users assume
`.u.quantity` applies to every column. Target/header angular conversion is
implemented; the pointing properties themselves do not normalize other units.

All four pointing arrays and hold arrays match after numeric dtype
normalization. For all four files, `v3.time - v3.time[0]` exactly matches the
v2 elapsed vector. No interpolation, pointing correction, synchronization or
coordinate-frame transformation was compared or introduced.

## 7. Measured materialization and performance

### First complete metadata access

These counts include v2's eager metadata during open, and v3's first `.meta`
after its payload-lazy open. They exclude later axis-table requests.

| Input | v2 backend calls / bytes | v3 backend calls / bytes | Main reason |
|---|---:|---:|---|
| KIDs VNA | 20 / 36,620 | 20 / 604 | No ADC or LO-vector metadata materialization in v3 |
| KIDs target | 20 / 24,060 | 20 / 604 | Same |
| KIDs tune | 20 / 31,140 | 21 / 14,764 | Both inspect LO; v3 excludes ADC, adds acquisition times |
| KIDs timestream | 17 / 16,972 | 17 / 588 | No ADC; v3 metadata omits LO center, adds times/kind |
| Each telescope file | 7 / 328 | 10 / 324 | v2 program + unused Az/El pairs; v3 adds identifiers, filename, integration time and second endpoint |

Repeated accessor `.meta` issues **zero backend reads in both implementations**.
These small logical reads do not imply small actual I/O on remote/chunked
storage. They also do not measure Python model/schema construction cost.

### Repetition, Dask and hidden work

| Action | Observed consequence |
|---|---|
| v3 KIDs `mapper.get_metadata(ds)` again, default NumPy cache | No further backend reads after `.meta`, but repeated validation, Time/Quantity construction and tune LO boundary computation |
| Same, Dask target | 23 backend calls / 616 bytes **each call**; ObsType read 3 times and RoachIndex twice within one build |
| Same, Dask tune | 24 calls / 14,776 bytes each; rereads LO and recomputes boundaries |
| v3 tel direct metadata builder again | Two endpoint reads / 16 bytes even with default cache; slicing the time array did not populate a reusable full-array cache |
| v3 first raw `.f_chan`, default open after metadata | Tone header materializes: VNA 8,000 B; target 5,344 B; tune 10,112 B. Timestream also needs uncached center scalar |
| v3 first raw `.f_chan`, Dask | Tone stays lazy, but center scalar is fetched **twice** (two `.values` evaluations) |
| v3 repeated successful channel table, Dask target | 29 backend calls / 22,008 bytes plus oversized allocations; includes full metadata reconstruction |
| v3 tune reduction with Dask | Still eager for I/Q and LO due to `.values`; ancillary non-scalar headers may stay lazy. Dask does not make this reducer an out-of-core algorithm |
| v3 repeated reduced `f_chan`/`frequency` property | New view-level arithmetic/dimension alignment; no final frequency-grid cache. Dask can reevaluate graph leaves when results are requested separately |

The concern is **repeated computation of the same known structure or small
headers**, not memoization keyed by floating-point data. The most direct
improvements would reuse one local kind/center read per operation, avoid
building full metadata just to label a table, and avoid computing boundaries
again when an operation already has them. Global source caches are unnecessary.

### Local timings (milliseconds, medians of three fresh-reader trials)

| Input | v2 open + metadata | v3 open + metadata | v2 open + metadata + numerical outputs | v3 open + metadata + numerical outputs |
|---|---:|---:|---:|---:|
| KIDs VNA | 9.30 | 4.48 | 151.15 | 81.66 |
| KIDs target | 6.46 | 4.34 | 63.83 | 35.25 |
| KIDs tune, **both blocks** | 7.70 | 5.92 | 112.94 | 67.86 |
| KIDs timestream, full I/Q | 5.50 | 4.57 | 42.53 | 17.74 |
| Telescope 145605/0001 | 4.22 | 12.41 | 5.90 | 13.18 |
| Telescope 145606/0001 | 3.91 | 12.68 | 5.05 | 13.34 |
| Telescope 145609/0001 | 4.27 | 12.12 | 5.78 | 13.09 |
| Telescope 145609/0002 | 4.36 | 13.88 | 6.13 | 15.03 |

Sweep output timing includes means, standard deviations and channel/sweep/full
frequency arrays. v2 also constructs its normal channel tables, complex object
and attached metadata. Telescope timing reads all six arrays and includes an
explicit elapsed-time calculation for v3. Timestream v2 constructs a
MultiTimeStream and channel metadata, whereas v3 reads I/Q into arrays. These
are actual API-workflow costs, **not isolated equivalent kernels**. They exclude
CLI startup, Zarr/Parquet writes, database ingestion and web visualization.

The fact that v3 KIDs is faster in these trials does not cancel its channel-table
memory regression or broadcast-header reads. The telescope slowdown is a
reason to profile Dataset/open/mapper/model overhead before optimizing scalar
arithmetic—not a reason to abandon lazy xarray access.

## 8. Focused follow-up, in dependency order

| Priority | Work | Why needed / acceptance condition |
|---|---|---|
| First | Correct channel table shape/block handling and channel selection | Real fixture failures. One row per readout channel, correct selected time/channel dimensions, no O(C²) broadcast, explicit block behavior |
| First | Provide the raw sweep-axis information using existing LO/block semantics | All raw fixture helper calls fail; preserve actual counts/start/end offsets needed for bounded reads and diagnostics |
| First | Limit reduced-child variables/broadcasting and make its metadata readable | Avoid unrequested ADC/model loading and duplicate observation-wide headers; both raw-root and selected reduced-view metadata must work |
| Next | Add model-parameter header/table access | Present in three reference files and used by v2; named columns/block handling are missing |
| Next | Resolve metadata contract gaps explicitly | Decide observing program, roach component, repeat type, LO center for timestream, provenance separation and timestamp timezone; do not equate integration with elapsed duration |
| Next | Unify frequency/units and time/hold semantics | Remove inconsistent center fallbacks; specify absolute vs elapsed time and raw versus normalized flags/units |
| Next | Remove verified repeated small reads/computation | Kind/roach/center reads, table metadata rebuild, repeated block work; maintain current accessor-local lifecycle |
| Before claiming full reader parity | Add missing-data/partial-block/alternate-layout and processed-file fixtures | Current numerical pass covers ordinary raw files, not masked input, legacy diagnostics or solved data |
| Separate phase | Connect timestream processing and file/ORM catalog | Raw read parity is not a functioning PSD/solve pipeline; catalog must consume a settled metadata contract |

No recommendation here requires a new universal cache, compatibility wrapper
hierarchy or replacement for the established schema/mapper/accessor/view design.

## 9. Reproduce

From the Python workspace root:

```sh
/Users/ma/.pyenv/versions/tolteca_v2/bin/python \
  tolteca/design/probes/reader_comparison.py v2 \
  toltec_astro_dev/run/data_lmt /tmp/reader-comparison-v2.json

toltec_astro_dev/.venv/bin/python \
  tolteca/design/probes/reader_comparison.py v3 \
  toltec_astro_dev/run/data_lmt /tmp/reader-comparison-v3.json
```

The trace captures expected failed v3 helper calls in its `error` fields rather
than aborting. It does not mutate input Datasets beyond normal reader caches.
Numeric hashes normalize numeric dtype for comparison, while shapes, original
dtypes, units and mask counts are retained separately. Hash agreement alone
does not establish layout or dtype equivalence.

From `tolteca/`, the existing independent v2 → persisted-v3 parity test:

```sh
TOLTECA_V2_PYTHON=/Users/ma/.pyenv/versions/tolteca_v2/bin/python \
  ../toltec_astro_dev/.venv/bin/python -m pytest \
  tests/tolteca_kids/test_v2_parity.py -q
```

Result on this snapshot: **4 passed**. Reader metadata/representation tests are
in `tests/tolteca_datamodels/toltec/test_metadata_reading.py` and
`tests/tolteca_datamodels/lmt/test_tel.py`: **32 passed** in this audit. These
complement but do not replace
the missing full-reader parity cases described above.

## 10. Reader correction iterations

### Policy and scope

The fixed acquisition structure in `tolteca_ref_data` is the supported input
specification for this work. Preserve v2 field values and established reading
behavior. Do not turn required fields into optional fields or substitute
plausible values for missing headers. A behavior change needs an explicit
reason and review; passing reduction tests alone does not authorize it.

Numeric acquisition codes remain numeric. Human-readable names are separate
properties, not replacements for the stored codes. Do not equate a readout
code with a derived analysis classification merely because their names overlap.

| Iteration | Scope | Status |
|---|---|---|
| T1, 2026-09-18 | Telescope metadata and six v2 pointing/time/flag outputs | Scientific outputs retained; metadata decisions superseded by T3 |
| T2, 2026-09-19 | Restore FileLoc source identity and repair serialization/URL handling | Tollan repair approved and committed as `de75e82`; FileLoc opening/discovery retained, metadata source ownership superseded by T3 |
| C1, 2026-09-20 | Pandas catalog, parsed original filename, Arrow struct and dotted selection | Implemented for review; no change to telescope reader field values or outstanding metadata decisions |
| C2, 2026-09-20 | Original-filename fallback and correction of guessed identifiers | Implemented for review; header/original consistency checks are opt-in and off by default |
| T3, 2026-09-20 | Header-authoritative telescope metadata and parsed original filename | Implemented for review; telescope-first reference contract, no KIDs/catalog edits in this iteration |
| C3, 2026-09-20 | Shared `FilepathInfo`, `filepath_info[_orig]`, and `file_*[_orig]` aliases | Implemented for review; opening policies and scientific reading unchanged |
| C4, 2026-09-20 | Filename rules, Pydantic projections, accessor catalog factory, and user guide | Implemented for review; differences and verification recorded below |
| K1 | KIDs field requirements, raw codes, reader correctness and table costs | Not changed in T1; baseline findings remain open |
| Shared metadata/catalog | Align numeric identity contracts across the base, file catalog and ORM consumers | Follow-up; no DB/Web migration in T1 |

### T3: telescope metadata reference contract

This iteration finishes the confined telescope design before applying it to
KIDs, catalogs, or ORM consumers. It uses the existing shared `FilepathInfo`
parser without changing discovery policy. The pending KIDs/catalog work remains
unmodified and uncommitted.

#### Header evidence and approved behavior changes

All four reference telescope files contain `Header.TelescopeBackend.Master=0`.
T1 checked the unrelated `Header.Toltec.Master` name and incorrectly concluded
there was no readable telescope master. T3 reads the actual telescope header.
The LMTMC rows also record the full `/data_lmt/tel/...` path and
`Instrument="tel"`. Its `Receiver="Toltec"` is a separate receiver identity.

| Field / behavior | T3 contract | Why this changes the earlier iteration or v2 |
|---|---|---|
| `master` / `master_str` | Required nullable numeric enum / computed readable name; read `Header.TelescopeBackend.Master`. Missing header produces explicit `None`; invalid codes fail. | Preserve recorded values instead of assuming TCS. Nullable master permits the explicitly requested simulated/test-file case; it is not a model default. |
| `instru`, `instru_component` | Fixed literals `lmt`, `tcs`. | Encode the telescope model's identity; do not allow arbitrary strings through `Literal[...] \| str`. |
| `interface` | Required readout-actor string. This mapper supplies `tel`; metadata can represent another actor such as `tel2`. | User-approved distinction between instrument, component, readout actor, and filename token. Filename `tel_toltec` is not substituted for this actor. |
| Quartet observation numbers | Required `Header.Dcs.ObsNum`, `SubObsNum`, `ScanNum`. | Header authority was explicitly chosen over v2's filename-first telescope behavior. A relocated or renamed current source cannot change observation identity. |
| `obs_pgm` | Required, case-preserved `Header.Dcs.ObsPgm`. | v2's `mapping_type` was the name for this raw value, not a derived map classification. Taco's `get_obs_pgm()` consumes it as the program. No unsupported derived classification or compatibility alias is introduced. |
| `filepath_orig` | Required decoded `Header.File.Name` string. | Acquisition identity is independent of the runtime location. This header contains a path, not only a basename. |
| `filepath_info_orig` | Cached computed `FilepathInfo` parsed from `filepath_orig`, using the same parser as discovery. | Group filename components instead of duplicating flat `file_*` fields. `ut` is UTC, not mtime; telescope date-only names represent midnight on that date, not exact `t0`. Extension excludes the leading dot, as specified by the shared parser. |
| Current `source` | Not a field on `LmtTelMetadata`. Current location stays in the opened Dataset and/or the separate discovery record. | Avoid mixing relocation state with observation metadata. `FileLoc` support for discovery/local opening and the explicit remote-staging requirement remain intact. |
| Original-path consistency | `mapper.get_metadata(..., check_filepath_orig=False)` by default. Opt-in checking compares observation numbers and master where both representations contain them, raising on conflict. | Trusted headers incur no routine regex parsing/check. The checker diagnoses internal corruption and never replaces header values with guesses. It does not compare a filename date against sample start or compare readout actor with filename token. |

Other telescope fields are still required: program, goal, original path,
integration timer, both target positions, and a nonempty one-dimensional time
array. No defaults were added for them. The only model defaults are the two
instrument/component constants. Goal lowercasing, main/reference ICRS positions,
first/last Unix timestamps, native integration timer (including zero), native
pointing arrays, elapsed-time calculation, and integer flag conversion retain
the established values and computation behavior.

The frozen dataclass's `filepath_info_orig` is parsed once on explicit access or
normal serialization. It is excluded from `repr`, so diagnostic formatting
does not parse the filename. A malformed acquisition name raises at that access
or when explicitly checking consistency, not during ordinary metadata loading.
Normal Pydantic serialization includes the computed model and `master_str`;
`round_trip=True` emits only input fields and does not force the cached parse.
This follows Pydantic's support for
[computed cached properties on dataclasses](https://docs.pydantic.dev/latest/api/fields/#pydantic.fields.computed_field).

#### Field descriptions

Description review: all four reference telescope files supply variable
`long_name` attributes, with identical wording across the fixtures. Useful
labels include "Observation Program", "Observation Goal", "Total Integration
Time" (units `s`), "Right Ascension", and "Declination" (units `rad`). Other
labels are terse or implementation-oriented: "filename", "Master", "TelTime",
and "Telescope.AzActPos". Metadata descriptions use static, semantic wording
informed by these attributes; physical header keys remain in the mapping
schema, not user-facing descriptions. `filepath_orig` is described as "Original
acquisition file path" because the recorded value is a path despite its
`long_name` being "filename". This documentation-only correction adds no
attribute lookup or data reads at runtime.

#### Computation and read cost

Measured against all four fixtures in ordinary and Dask-backed modes:

- Opening and accessor/mapper construction: **zero variable-value reads**.
- First metadata access: **12 selections / 456 logical returned bytes** per
  file, versus T1/T2's 7 / 312. The additional 144 bytes are the original path
  (128-byte header), three observation numbers, and master (four int32 values).
  This is the explicit cost of reading authoritative headers instead of using
  the current filename or a constant. It does not load pointing streams.
- Cached metadata and accessing/caching `filepath_info_orig`: **zero further backend
  reads**. Ordinary loading and its debug log do not parse the filename.
- Calling the mapper again bypasses the accessor cache: on ordinary inputs it
  rereads the two endpoint selections (16 bytes), while this Dask setup repeats
  all 12 selections (456 bytes). No new source-wide cache is introduced.
- Native pointing handles and Dask expression construction are unchanged.
  Ordinary elapsed-time/flag conversion still computes and is accessor-cached;
  Dask construction performs no tasks or reads until materialization.

These are logical backend return sizes, not physical disk/network I/O. The
evidence and diagnostic timings are in
[`reader-comparison-v3-tel-t3.json`](probes/reader-comparison-v3-tel-t3.json).
No wall-time speedup is claimed; xarray opening/model overhead remains separate
from these small additional reads.

#### Verification and remaining boundaries

The telescope parity tests compare all four fixtures with the actual v2
interpreter, with both ordinary and Dask-backed inputs. They preserve exact
array values, dtypes, units, target coordinates and `t0`; renamed or reorganized
metadata fields are checked explicitly rather than hidden by loose comparison.
The LMTMC slice independently checks the quartet, program, goal, original path,
actor and rounded integration time. Synthetic cases exercise missing/invalid
master, strict required fields, literal identities, renamed current files,
header/original conflicts, optional checks, fixed-width text, and serialization.

Final verification: **86 telescope tests passed**, including the 8 actual-v2
parity cases; **750 tests passed** in the complete Tolteca suite with v2 parity
enabled and benchmark timing disabled. Scoped pre-commit hooks, scoped `ty`,
and `git diff --check` passed. Repository-wide `ty` still reports the same 13
diagnostics outside this scope (KIDs wrapper return annotations and intentional
invalid-input/mutation tests). The full `just qa` recipe was not run; the scoped
hook run skips its repository-wide typecheck in favor of the explicit checks.
`just review` was regenerated, with a telescope-code/test subset extracted from
the same Git HEAD-to-working-tree diff. No changes were committed.

The shared `LmtObsQuartetMixin.master` still has its prior string annotation.
Telescope metadata overrides it with the requested numeric nullable type; KIDs,
SourceInfoModel, ObsSpec and ORM contracts are not migrated in this change.
The enum still covers only TCS/ICS/CLIP, as before; the DB's code-3 REPEAT case
remains the previously recorded gap, not an excuse to fabricate a value.
The shared path naming and catalog aliases were subsequently aligned in C3 below.

### C3: explicit path models and short catalog aliases

The approved naming separates structured objects from query shorthand:

- `FilepathInfo` replaces `FilenameInfo`; directory-derived master information
  is part of its contents.
- `filepath_orig` remains the recorded path string. `filepath_info_orig` is its
  cached parsed model on telescope metadata and source records.
- Source records store the effective path fields once in `filepath_info`.
  `file_ut`, `file_suffix`, and `file_ext` are properties of that object, not
  independently mutable values. This replaces the previous flat input fields;
  it is an intentional v3 model/serialization change, without compatibility
  aliases. `to_context()` still exposes scalar shorthand for config conditions.
- Catalogs store effective and original models as two Arrow struct columns.
  Queries accept both `filepath_info.ut` / `filepath_info_orig.ut` and
  `file_ut` / `file_ut_orig`; the same rule applies to every path field.
  Aliases are resolver entries, not additional persisted DataFrame columns.
- `check_filepath_orig` is the consistent name for the opt-in header/path
  comparison; its default remains false.

The effective field selection is unchanged: recognizable current filenames
retain suffix/date/extension, an unrecognized name can fall back to the original
path, and original-directory master outranks a relocated directory. Explicit
header identifiers still overwrite top-level source metadata, not the parsed
path's observation numbers. Telescope headers and arrays are not changed.

The previous discovery `timestamp` was timezone-naive. Its replacement
`file_ut` explicitly carries UTC, matching the already UTC-aware original
model. This is a representation change, not a change in the acquisition time;
no filesystem mtime or local timezone interpretation is introduced.

Aliases project only requested Arrow fields. Pandas mutates resolver entries
during expression evaluation, so projected values live in a writable,
query-local dictionary. Nothing is cached on the accessor across queries:
reordering, duplicate indices, and replacing a struct column remain visible.
Tests reject path parsing/model reconstruction during selection and assert
that a scalar-only query performs no struct projections. Catalog reconstruction
reads the effective struct; the original cached model is still derived from
`filepath_orig`. Parquet round-trips preserve both typed structs and UTC.

No new file-opening policy, automatic remote transfer, KIDs scientific-field
change, or DB/Web migration is included.

Verification: **762 Tolteca tests passed**, with the actual v2 interpreter
enabled and benchmark timing disabled. Scoped pre-commit hooks and scoped
type checks passed; the repository-wide typecheck's previously recorded
unrelated diagnostics are outside this change. The workspace and focused
review diffs were regenerated from Git HEAD, with no commit made.

### C4: source records, catalog projection, and guide organization

The v2 functions `guess_info_from_source()` and `guess_info_from_sources()`
remain the public guessing entry points. Filename recognition now lives in
`FilepathInfo.from_path()`: compiled patterns capture model field names directly,
including ROACH, and each pattern carries its UTC date format. Pydantic validates
numeric captures. Kind guessing uses an ordered, compiled interface/suffix/ext
rule table through `FilepathInfo.guess_data_kind()`. This restores v2's explicit
rule-table organization without copying its wildcard shadowing bug.

| Behavior | Decision / justification |
|---|---|
| Suffix grammar | Accept non-dot/non-slash suffix characters, including hyphens, as v2 did. Telescope/Wyatt date-first names also accept suffixes. The prior v3 `\w+` restriction and missing telescope suffix group were unnecessary limitations. Extension grammar is unchanged. |
| Auxiliary kinds | Require `.nc` for telescope, Wyatt, HWPR and housekeeping kinds. A recognizable `.txt` path retains parsed identity but has `Unknown` kind. V3 previously ignored these extensions. |
| Processed sweeps | Retain v3's explicit processed kinds. Executing the v2 parser confirmed that its earlier wildcard `.nc` rule incorrectly classified these as `RawTimeStream`. The new table uses an empty-string match for an absent suffix, never `None` as both absence and wildcard. |
| Unknown suffixes | Retain `Unknown`, rather than reproducing v2's broad `RawTimeStream` guess. Recognized identifiers remain available for subsequent header enrichment. |
| `hold_flag` | Rename the v3 telescope schema/accessor member from `holdflag`. The v2 reference bridge keeps the old name and compares identical values, dtypes and units in ordinary and Dask modes. |
| Source aliases | `file_ut`, `file_suffix`, `file_ext` become Pydantic computed fields backed by simple properties. They appear in ordinary dumps, but not round-trip input dumps or catalog columns. They are not separately cached. |
| Source serialization | Normal dumps retain FileLoc structure and the native enum representation. `context={"flat": True}` requests resolved source strings and kind names via field serializers; name validation also handles composite flags. Source serialization does not change Tollan's FileLoc contract. |
| Configuration projection | Remove `to_context()`. Consumers call `model_dump(mode="json", context={"flat": True}, exclude={"filepath_info", "filepath_info_orig"})`. The runner computes this once and reuses it for configuration, logging and live input metadata. Native Pydantic datetime output uses `Z` instead of the earlier manual `+00:00` suffix; the represented UTC instant is unchanged. |
| Catalog construction | Replace the free function with `ToltecFileAccessor.from_info_list()`, paired with `to_info_list()`. One Python-mode dump per record feeds the DataFrame and typed Arrow structs, without a JSON conversion or unused alias computation. Empty schemas, null fields and query/Parquet behavior remain covered. |

The existing frozen `SourceInfoModel` is retained. A local Pydantic 2.13.5 probe
confirmed that `model_copy(update=...)` retains cached-property values, even when
their inputs change. Plain aliases avoid that problem. The cached original-path
parse remains, and header enrichment continues constructing fresh validated
records. No cache invalidation framework, new opening mode, remote transfer,
DB/Web migration, or change to scientific arrays was added.

Public documentation is organized under **Working with observation data**:
discovery/selection owns location and source-record handling; reading owns
resource lifetimes, metadata, telescope arrays, sweep views, and serialization.
Design decisions and v2 discrepancies are recorded here, not in those chapters.

Verification: **823 tests passed**, with the v2 interpreter enabled and benchmark
timing disabled; scoped pre-commit and type checks passed. All **21 Python guide
examples** ran using temporary output directories and reference files for reader
inputs. A 10,000-call filename-guessing smoke check measured roughly 52 microseconds
before and 53 after; this is not a statistical performance comparison. Tests cover
single native catalog dumps, no original-path parse for scalar config projection,
and existing no-I/O query/read boundaries. Full Sphinx verification is performed
on a temporary copy to avoid generated changes in the checkout.

### C5: free master names and deterministic kind lookup

Path parsing now stores `master_str: str | None` directly. It no longer converts
a directory name to `ToltecDBRawObsMaster` and back, nor exports a numeric
`FilepathInfo.master`. `simu` and other names without database records are valid.
The existing top-level source/catalog `master` column remains a name, but accepts
an unrestricted string. Raw telescope metadata keeps its numeric header `master`
unchanged. The opt-in header/path consistency check compares names; a missing
header code is not synthesized from a directory.

Without the enum whitelist, searching arbitrary ancestors after any `toltec`
component incorrectly identifies development directories as masters. Guessing
therefore requires `toltec/<master>/<interface>/<filename>`, allowing the `tel`
directory for telescope filenames. In other layouts the path-derived master
is unknown. This deliberately tightens the former ancestor search; it does not
change the reference acquisition layout or header authority.

`guess_data_kind()` now dispatches KIDs interfaces to an extension/suffix table,
and auxiliary interfaces to an interface table, retaining the general `tel_*`
recognizer. Tables are built once at import. No per-source cache, payload read,
inferred instrument fields, or dynamic rule registration is introduced.
The old and new implementations returned identical kinds for **7,308** tested
interface/suffix/extension combinations, including unsupported combinations.
Tests separately enumerate all supported KIDs suffixes for all 13 ROACHs.

A local, same-process microbenchmark alternated old/new implementations across
nine batches of 100,000 calls each. Medians in microseconds per kind lookup:

| Case | Previous rule scan | Table lookup |
| --- | ---: | ---: |
| VNA sweep | 0.319 | 0.160 |
| Processed timestream | 1.810 | 0.162 |
| KIDs property table | 0.627 | 0.167 |
| Unknown KIDs suffix | 2.734 | 0.163 |
| Telescope | 1.478 | 0.233 |

These measure only kind classification, not opening files or complete source
identification. Full filename identification is still dominated by location,
date and model processing; no corresponding whole-pipeline speedup is claimed.
The local comparison script is `/tmp/toltec-file-kind-benchmark.py`.

Serialization contracts are unchanged. FileLoc already supports structured
Pydantic serialization, which remains the default. Its string serializer is
only the explicit `flat` projection for catalogs and condition dictionaries.
That projection preserves a resolved location, not its original relative path
or parent-resolution context. `_validate_data_kind` is needed for the projected
enum names (including combined flags), not ordinary Pydantic Python/JSON round
trips. Tests cover both forms, including zero and combined flags.

Verification: **1,239 Tolteca tests passed**, with the v2 interpreter enabled and
benchmark timing disabled. Scoped pre-commit hooks and type checks passed.
Simulated masters are exercised through filename/original-path identification,
header enrichment, catalog selection and Parquet round trips. Raw-header master
tests and v2 telescope parity remain passing. Nothing is committed in this round.

### C1 catalog scope

The approved catalog choice is pandas with ordinary scalar columns and a typed
Arrow `filename_info_orig` column. A resolver delegates nested field access to
pandas' struct accessor; it does not traverse Pydantic objects during queries.
`SourceInfoModel.filepath_orig` is the recorded acquisition path, separate from
the current `source`. The cached parsed model exposes UTC `ut`, filename
interface, quartet numbers, suffix, and extension. KIDs catalog enrichment reads
the original path through the existing mapper's filename field. Missing or
unrecognized original filenames remain null in discovery records; this does
not make complete reader metadata optional.

This patch does not change telescope metadata's pending master/nullability,
interface, observation-program, or original-path integration decisions. It
preserves the existing discovery timestamp and scalar query behavior. The
catalog tests include both telescope and KIDs original-filename fixture headers,
Parquet round-trips, row/index alignment, nulls, and a renamed-file CLI selection.
See `catalog-dataframe-research.md` for the alternatives and measurements.

### C2 identification precedence

V2's KIDs reader overwrites filename guesses with loaded metadata, and
`ToltecFileAccessor._update_from_item_meta()` writes non-null scalar reader
values into existing catalog rows. The v3 conflict rejection in
`kids.source.identify_source()` was not equivalent and has been replaced with
header correction. Compatible, more specific filename kinds retain the existing
v3 behavior; contradictory kinds are corrected by the reader.

The approved v3 addition is fallback to the recorded original acquisition path
when the current filename cannot be recognized. Current and original paths use
the same parser for filename components, ROACH, and acquisition-directory master.
The parsed original exposes numeric `master` plus `master_str`. The existing
top-level discovery `master` column remains string-valued pending the separately
tracked shared numeric-code alignment; this patch does not migrate ObsSpec/ORM.

Explicit header fields are authoritative. An original-path master is a stronger
guess than a relocated directory, but does not override an explicit header.
`check_filename_orig=False` is the default. When enabled, the reader compares
available original-path ROACH, observation identifiers, and master against the
corresponding explicit headers and raises on disagreement. It does not compare
current path guesses, filename time against observation start, or a raw
acquisition suffix against processed-product kind.

Recognizable current filenames keep their suffix/time fields. A missing suffix
is not fabricated from data kind; an actual suffix is recovered from the
original filename only in the unrecognized-name fallback case. Records remain
immutable snapshots and retain their current FileLoc. Scientific readers and
the telescope metadata contract are unchanged. Query aliases and the three
file-opening policies remain a separate follow-up.

### T1 evidence and established consumers

The v2 baseline is `tolteca_datamodels/lmt/tel.py`, including its filename
parser, `_update_meta_from_io_data()`, and `read()`.
Its raw telescope metadata is not just the fields declared on its model:
`__dict__.update()` adds established values consumed downstream. T1 explicitly
models those values instead of reproducing that unvalidated mutation mechanism.

The installed v2 `taco_recipes` checkout was also examined:

- `scripts/file_utils/toltec_file_utils.py:get_lmt_info_table_from_toltec_info`
  matches telescope files by observation identifiers and TCS master. The **file
  catalog** interface is `tel_toltec`; this does not mean that the v2 reader's
  `interface="lmt"` should be renamed.
- `scripts/file_utils/toltec_file_utils.py:get_obs_goal` reads the required
  `Header.Dcs.ObsGoal` directly and lowercases it.
- `scripts/v1/file_utils_v1.py:get_obs_pgm` consumes reader `mapping_type`;
  `get_obs_goal` looks for reader interface `lmt`.
- `scripts/ql_maps/dispatch_reduction.py` branches on program `On`, master and
  observation goal. Besides the current `LmtObsGoalType` members it recognizes
  `azscan`, `elscan`, and `m3offset`. Rejecting those strings or translating them
  to `unspecified` would change the dispatch inputs.

The four corresponding rows in `test_fixtures/lmtmc_slice.csv` confirm the
observation numbers and program/goal pairs:

| Observation / scan | `ObsPgm` | `ObsGoal` | Recorded integration time |
|---|---|---|---|
| 145605 / 0001 | `VnaSweep` | `Calibration` | 0 s |
| 145606 / 0001 | `TargSweep` | `Calibration` | 0 s |
| 145609 / 0001 | `Tune` | `Pointing` | 0 s |
| 145609 / 0002 | `Lissajous` | `Pointing` | 60.320000087609515 s in netCDF; 60.32 in CSV |

CSV timestamps are second-resolution catalog values, not replacements for the
first sample timestamp. CSV `Instrument="tel"` and `Receiver="Toltec"` describe
different fields from the reader's `instru="lmt"` and `instru_component="tcs"`.
The CSV is cross-check evidence, not an implicit override source for the reader.

The SQLite reference slice has integer `Master` and `ObsType` columns. Its
master lookup is `0:TCS, 1:ICS, 2:CLIP, 3:REPEAT`; its observation-type lookup is
`0:Nominal, 1:Timestream, 2:VNA, 3:TARG, 4:TUNE`. The telescope files do **not**
contain `Header.Toltec.Master` or `Header.Toltec.RepeatLevel`. Telescope
`master=0` therefore preserves v2's defined TCS identity, not a fallback after
failing to read a header. There is no telescope numeric `kind` header to map
from `ObsPgm`; T1 does not invent one. Reconciliation of KIDs raw `ObsType` with
derived `ToltecDataKind` remains K1 work.

### T1 field behavior

| Field / output | Current implementation and v2 comparison |
|---|---|
| `master` / `master_str` | Integer-compatible TCS enum value `0`; readable `"tcs"` exposed separately. v2 used integer `0` plus `master_name="tcs"`. JSON retains numeric `master`. |
| `interface`, `instru`, `instru_component` | Restored reader values `lmt`, `lmt`, `tcs`. No receiver-derived fallback. |
| `obsnum`, `subobsnum`, `scannum` | Opened filename takes precedence, as in v2. Native header mappings remain accessible but do not silently replace these values. |
| `source`, `file_timestamp`, `file_suffix`, `file_ext` | Explicit provenance fields. Same path, UTC filename date, suffix and leading-dot extension values as v2. `file_suffix=None` represents a genuinely absent filename suffix. See source representation exception below. |
| `mapping_type` | Required `Header.Dcs.ObsPgm` string, case preserved. Restores an established downstream field. |
| `obs_goal` | Required `Header.Dcs.ObsGoal`, lowercase as in v2; no enum whitelist or `unspecified` substitution. Empty and unlisted strings remain strings. |
| `target`, `target_off` | Required two-entry RA and Dec headers, radians in ICRS. Entry 0 is the main position, entry 1 the reference position; no one-entry fallback, no coordinate subtraction. |
| `t0` | First native Unix sample converted to Astropy Time with `format="isot"`, matching v2's value and format. |
| `t1`, `t_exp` | Retain existing v3 additions: last sample and required integration timer, respectively. Zero integration is not replaced by the elapsed span. These have no populated v2 telescope counterpart. |
| `time` | Restored elapsed seconds `TelTime - TelTime[0]`. Full values and dtype equal v2 exactly. |
| `time_unix` | Native timestamp DataArray, for access before elapsed-time computation. This is an additive v3 convenience. |
| `az`, `alt`, `ra`, `dec` | Native pointing DataArrays. All four complete arrays, dtypes and radian units equal v2 on every fixture. |
| `holdflag` | Restored `astype(int)` conversion. Values and dtype equal v2. The stored categorical `units="boolean"` label is removed from the result, not from the input. |
| Additional desired/map pointing fields | Existing schema mappings remain accessible; no new automatic reads. v2's extra target Az/El header reads were discarded without producing output and remain omitted. |

Missing required headers fail through the mapper's normal errors. Incorrect
target-header shape and empty time arrays fail explicitly. There is no new
error hierarchy or source-wide cache.

### T1 changes and exceptions for review

| Decision | Justification and review boundary |
|---|---|
| Require goal and both target positions | The fixed fixtures contain them. Missing-goal compatibility in v2 is not extended to the v3 input spec, per the agreed policy. Ordinary goal strings and both coordinate values retain v2 semantics. |
| Keep `obs_goal` as a required string | v2 dispatch accepts meaningful values absent from the existing enum. Enum constants remain available for comparisons; an incomplete enum must not reinterpret raw metadata. |
| Use `master_str`, not `master_name` | Requested numeric-code/readable-string naming. No compatibility alias is introduced. |
| Initial `Path` source representation, withdrawn in T2 | v2 used `FileLoc`. Its JSON-validation failure was repaired in Tollan, and telescope metadata now retains `FileLoc` rather than discarding the host and original path. See T2 for the approved contextual-parent and URL contract. |
| Dataset arrays and typed metadata instead of `LmtTelData` / Quantity arrays | Retain the established v3 xarray/accessor architecture; compare numerical values, dtypes and units independently of container type. |
| Keep `t1` and integration timer | Existing v3 typed time-span fields expose additional native information without altering `t0` or elapsed time. The CSV corroborates the integration timer but is less precise. |
| In-memory source filename | An in-memory Dataset has no opened path, so its required filename header supplies that path. This does not override the opened filename for file-backed data. |
| Cache elapsed time and integer flags on the accessor | These are value conversions, not native handles. Reusing their results avoids repeated ordinary-array arithmetic; Dask caches the expression, not computed results. The same source-lifetime rule as cached metadata applies. |

The shared `LmtObsQuartetMixin.master` annotation still says `str`. T1 overrides
it for telescope metadata only; the shared base and KIDs/catalog contracts are
**not yet consistent**, even though the focused type check passes. That needs
a deliberate shared numeric-code update, not an unnoticed union or conversion
inside this confined telescope change.

Current `tolteca_db` CSV ingestion has its own metadata model and string master;
`tolteca_web` telescope plotting still reads netCDF fields directly. Neither
currently consumes this reader metadata model. Their schema integration and
optional-field policies remain follow-up friction, not hidden T1 changes.

### T1 computation and performance

Evidence is saved separately in
[`reader-comparison-v3-tel-t1.json`](probes/reader-comparison-v3-tel-t1.json).
The original v2/v3 JSON files remain unchanged. The probe now accepts a
telescope-only directory, traces Dask telescope access too, and consumes
`time` directly rather than repeating the now-built-in elapsed conversion.

On each of the four files:

- Opening performs **zero variable-value reads**. This does not imply zero
  filesystem I/O: netCDF/xarray still inspect file structure.
- First metadata access performs **7 backend reads, 312 returned bytes**:
  two scalar time endpoints, two two-element coordinate headers, program,
  goal, and integration timer. No pointing streams are materialized.
- Cached metadata performs **zero** further backend reads. Calling the mapper
  directly again bypasses the accessor metadata cache; it rereads the two time
  endpoints on ordinary inputs, or all seven selections with this Dask setup.
- On non-Dask inputs, accessing `time` and `holdflag` reads and converts the
  full corresponding arrays (plus one scalar first-time read). These are not
  zero-cost handle properties. The other four pointing properties remain
  native handles. `time_unix` allows slicing before time arithmetic.
- On Dask inputs, constructing `time` and `holdflag` executes **zero tasks and
  zero variable reads**. Materialization is deferred; caching the expression
  does not persist Dask results or eliminate later recomputation.
- Complete output remains linear in sample count, with no broadcast expansion.
  Input array values, dtypes and attrs are not mutated by the conversions.

Median warm-process timings in milliseconds, three measured trials (OS cache
uncontrolled), are descriptive rather than a speedup claim:

| Telescope file | Baseline v2 open + metadata | T1 v3 open + metadata | Baseline v2 full output including open | T1 v3 full output including open |
|---|---:|---:|---:|---:|
| 145605/0001 | 4.22 | 11.65 | 5.90 | 12.69 |
| 145606/0001 | 3.91 | 12.27 | 5.05 | 13.22 |
| 145609/0001 | 4.27 | 11.87 | 5.78 | 13.10 |
| 145609/0002 | 4.36 | 13.71 | 6.13 | 15.52 |

The v3 small-file overhead remains. This iteration restores contracts; it does
not establish a performance improvement or a cause for all that overhead.

### T1 verification and next review

- **59 telescope tests passed**, including all four files against an independent
  v2 subprocess with both ordinary and Dask-backed v3 inputs (8 parity cases).
  Tests require exact metadata values, full array values, dtypes, and units.
- LMTMC comparisons cover all four rows; only the CSV integration timer uses
  a tolerance reflecting its decimal rounding. Reader outputs are not rounded.
- Tests cover missing required headers, fixed two-position layout, unlisted
  goal strings, filename precedence, DataTree access, metadata caching, bounded
  endpoint reads, Dask deferred computation and unchanged source attrs/values.
- Metadata JSON round-trip is tested separately. SkyCoord JSON precision is
  checked to `1e-14` radians; native v2/v3 coordinate comparisons are exact.
- Focused Ruff checks pass excluding the repository's existing `CPY001`
  copyright-header rule; focused `ty` checks pass. **Full pytest: 665 passed**
  (`--benchmark-disable` for the final run; the earlier 660-test run also passed
  with benchmarks enabled). This is not a claim that the repository-wide
  `just qa` recipe was run.

Reproduce the telescope comparison from `tolteca/`:

```sh
TOLTECA_V2_PYTHON=/Users/ma/.pyenv/versions/tolteca_v2/bin/python \
  ../toltec_astro_dev/.venv/bin/python -m pytest \
  tests/tolteca_datamodels/lmt -q

../toltec_astro_dev/.venv/bin/python design/probes/reader_comparison.py v3 \
  ../toltec_astro_dev/run/data_lmt/tel /tmp/reader-comparison-v3-tel-t1.json
```

Review T1 before proceeding to K1. In particular, settle the provenance
representation and shared numeric-identity follow-up explicitly; do not
silently broaden this patch to DB/Web or the KIDs reader.

### T2: FileLoc and remote source identity

The T1 `Path` replacement is withdrawn. `FileLoc` was already ported to v3;
its JSON validation failure did not justify discarding remote identity.
`LmtTelMetadata.source` and `SourceInfoModel.source` now retain `FileLoc`.
Filename identification uses its path component and retains the complete
location, including HTTP queries and fragments, without fetching the resource.

| Correction | Reason / behavior boundary |
|---|---|
| Remove the redundant pre-validation Path conversion | Pydantic already validates Path fields. Converting JSON strings into Path objects before its JSON-mode path validator caused round-trip failures. No custom serializer is needed. |
| Preserve parsed URLs instead of unquoting/rebuilding the entire URL | Previously HTTP paths gained an extra slash, and escaped path/query/fragment delimiters could be reinterpreted. Decode the path component only when exposing a filesystem-style path. |
| Encode an explicitly overridden URL path | Preserve literal spaces, percent signs and delimiters in the path without changing query/fragment meaning. |
| Reject unsupported URL schemes | Do not reinterpret `s3://...` as a `host:path` location with hostname `s3`. The supported schemes remain file/HTTP/HTTPS; this does not add an S3 backend. |
| Treat HTTP localhost as remote | `exists()` must not check a local file merely because an HTTP URL has a localhost authority. It still does not probe remote availability. |
| Preserve original relative path and supplied parents in model JSON | Unset parents remain `None` and relative parents remain relative, as in v2. Reconstructing a relative local location resolves against the then-current working directory unless an absolute parent is supplied. Flat catalogs instead carry a resolved local path or remote URL string. |
| Make the URL hostname authoritative | URL strings and dictionaries populate the same hostname. An explicit conflicting `netloc` raises instead of selecting a different parent while retaining the URL's host. This intentionally changes mixed URL/netloc dictionary input. Hostnames exclude ports and credentials, which stay in the URL. |
| Use the remote parent for HTTP replacement paths, including localhost | A network URL does not identify a local filesystem path. A relative replacement requires an absolute remote parent; an absolute URL path does not need either parent. |
| Guard local-path consumers | `info.filepath` rejects remote locations. `open_lmt_tel` accepts local FileLoc/file URLs but explicitly requires staging for remote sources. No implicit transfer or caching is introduced. |

The KIDs CLI and runner's display/log expressions now request the local filepath
explicitly so changing the catalog record's source type does not change ordinary
local CLI output. No KIDs scientific fields or reduction behavior changed.

v2's `FileIOBase` distinguished the original source location from the local file
being opened; its KIDs netCDF reader rejected direct remote access. V3's web
service has separate HTTP Zarr access. Neither fact supplies a shared raw-file
transfer implementation. Remote staging/opening, lifetime management and
preservation of origin alongside a staged local filename remain a separately
reviewed I/O task. T2 does not add a new provenance keyword or an implicit cache.

Before the restart, focused verification: **93 FileLoc tests** and **102 telescope/catalog tests**
passed, including existing v2 parity on all four telescope fixtures in both
ordinary and Dask modes. Added cases cover JSON round-trips, relative parent
context, encoded separators, URL query/fragment preservation, remote metadata
parsing with file-opening hooks disabled, and explicit local-open guards.
Before the restart, full-suite results: **1,287 Tollan tests and 677 Tolteca tests passed**,
with `TOLTECA_V2_PYTHON` enabled and Tolteca benchmark timing disabled. Focused
Ruff and `ty` checks pass (Tolteca retains its existing CPY001 exclusion for
this check). `just review` was regenerated; no commit or remote transfer was
performed. The repository-wide `just qa` recipe was not run.

Review correction: the earlier CWD-capture policy was rejected because FileLoc
must retain contextual parent inputs for relocation. The FileLoc implementation,
tests, and documentation were restored with Git from Tollan `658b701` before
reimplementation; unrelated pipeline and telescope changes were preserved.
The fresh implementation leaves parent resolution unchanged and confines its
behavior changes to the URL repairs and hostname contract above. Public usage
is documented in FileLoc's API docstrings rather than a separate guide.

Fresh verification: `just qa` passed all pre-commit checks, including Ruff and
`ty`, and **1,286 Tollan tests** (including **92 FileLoc tests**). The focused
telescope/catalog run passed **71 tests**, with `TOLTECA_V2_PYTHON` enabled for
reference-data parity. The focused Tollan change was subsequently approved and
committed as `de75e82`; the unrelated Tollan pipeline changes remain uncommitted.

Post-commit telescope check: **678 Tolteca tests passed** with v2 parity enabled
and benchmark timing disabled. Additional integration coverage confirms that
metadata retains unset FileLoc parents through JSON reconstruction, and that
reference-file metadata performs exactly the seven listed backend selections
on first access and zero on cached access. The `target_off` field description
now says reference sky coordinate, consistent with the data and public guide.
No numerical reader behavior changed in this follow-up. Focused Ruff (with the
existing `CPY001` exclusion), formatting, and `ty` checks pass; the Tolteca-wide
`just qa` recipe was not run. Telescope changes remain uncommitted for review.

References for the chosen boundaries:
[Pydantic Path validation and serialization](https://docs.pydantic.dev/latest/api/standard_library_types/#paths),
[fsspec remote/file caching](https://filesystem-spec.readthedocs.io/en/latest/features.html#caching-files-locally).

### T4: accepted telescope names and elapsed duration

The user reviewed the
[pre-overnight naming comparison](lmt-tel-metadata-review-20260921.md) and chose
`t` for elapsed samples, `tel_time` for native timestamps, and `alt` rather than
the telescope's legacy `El` vocabulary. This focused follow-up changes the
following interfaces, without compatibility aliases:

| Change | Preserved behavior / justification |
| --- | --- |
| Accessor `time` → `t` | Exact native timestamp subtraction, dtype, seconds unit, per-accessor caching and Dask deferral retained |
| Accessor `time_unix` → `tel_time` | Native handle retained; snake case follows the recorded TelTime name |
| Metadata `target_off` → `target_ref` | Same second absolute ICRS position; no coordinate subtraction or reordered entries |
| Add metadata `t_elapsed` | Cached `(t1 - t0).to(u.s)`, using the already-read TelTime endpoints; no additional file reads |
| Add `az_des`, `alt_des`, `az_map`, `alt_map` properties | Expose existing schema mappings as native lazy DataArrays |
| Explicit schema names | Header pairs use `ra`/`dec`; sampled coordinates use `source_ra_act`/`source_dec_act`; actual horizontal samples use `az_act`/`alt_act`; timestamp mapping uses `tel_time` |

`t_exp` still reads the header integration timer unchanged, including zero.
It is not replaced by `t_elapsed`, and equality between those durations is not
required. The derived duration uses Astropy Time subtraction; the existing
sample-array subtraction remains unchanged. Header completeness, source
identity, file-opening behavior, and optional integrity checking are unchanged.
The common parser, shared UIDs, and guide consolidation remain separate work.

Verification: **580 focused telescope/catalog tests passed**, including all
eight ordinary/Dask telescope comparisons against the actual v2 interpreter.
Native time/pointing handles are tested for zero backend reads on all four
fixtures in both modes. Cached elapsed duration, JSON serialization, fresh
cache after metadata replacement, and unchanged integration time are covered.
Scoped Ruff (existing CPY001 exclusion), formatting, and `ty` checks pass.
The full Tolteca pytest suite also passes: **1,263 tests**, with the actual v2
interpreter enabled and benchmark timing disabled. The repository-wide
`just qa` recipe was not run. This naming patch was approved and committed as
`d488967`.

### T5: structural telescope identification at open

`LmtTelAccessor` now uses its existing `_validate()` hook to require the mapped
`tel_time` variable in the current payload. It checks through `mapper.has_arr`;
an attribute or bare dimension with the same physical name does not qualify,
nor does a mapping previously resolved against a different source.

This matches v2's telescope recognition criterion: native TelTime presence,
not a complete-header or numerical-value check. Empty or scalar timestamp
variables still pass recognition and fail the existing shape check when
metadata is requested. Required headers remain lazy. No additional public
identification API or physical-name list is introduced.

`open_lmt_tel()` initializes the registered accessor once and returns the same
open Dataset. If accessor initialization fails, including interruption, it
closes the Dataset and re-raises the exception. Successful opens keep the
normal caller-owned lifetime. Existing schema-resolution errors now surface
during opening rather than on first accessor access; directly constructing an
accessor on an unsupported source also fails immediately.

No sample values or metadata headers are read by this identification check.
There are no changes to numerical results, filename parsing, remote staging,
integration-time handling, or optional original-path consistency checks.

Verification: **108 telescope tests passed**, including actual v2 comparisons.
New cases cover Dataset/DataTree recognition, unrelated netCDF inputs, reuse of
a mapper with an incompatible source, cleanup, one-time initialization, and
deferred metadata validation. The full Tolteca suite passes **1,278 tests** with
the v2 interpreter enabled and benchmark timing disabled. Scoped Ruff (existing
CPY001 exclusion), formatting and `ty` checks pass. `just qa` was not run.
This follow-up remains uncommitted for review.
