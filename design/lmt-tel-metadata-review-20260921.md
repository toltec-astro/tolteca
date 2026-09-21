# LMT telescope and file metadata — pre-overnight review

This is a design review, not the public user guide. It revisits the work in
`/tmp/toltec-python-metadata-catalog-review.diff`. The initial review made no
runtime changes. The subsequent accepted telescope naming changes are recorded
below; identifier, integrity-check, parser, and guide-consolidation proposals
remain separate work.

## Review boundary and evidence

The pre-overnight Tolteca work is preserved in `6ef2c161..fe3b722`. The later
overnight work starts **after** `fe3b722`. The telescope reader/metadata,
`lmt_base/metadata.py`, and `toltec/file.py` are unchanged between that checkpoint
and the current `fa04291`. The supplied diff identifies its baseline as
`6ef2c16`; it is a saved working-tree review, not an overnight-range diff.

Evidence inspected:

- Current [telescope reader](../src/tolteca_datamodels/lmt/tel.py),
  [telescope metadata](../src/tolteca_datamodels/lmt/metadata.py),
  [base metadata](../src/tolteca_datamodels/lmt_base/metadata.py), and
  [filename/catalog implementation](../src/tolteca_datamodels/toltec/file.py).
- Installed v2 source under
  `/Users/ma/Codes/toltec/toltec_astro/tolteca/src`: `lmt/tel.py`,
  `toltec/file.py`, and the reader's tests.
- `taco_recipes/scripts/file_utils/toltec_file_utils.py`, pointing recipe
  telescope-file discovery, and KIDs uses of observation/file UIDs.
- All four reference telescope files through
  `toltec_astro_dev/run/data_lmt/tel`, opened read-only. This review inspected
  header values/attributes, array structure, and telescope-time endpoints;
  it did not add an all-array read to metadata construction.
- `tolteca_ref_data/test_fixtures/lmtmc_slice.csv`, read-only.
- DB `ingest/netcdf_to_parquet.py` and `repository/file_api.py`; Web
  `obs/viewer.py`, `obs/tel_viewer.py`, and the older `toltecTelViewer`.
- Existing [reader comparison and iteration record](reader-v2-v3-comparison.md).

## 1. Main/reference coordinates

The telescope files contain two entries each in `Header.Source.Ra` and
`Header.Source.Dec`, with radian units. Their descriptions are simply
`Right Ascension` and `Declination`; the dimension names do **not** label the
entries as main/ref or on/off. In these four fixtures, the two positions are
equal, so their values alone cannot establish the index convention.

V2 explicitly constructs two absolute ICRS `SkyCoord` objects from indices 0
and 1, exposed as `target` and `target_off`. It does not compute an offset.
The LMTMC export uses `MainTime` and `RefTime`, and the Lissajous header describes
`TRef` as reference-position integration time. Those support the vocabulary
**main/reference**, but are not independent proof of the coordinate index order.

Accepted: retain the established index order and coordinate values;
rename `target_off` to **`target_ref`**. Keep `target` for the main position.
Describe them as “Main target sky coordinate” and “Reference target sky
coordinate”. This is a field/serialization rename, not a numerical change.
Do not add another `target_main` alias without a demonstrated use.

## 2. Schema names versus accessor properties

The schema should distinguish stored fields precisely. The accessor can retain
the convenient astronomical names that v2 consumers use. In particular,
`schema.ra` and `accessor.ra` need not describe the same representation: the
former is a header pair; the latter is the actual sampled pointing trajectory.

### Agreed schema names

`TB` below abbreviates `Data.TelescopeBackend`, not a new code namespace.

| Stored field | Pre-change schema key | Agreed schema key | Meaning / comment needed |
| --- | --- | --- | --- |
| `Header.File.Name` | `filepath_orig` | retain | Direct metadata field |
| `Header.Dcs.*` identifiers/program/goal/integration | matching metadata names | retain | Direct metadata fields |
| `Header.Source.Ra`, `.Dec` | `target_ra`, `target_dec` | `ra`, `dec` | Two main/reference header positions, not sampled actual pointing |
| `TB.TelTime` | `time` | `tel_time` | Native telescope timestamps, not elapsed time or backend/PPS clocks |
| `TB.TelAzAct`, `.TelElAct` | `az`, `alt` | `az_act`, `alt_act` | Actual telescope horizontal position |
| `TB.TelAzDes`, `.TelElDes` | `az_des`, `alt_des` | retain | Desired telescope horizontal position |
| `TB.TelAzMap`, `.TelElMap` | `az_map`, `alt_map` | retain | Stored map-coordinate streams |
| `TB.SourceRaAct`, `.SourceDecAct` | `ra`, `dec` | `source_ra_act`, `source_dec_act` | Actual sampled equatorial position |
| `TB.Hold` | `hold_flag` | retain | Integration-hold flag, not a physical quantity |

Use `alt` throughout schema and accessor names, consistent with Astropy's
horizontal coordinates. The old telescope's `El` spelling remains only in
physical field names. Other native names use snake case, such as `TelTime`
becoming `tel_time`.

Put brief group comments next to actual/desired/map mappings. The files label
the map fields `Telescope.azMap` / `Telescope.elMap`; Web uses them for its
map-coordinate plot. Neither source establishes a complete projection or
cos(elevation) convention. Do not invent those definitions in comments.

### Public-property comparison

V2 values below refer to `LmtTelFileIO.read()` unless marked as metadata.
V3 entries are relative to `ds.lmt_tel`.

| V2 public name / use | Pre-change v3 | Agreed interface |
| --- | --- | --- |
| `time`: seconds since first telescope sample | `time` | Rename to `t`; preserve exact subtraction behavior |
| Native Unix time available through the node mapper | `time_unix` | Rename to `tel_time`; preserve lazy native access |
| `az`, `alt`: actual horizontal samples | `az`, `alt` | Keep; map through `az_act` / `alt_act` |
| `ra`, `dec`: actual equatorial samples | `ra`, `dec` | Keep; map through `source_ra_act` / `source_dec_act` |
| No corresponding reader properties; Web reads desired streams directly | Schema only | Add `az_des`, `alt_des` native lazy properties |
| No corresponding reader properties; Web reads map streams directly | Schema only | Add `az_map`, `alt_map` native lazy properties |
| `holdflag`, consumed as an integer mask | `hold_flag` | Keep current snake-case name and integer values |
| `t0` on returned data and metadata | `meta.t0` | Keep metadata ownership; no duplicate shortcut needed |
| No populated `t1` / `t_exp` telescope output | `meta.t1`, `meta.t_exp` | Keep last recorded sample / header integration timer distinct |
| No corresponding telescope output | None | Add computed `meta.t_elapsed = t1 - t0`, in seconds |
| `target` on returned data and metadata | `meta.target` | Keep main coordinate in metadata |
| `meta.target_off` | `meta.target_off` | Rename to `meta.target_ref` |
| Extra metadata `mapping_type = Header.Dcs.ObsPgm` | `meta.obs_pgm` | Keep actual program name; do not reinterpret it as a scan geometry |
| `meta.obs_goal`, lowercased; Taco reads the DCS header directly | `meta.obs_goal` | Preserve current values and DCS source |
| Extra `master=0`, `master_name='tcs'` inserted by v2 | Recorded `meta.master`, derived `meta.master_str` | Keep recorded numeric code; never reinstate v2's hardcoded master |
| Opened metadata changed `interface` to `lmt`; filename guess was `tel_toltec` | `meta.interface == 'tel'` | Keep actor `tel`; parsed filename token remains separate |
| `ref_frame='icrs'` on returned data | Coordinate frame on `meta.target` / reference | No extra scalar field needed for the current fixed coordinate contract |
| `source` and filename guesses on v2 IO metadata | Source in discovery record; `filepath_orig` and parsed original info in metadata | Keep source location separate from acquisition identity |

V2's unused reads of `Header.Source.Az/El` did not produce an additional public
coordinate object. Do not add eager reads merely to reproduce discarded work.
Taco's telescope integration primarily discovers the file by observation
numbers and reads `Header.Dcs.ObsGoal`; it is not evidence that every proposed
accessor property already had a production consumer.

## 3. Timing: exposure is not sampled span

No explicit observation-start/end timestamp pair was found in these telescope
headers. `Header.TimePlace.UTDate` is labelled a **UT1 date**, and `UT1` is an
angle; neither is a substitute for a recorded first/last UTC sample pair.
Weather/radiometer update timestamps are not observation boundaries.

| Observation / scan | Header integration time (s) | Requested time (s) | Last minus first `TelTime` (s) |
| --- | ---: | ---: | ---: |
| 145605 / 1, VNA | 0 | 5 | 28.615298 |
| 145606 / 1, target sweep | 0 | 5 | 14.277704 |
| 145609 / 1, tune | 0 | 60 | 56.711381 |
| 145609 / 2, Lissajous | 60.320000087609515 | 60 | 68.508325 |

The displayed sample spans are rounded for this table only. The LMTMC row for
145609/2 reports integration time 60.32 s and date/time 09:27:56 UT; the first
telescope sample is approximately 09:28:06.880 UTC. Thus the catalog timestamp
is not an interchangeable first-sample timestamp either.

Recommendation:

- Keep `t0` / `t1` from first/last `TelTime`, as now.
- Keep `t_exp` from `Header.Dcs.IntegrationTime`, including zero.
- Add a computed **`t_elapsed`**, explicitly defined
  as `t1 - t0`. This needs no additional file reads. It is not an effective
  integration time, so `t_exp_eff` would imply more than the evidence supports.
- Do not require `t_exp == t_elapsed`, and do not silently use one as a fallback
  for the other. Leave the existing elapsed sample-array arithmetic unchanged.

## 4. Observation identifiers

The intended shared owner is correct: quartet composition belongs in
`LmtObsQuartetMixin`; raw-observation and interface-qualified identifiers belong
in `LmtRawObsMetadata`.

However, existing consumers do not all use the same representation:

| Existing code | Current convention |
| --- | --- |
| v2 `SourceInfoModel.uid_raw_obs` | `obsnum-subobsnum-scannum` (no master) |
| v2 `uid_raw_obs_file` | Triplet plus ROACH number, or interface when ROACH is absent |
| Web `obs/viewer.py` quartet/raw-observation helpers | `master-obsnum-subobsnum-scannum`, master is a name |
| DB NetCDF ingestion's `SourceInfo.uid_raw_obs` | Legacy triplet |
| DB ingestion's `DataProdInfo.uid_dp_raw_obs` | Master-prefixed quartet with underscores |
| DB repository `SourceInfo.uid_raw_obs` | Assigned the data-product database primary key |

Proposed public values, matching the Web's readable quartet convention:

```text
quartet          = tcs-145609-0-2
uid_raw_obs      = tcs-145609-0-2
uid_raw_obs_file = tcs-145609-0-2-tel
```

Use a canonical **master name** here, while preserving the integer acquisition
code in telescope `master`. The base currently annotates `master` as `str`,
whereas telescope overrides it with `ToltecDBRawObsMaster | None`. The common
composition must depend on a typed `master_str` contract, not `str(master)`
(which would yield the numeric code for telescope metadata). Reconciling that
base-field typing is part of the identifier change, not an incidental cast.

Since an absent master is explicitly valid, recommend nullable quartet/UIDs
when it cannot be named, rather than creating `None-...` or guessing `tcs`.
A simulated source with an explicitly known name such as `simu` can have an
identity without an official database code. That does not authorize inferring
a missing telescope master from its instrument name or current directory.

These are logical acquisition IDs, not hashes of file content or unique IDs
for every copied/processed file. Distinct source locations and suffixes may
refer to the same acquisition/interface. DB primary keys stay separate.
Observation selection should operate on typed component columns, not repeatedly
split display UIDs. In particular, the current Web parser does not correctly
handle a master name containing `-`; do not promise arbitrary-name UID strings
are already accepted by every ObsSpec/parser consumer.

Use computed fields, with cached derivation where useful, rather than accepting
UIDs as independently mutable input fields. Pydantic supports computed
properties on models and dataclasses, including cached properties and nested
model returns. Keep round-trip input serialization separate from display
serialization. [Pydantic computed-field contract](https://docs.pydantic.dev/latest/api/fields/#pydantic.fields.computed_field).

Log `uid_raw_obs_file` for identified data. When unavailable, log the recorded
path plus the fact that the master is unknown; do not format a fabricated UID.

## 5. Opening, metadata validation, and integrity checks

These have different costs and should not be combined into an implicit full
validation pass:

| Operation | Proposed responsibility | Payload cost |
| --- | --- | --- |
| Accessor `identify(data_source)` | Recognize telescope structure through its schema, starting with native `TelTime` presence/layout | No numerical-value reads |
| `open_lmt_tel()` | Open once, invoke identification, close on rejection, raise a useful `ValueError` | Same structural cost; no eager `.meta` |
| `.meta` | Read required scalar headers, the coordinate pairs, and time endpoints; validate their existing contracts | Same bounded reads as today |
| Explicit `check_integrity()` | Check original-path identifiers against headers and defined metadata/data invariants | Header/endpoints and structural checks; full time-stream checks only if explicitly requested |

V2's content identification checked for `Data.TelescopeBackend.TelTime`.
Current v3 `open_lmt_tel()` has no equivalent check. Contrary to an earlier
assumption, `open_toltec()` itself also just opens xarray; KIDs kind detection
and source enrichment happen subsequently. Do not copy a supposed opener
validation policy that does not exist.

Keep recognition distinct from completeness: a recognized but corrupt telescope
file should fail with the missing/invalid metadata field, not be relabelled as
an unrelated file. The reusable check belongs with the accessor and should use
schema mappings, not another physical-name list in the opener.

Move the optional original-path consistency check into the explicit integrity
operation; do not add duration checks to a flag named `check_filepath_orig`.
Normal opening/metadata access must not turn on this optional work. Structural
length checks do not require array values; monotonic/finite checks over every
timestamp do and must be requested explicitly. No new exception hierarchy is
needed.

Only compare fields known to express the same fact. For example, fixture
145605 has DCS goal `Calibration` but TelescopeBackend goal `Pointing`; other
fixtures differ in the opposite direction. A blanket equality check over
similarly named headers would incorrectly reject the reference data. The
original telescope path `/data_lmt/tel/...` has no master component to compare.

Accessor validation is compatible with xarray's extension mechanism; it does
not require a Dataset subclass. Note that xarray wraps an `AttributeError`
raised during accessor initialization, so a deliberately reported invalid-file
`ValueError` is clearer for this API. [xarray accessor examples](https://tutorial.xarray.dev/advanced/accessors/01_accessor_examples.html).

## 6. Filename parsing and classification ownership

The present ownership is backwards: `LmtTelMetadata` imports `toltec.file`,
whose model carries a ROACH field and whose classifier hardcodes branches for
KIDs versus auxiliary interfaces. A common parser must not import those
instrument decisions.

Recommended small components, not a plugin framework:

| Owner | Responsibility |
| --- | --- |
| `lmt_base/file.py` common `FilepathInfo` | Interface, observation numbers, master name, `ut`, suffix and extension; local/remote path-component parsing without opening files |
| A frozen `FilenameFormat` record and parser | Keep compiled full-match regex and UTC format together; return named fields to the selected model; do not duplicate manual field-type conversion beside Pydantic |
| `toltec/file.py` extended `FilepathInfo` | Add nullable ROACH index and TolTEC format/classification configuration |
| A typed kind-rule table/index | Map normalized interface, suffix and extension criteria to supplied kind values; no knowledge of what KIDs, telescope, or housekeeping means |
| Domain-owned configuration | Declare the recognized interfaces, filename formats, aliases, and kind rules; the algorithm does not select `_KIDS` versus `_AUXILIARY` dictionaries |
| Catalog accessor and its resolver | Own Arrow struct projection and the public `file_*` / `file_*_orig` aliases |

For filename syntax, deterministic means a complete filename has an
unambiguous interpretation. It cannot generally be dictionary-looked-up before
parsing. The current two compiled formats describe date-first and
observation-first layouts; a typed format collection is sufficient. Both
accept telescope prefixes, so dispatching solely on `tel_*` cannot select one
format. Test the disjoint layouts and preserve accepted names rather than
introducing a second ad hoc prefix/format decision tree for an unmeasured gain.

Kind classification, in contrast, should be an indexed lookup. Represent exact
and subset keys in one registry, with **unspecified criterion distinct from an
explicitly absent suffix**. Exact matches take precedence over a less-specific
rule; conflicting equally specific rules are a registry-definition error, not
an order-dependent answer. For three keys, only a small fixed number of key
subsets exists, so indexes can be prepared once rather than scanning every
suffix rule for every file. Interface families/aliases belong to that
configuration, not `if interface.startswith('toltec')` in the lookup algorithm.

Examples the rule set must preserve:

| Criteria | Result |
| --- | --- |
| KIDs interface, `nc`, suffix `tune` | `Tune` |
| KIDs interface, `nc`, suffix absent | `RawTimeStream` |
| KIDs interface, `nc`, suffix `tune_processed` | `ReducedTargetSweep` |
| `tel_toltec2`, `nc`, any suffix | `LmtTel2` |
| Other supported telescope interface, `nc`, any suffix | `LmtTel` |
| Telescope interface, `txt` | `Unknown`, not telescope NetCDF |
| KIDs interface, `nc`, unknown suffix | `Unknown`, not raw timestream |

The current `ToltecDataKind` also contains telescope/auxiliary constants. The
common lookup must accept domain-provided result values without importing this
enum. Moving or splitting the public enum itself is a separate consumer-facing
change; it is not necessary to remove the branching from this algorithm.

Make `_FilepathAliases` own its immutable alias definitions and its per-query
projection cache. The current module-level `_FILEPATH_ALIASES` is not wrong for
performance, but unnecessarily separates those responsibilities. Derive aliases
from the configured struct fields; do not hand-maintain a second list. Preserve
Arrow column projections rather than traversing Python objects row by row.

Unchanged behavior requirements include remote locations not being fetched,
contextual FileLoc parents, invalid-date rejection, current/original suffix
precedence, free master names, exact kind matches, nullable guesses, and
round-trip serialization. The refactor must not smuggle in normalization or
compatibility changes under the label of a registry cleanup.

## 7. Focused user guide and docstrings

The current `observation-data.md` is mostly an entry page linking two other
guides, so it has not achieved the requested single focused walkthrough.
Replace that three-page split with one actual guide, with this sequence:

1. Open a real reference telescope file, inspect typed metadata, and load a
   small pointing slice while the Dataset is open.
2. Explain acquisition identity versus current source location, using that
   same observation and its recorded original path.
3. Build a filename-only catalog, select by observation numbers / `file_*`,
   and show explicitly when headers are read to confirm or enrich guesses.
4. Select associated **KIDs** files for the same observation and run
   `tolteca_kids --show_inputs`, `--show_config`, and `--dry_run`.
5. Show the equivalent Deploy invocation, with the launcher/application option
   boundary clear.

The no-op Kids CLI does not process telescope files. The guide must not imply
that it does. Likewise, general telescope catalog header enrichment is not yet
implemented by the current KIDs-specific `identify_source`; any new shared
enrichment API must be implemented and tested before the guide promises it.
Keep ObsSpec grammar and reduction/product details as linked reference/separate
task guides. Remove the duplicate raw-data/file-catalog guide pages from the
toctree and update their links when this consolidation is implemented.

The telescope `filepath_info_orig` property needs only a contract such as:

> Parsed identifiers from the recorded acquisition path.

An applicable `ValueError` statement can follow. The source-record counterpart
also states its nullable return. Cache implementation and this review's
provenance discussion do not belong in those property docstrings.

## 8. Proposed implementation/review sequence

1. The property/schema table and sampled-duration name are accepted and
   implemented in the focused T4 follow-up. Shared identifiers and structural
   opener checks remain separate implementation steps; neither is silently
   included in this naming patch.
2. Extract common filepath machinery and the configured kind lookup without
   changing accepted syntax, source precedence, or query behavior. Test the full
   existing kind matrix, unknowns, both layouts, registry ambiguity, and remote
   FileLoc cases. Reconcile base master typing as part of the shared-identity
   change, not by putting TolTEC conversions into generic code.
3. Consolidate and execute the guide's examples against the fixture. Move
   rationale only into this design review. Review each focused diff separately
   from the already committed overnight slice.

Do not silently migrate DB keys, change scientific arrays, or start another
mapper redesign. Existing full-array telescope/v2 parity tests remain the
numerical baseline; the additional fields and names above require their own
contract tests.

## Verification during this review

The unchanged reader/catalog baseline passed **577 tests** in 7.22 s, including
the real-v2 telescope comparisons:

```sh
TOLTECA_V2_PYTHON=/Users/ma/.pyenv/versions/tolteca_v2/bin/python \
  ../toltec_astro_dev/.venv/bin/python -m pytest \
  tests/tolteca_datamodels/lmt \
  tests/tolteca_datamodels/toltec/test_file.py \
  tests/tolteca_datamodels/toltec/test_file_catalog.py \
  -q --benchmark-disable
```

This verifies the starting behavior; it does not test the proposed APIs above.
No repository-wide QA, dependency update, commit, or fixture modification was
performed. The existing input diff was not overwritten. Changes from this
review are this design document and its link in the review index.

### Accepted naming follow-up

The subsequent telescope patch implements the agreed table above: `t`,
`tel_time`, `target_ref`, desired/map pointing properties, canonical `alt` in
schema and accessor, and computed metadata `t_elapsed`. It retains the scalar
timing values on metadata rather than adding duplicate accessor properties.
The old names are replaced, with no compatibility aliases. The telescope
`filepath_info_orig` docstring is shortened to its public contract.

The updated focused run passed **580 tests**. All existing v2 numerical-array
comparisons remain exact. Added tests cover native-handle laziness in ordinary
and Dask modes, no extra metadata reads for elapsed duration, and computed-field
caching/serialization. The full Tolteca pytest suite passes **1,263 tests** with
v2 parity enabled; scoped lint, formatting and type checks pass. `just qa` was
not run, and this follow-up is uncommitted. See
[T4 in the reader comparison](reader-v2-v3-comparison.md#t4-accepted-telescope-names-and-elapsed-duration).
