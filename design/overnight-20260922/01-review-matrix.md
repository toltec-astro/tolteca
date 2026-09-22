# 01 — Initial end-to-end review matrix

This is the first findings inventory, not certification of each layer. Read all
prior 20260921 step records across Tolteca, DB and Web, then inspected the current
CLI, runner, file stores/ObsSpec, header source adapter, configuration, reduction
step/view, output writer, DB registration/catalog, and Web product service/figure.
The standalone numerical reducer still needs a focused code/performance review.

## Fresh integration baseline

Executed from the workspace, using the existing environment and isolated Deploy:

```sh
./toltec_astro_dev/.venv/bin/python tolteca/design/probes/kids_vertical_slice.py \
  toltec_astro_dev/run/data_lmt \
  toltec_astro_dev/run/overnight-slice-20260922-baseline \
  --v2_python /Users/ma/.pyenv/versions/tolteca_v2/bin/python \
  --deploy_source worktrees/tolteca_deploy-overnight/src
```

Passed at 05:23:59 UTC. Four cases: VNA, target, tune block 0, tune last block.
Exact `assert_array_equal` comparisons cover I, Q, unc_I, unc_Q, f_chan, sweep,
frequency; units agree. The same test executes direct/planned no-op invocations,
product writing/reopening, real local SQLite registration, Parquet export,
selected Web traces, and Dash HTTP callbacks. No tolerances or fixtures changed.
Generated report/logs/products remain under the output directory, not in Git.
No server or browser session was started; HTTP test-client checks are not pixel
or interactive browser verification.

## Findings by boundary

| Boundary | Current evidence | Follow-up / constraint |
| --- | --- | --- |
| Filename evidence / source identity | Direct registries and eager stored fields; source kind classified separately; 132 recorded projections preserved | Complete in step 00. Do not reintroduce filename-owned kind, prefix→UID coupling, or inferred header values. |
| Telescope discovery | `_resolve_obs_spec` searches only `root/toltec/<master>/<interface>`. The two actual 145609 telescope files are under `root/tel/`. `145609/tel` returns no files. | Add a bounded explicit telescope-interface discovery path; preserve current unqualified KIDs selector results. Canonical interface is `tel`, stream selection uses `file_prefix`. |
| ROACH discovery | `145609/0` returns the two actual raw/tune files. An initial `145609/8` probe returned none because only ROACH 0 payloads are in this fixture; that is not a resolver defect. | Keep filesystem evidence distinct from the broader 132 filename records in the DB/LMTMC slices. |
| Source opening / header enrichment | `build_input_catalog` always opens known ROACH and unknown-name `.nc` sources, routes them to the KIDs identifier, skips known telescope headers | Not the agreed general opening-policy solution. Investigate typed telescope enrichment and explicit no-open/on-demand/all policies separately; avoid accidental full tune metadata/LO reads. |
| Catalog field meanings | Nested path fields and aliases use Arrow, source kind is top-level. Header enrichment still returns only identity/kind, not a general typed scalar metadata projection | Review against v2 drivefit/taco usage before adding fields. Do not create another untyped header dictionary or reduce arrays for selection. |
| ObsSpec master labels | Source records preserve free master labels such as `simu`; ObsSpec restricts master to `ToltecMasterType`, and a test intentionally rejects `simu` | Explicit policy mismatch to decide, not silently "fix" during telescope directory work. Never infer TCS from a masterless telescope path. |
| CLI/runtime | Shared `files` config, explicit file options, `kids\\..+` dotted override restriction, per-input independent runtime, optional output | Established boundary holds in tests/probe. Keep Kids independent of deployment/env interpretation; inspect repeated handler/config work before optimizing. |
| Deploy | Isolated `plan_kids` produces explicit roots and forwards config paths; shared generic `run` stays unchanged | Existing `run_kids` is an additive prototype for review. Do not silently merge it into the shared C++ checkout or generalize launcher behavior. |
| Steps / live data | Runner wraps source Dataset in DataTree; reduction delegates to standalone SweepReducer; views expose result arrays/context; output remains a configured step | Review context ownership, repeated aliases and failure paths before simplifying anything. No new array-bearing StepData model. |
| Persisted source metadata | Manifest includes the shared source model; nested filename-kind removal changes its development payload shape | New products round-trip in the fresh probe. Regenerate test artifacts; do not add compatibility readers for yesterday's unreleased model. |
| DB projection | Registration checks lineage; Parquet catalog labels `file_ut` correctly. Older ORM `obs_datetime` still receives filename UT | Record naming/semantics friction. Do not equate filename UT with header/telescope observation time or invent a timestamp. Broader DB model unification remains separate. |
| DB identity | Existing DB source adapter intentionally preserves filename prefix as its `interface`; datamodel interface is now canonical | Keep explicit adapter boundary for this slice. No silent UID/database-key migration. |
| Web | Product IDs distinguish reruns; scoped opens close; batched channel reads; exact trace parity passes | Review frequency-grid allocation and chunk/read amplification. No cross-source or float-keyed cache. Legacy portal remains separate and still has `tollan.db` debt. |
| Guide | `observation-data`, `raw-data`, `file-catalogs` remain separate pages | Consolidate a focused telescope→typed metadata→catalog/select→KIDs no-op→Deploy walkthrough using executed real examples. Keep output-specific details separately linked. |

## V2 and actual-system checks

Read the v2 `LmtFileStore` implementation and taco's
`scripts/file_utils/toltec_file_utils.py`. They separate telescope `tel/` from
TolTEC master/interface directories. `get_lmt_info_table_from_toltec_info` finds
associated telescope files through `lmt_fs.tel_path`, not a TolTEC master path.
Its explicit TCS prerequisite describes association with a known TolTEC source;
it does not make TCS filename metadata in a bare telescope path.

The reference root has four ROACH-0 payload files and four telescope payload
files. LMTMC/DB slices provide more recorded names, but not their local payloads.
Diagnostics and tests must not conflate these inventories.

## Next bounded step

Start with discovery for explicitly requested canonical `tel`, backed by the
recorded directory layout. State and test what existing unqualified and ROACH
selectors keep doing. Treat general header enrichment and scalar catalog design
as a separate review/fix, so this does not turn into another large rewrite.
Then inspect reducer/output lifecycle and computation costs, retaining the fresh
exact-parity baseline throughout. Finish with the guide and full verification.
