# 03b — Live invocation, result metadata and aliases

Findings checkpoint only: no lifecycle behavior is changed in this commit.

## Code and baseline reviewed

Read the current standalone SweepReducer, SweepReduce/Context/View, Kids assembly,
DataProdOutput/Context/View, manifest writer/reader, and Tollan Step call/context/
alias implementations. Compare v2 `tolteca_kids/pipeline.py`: its create_context
attaches a fresh live context before run, so failures leave the current incomplete
invocation available on the input. Current Tollan's default storage retains that
property, but the two KIDs overrides only construct a context and serialize it
after a successful run.

The normal fresh-tree slice remains exact. After 02b3/02c, 935 Tolteca tests pass
with actual-v2 enabled (54.86 s). A new complete integration report passes all
four v2 cases, units, CLI/no-op, Zarr, SQLite, Parquet, Web traces and Dash HTTP
callbacks at `toltec_astro_dev/run/overnight-slice-20260922-catalog-guide/`.
This does not exercise every interactive rerun or alias lifecycle.

## Reproduced gaps

Run `design/probes/check_step_lifecycle.py` with the actual ROACH-0 VNA file.
It reads inputs only and prints:

```text
Context attached by create_context: False
After failed rerun, context remains completed: True
After failed rerun, context equals previous success: True
Existing base context now recreates alias: True
Existing base context recreates base: False
Numerical calls for two unforced alias-only invocations: 2
```

1. **Live invocation visibility and failed reruns.** SweepReduce.create_context
   does not attach its context. During first execution, get_context cannot read
   the invocation; after an unsuccessful rerun, the old successful context is
   still what get_context reports. The exception propagates, so a normal
   SequentialPipeline stops. But after catching it interactively, a caller can
   still see completed=True and pass the old result to the publication guard.
   Existing first-failure tests intentionally expect no result/context and thus
   do not cover the established live-context contract. DataProdOutput has the
   same construction/storage split. Distinguish an attempt's state from success
   without inventing array-bearing StepData or serializing an open runtime.
2. **Alias back-link overwrites.** Step.alias creates a subclass sharing the
   original context class. Step.__init_subclass__ writes context_cls.step_cls
   each time; contexts made before alias creation therefore rebuild the latest
   alias rather than their original step. This also existed in v2 and must not
   be presented as a new numerical regression. A context-specific step binding
   needs a focused generic/runtime test; avoid solving it through broad typing
   or class-framework rewrites.
3. **Alias-only cache misses.** SweepReduce stores aliased results under its
   alias path, but its standalone reducer looks only for the canonical result
   child. Repeating an unforced alias-only step invokes _reduce twice. The probe
   counts computations, not backend reads; xarray's input cache can affect the
   latter. Preserve standalone SweepReducer independence and numerical behavior
   when deciding how the adapter presents an existing aliased result.

## Next bounded work

Start with the live-context contract: make create_context attach the invocation,
leave a failed invocation visibly incomplete, and prohibit publishing it. Keep
arrays native and preserve their values/source buffers. Check first failure,
failed rerun, run-time get_context, returned context identity, serialized reopen,
and independent aliases. One possible small design is an owned result-node
context before arrays exist, finalized to JSON metadata after success; evaluate
that against repeated reduction and failure behavior before choosing it. Do not
add a parallel raw-dict/model source of truth or a persistent resource cache.

Then address alias back-links and cache presentation separately. Tollan remains
on its previous branch; create the requested new overnight branch there before
any scoped infrastructure edit, preserving its unrelated .DS_Store.

Other reviewed output boundaries still hold: output is opt-in and a configured
step; existing destination directories are not overwritten; manifests mark
completion after metadata reopen; source/result views use schema/mapper access;
the runner closes its input. The writer copies all non-time metadata variables
and uses automatic Zarr chunking. Selection amplification/chunk policy and the
scope of exported metadata remain open review items, not grounds for changing
scientific values or adding unreviewed compression/dependency changes.
