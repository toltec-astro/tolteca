# 05a — Execute the reduction guide against the prepared-header contract

Final documentation review found that `kids-reduction.md` still imported the
identity-only KIDs helper for its interactive pipeline example and described
missing metadata as optional. Switch the example to shared
`tolteca_datamodels.filestore.identify_source`, matching the runner's complete
raw-header preparation. State the actual requiredness and processed-source
behavior. Keep the public explanation about use and observable behavior, not
this review history. No runtime implementation changes.

Add `design/probes/check_reduction_guide.py` to execute all eight trusted Python
blocks in a fresh temporary directory. `sweep.nc` and the two explicit acquisition
names are read-only links to the reference target/tune inputs. An empty recipe
directory exercises the documented default configuration. Numerical execution,
interactive step context, product reopening, typed manifest input and replay
recipe all run; the script asserts key results rather than only successful exit.
It removes its own temporary outputs and never modifies the reference files.

All **eight blocks pass**. The separate observation guide also passes all
**12 Python blocks and four CLI examples**. Its verifier checks mixed Arrow
metadata/Parquet projection and actual scalar selection. Logs:
`/tmp/tolteca-20260922-reduction-guide.log` and
`/tmp/tolteca-20260922-final-guide.log`. The new probe passes scoped Ruff and
formatting. These are executable local examples, not browser-rendering checks.

Self-review: manual interactive assembly in the example intentionally mirrors
the runner without introducing a second runtime API. Input lifetimes, per-input
configuration, live context and result-view ownership remain visible. No claims
are made about later scientific stages or full legacy recipe compatibility.
