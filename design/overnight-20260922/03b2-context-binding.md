# 03b2 — KIDs context adapters retain alias identity

Depends on Tollan `62e3a5c`; see its
[design record](../../../tollan/design/overnight-20260922/03b2-context-binding.md)
for the reproduced back-link bug and Pydantic inheritance choice.

The KIDs reduction and publication overrides now construct `self.context_cls`,
not the literal declared base model. Their `view()` factories pass the bound
context type alongside the existing context key. Directly constructed views
retain their existing default model and key; aliases should use the step's view
factory. This keeps live and deserialized contexts consistent and preserves
`make_step()` identity without adding fields to Zarr/JSON metadata.

Two new adapter regressions failed after the Tollan change but before this
update: hard-coded model construction was still returning the base step.
They now pass, covering live identity, JSON round-trip, both views and unrelated
base contexts. All 230 KIDs tests pass (41.93 s), including actual-v2 parity;
scoped Ruff, formatting and ty pass. Log:
`/tmp/tolteca-20260922-alias-pytest.log`.

Self-review: subtype creation occurs once per step class, not while reading
arrays or constructing invocations. The two casts narrow the existing broad
runtime class attribute to its declared generic type; they do not cast away
field validation. Numerical code and cache selection are unchanged. Dynamic
alias classes use JSON reconstruction, not a new module-global pickle registry;
that limitation is explicit in Tollan's record and alias API documentation.

Next: the independent alias-only cache miss from 03b, then output selection and
live resource boundaries. Full integration will be repeated after those fixes.
