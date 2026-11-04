# TolTEC terminology evidence and decisions

Status: substantive first audit, **2026-10-26**. This maintained record supports
the [TolTEC Data Handbook glossary](../../tolteca_helpdesk/handbook/glossary.md). It records
observed meanings, accepted decisions, naming proposals, and unresolved scopes.
It does not authorize API/schema renames or establish deployment readiness.

The shared scientific definitions moved to the handbook on 4 November 2026. TolTECA retains its domain guide index at the original URL. The migration supersedes the pending ownership statements in the November 1 record below.

## Domain ownership cleanup — 1 November 2026

The user accepted separating scientific terminology from package-specific
concepts after reviewing the `work_on_tolteca_helpdesk` discussion. This pass
changes documentation only. Baseline: TolTECA `8bbf013` with pending workspace
changes; per-file copies and the focused diff are retained under workspace
`run/terminology-domains-20261101/`.

The [helpdesk editorial record](../../tolteca_helpdesk/design/data-handbook.md#accepted-reading-structure--26-october-2026)
establishes the scientific chapter progression. The
[framework comparison](../../tolteca_helpdesk/design/probes/handbook-frameworks/README.md#proposed-publication-design)
proposes moving shared definitions into the handbook. That migration has not
occurred: the shared reference remains at `docs/terminology.md`, and existing
helpdesk links still target it. Its future migration must update ownership and
cross-project references together; this pass does not create a second glossary.

| Content | Maintained user-facing owner |
| --- | --- |
| Observing units, detector measurements, calibration and scientific representations | [Shared terminology](../docs/terminology.md), pending handbook migration |
| Source, identification location, filename information and reader metadata | [Read one acquisition](../docs/read-acquisition.md#sources-locations-and-metadata) |
| Model table, file/data accessors and query values | [File tables](../docs/file-tables.md) |
| Membership versus selected capabilities | [Data-model capabilities](../docs/file-tables.md#choose-registered-or-explicit-capabilities) |
| Product, artifact, item, index, reference and association | [Product guide](../docs/data-products-api.rst) |
| Calibration bindings, collections, results, execution and deployment | Existing domain guides linked from the terminology page |
| Naming proposals, historical comparisons and source evidence | This record and the owning design specifications |

The observation-data specification retains source, identity, model-table and
capability contracts under their respective owners. The guide introduces these
concepts where used; the shared reference links to the software guides instead
of reproducing their contracts or implementation status. Terminology moves from
Getting started to Shared reference in the documentation navigation.

The former glossary incorrectly described fitted-model bindings as planned and
the `reduce` section as proposed. The current `raw-obs-context.rst` demonstrates
`with_model()`, and `reduction-workdir.rst` demonstrates configured model, APT and
pointing preparation. Their guides now own those descriptions. Unimplemented
workflow proposals and downloaded design records are removed from the user
glossary; their evidence remains in `design/`.

Verification: fresh nitpicky HTML with warnings treated as errors passed;
29 guide doctests passed; both quick-start Python examples and the discovery
configuration validated. All 1,712 local links/fragments on the index,
terminology, observation-data and product pages resolved. Browser review checked
navigation, the domain-link table and the new concept explanations. The
documentation diff was reviewed separately and `just review` refreshed the
workspace diff.

The build emits an unrelated numpydoc `UserWarning` for `KidsView`'s unsupported
"Type Parameters" heading. It is not a Sphinx reference warning and remains
outside this terminology change. The first doctest invocation used the package
directory, so the raw-observation guide could not find its workspace-relative
Kidscpp prerequisite. Re-running from the documented workspace root passed all
setup, test and cleanup code. Existing external deployment links were retained;
their remote availability was not revalidated in this pass.
