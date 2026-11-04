# Observation-data documentation organization

## Scope and baseline

- Inspected revision: `8bbf013`, with the existing uncommitted workspace changes.
- Authority: the user's request to give data models a useful README and domain
  entry point, and the accepted source, file-table, ObsSpec, store and discovery
  contracts in [the specification](spec.md#observation-data).
- Method: [workflow and contract audit](../../refs/workflow-contract-audit/SKILL.md).
  This is a documentation-organization pass, not a new implementation or full
  package/test audit. Existing tests do not establish additional requirements.
- Product, reduction and operational contracts remain outside this pass.
  Concurrent edits to those domains in `spec.md` are preserved.

## Findings and dispositions

| Finding | Disposition |
| --- | --- |
| The repository README contains template placeholders and no usable example. | Replace it with installation prerequisites, a filename-only quick start, and task links. Include the same README in Sphinx and link it inline from the index. The quick start was removed on 2026-11-02; see the contract map. |
| The documentation index mixes every guide and API page into one list. | Add a domain/task table and grouped navigation; retain all existing pages. |
| Observation data has a large API page and a cross-domain walkthrough, but no domain entry page. | Add `docs/datamodels.md` to introduce independent entry paths and link existing worked examples and reference material. |
| The observation-data specification begins with model machinery and capability registration. | Introduce sources and acquisition use first; place capability selection and extension contracts after table use and discovery. |
| Candidate-discovery requirements mix file-store behavior with configuration orchestration. | Give file stores and input discovery separate sections. Stores return candidate paths; discovery returns a populated, selected model table. |
| Concrete/composite implementation details interrupt the collection contract. | Move workflow-extension contracts under instrument extensions; omit the private generic-binding/cache mechanism from the system specification. |
| The prior strict build reports five short heading underlines. | All five are already corrected in the working file at this pass's baseline; verify the current headings in the fresh build. |
| Two new telescope configuration types lack API targets in the full docs build. | Add their existing model declarations to `reduction-api.rst`; no reduction contract or implementation change. |

## Contract-to-documentation map

Current user decisions establish the boundaries below. Implementation and tests
are conformance surfaces, not independent authority for those decisions.

| User task / required boundary | Specification owner | Implementation | Teaching path | Relevant existing checks |
| --- | --- | --- | --- | --- |
| Inspect one acquisition; distinguish filename evidence from recorded metadata; retain resource ownership. | [Sources and metadata](spec.md#sources-and-metadata), [LMT observations](spec.md#lmt-observations), [TolTEC observations](spec.md#toltec-observations) | `core.file.source`, `core.filepath`, `core.source`, `lmt.tel`, TolTEC readers | `read-acquisition.md`, `file-tables.md` | Source, filepath, reader and LMT metadata tests |
| Collect rows; read authoritative metadata separately; query current values. | [Model tables](spec.md#model-tables) | `core.file` and concrete workflows | `file-tables.md` | Core file construction, opening, metadata and composite tests |
| Parse context-free constraints; interpret them where used. | [Observation selection](spec.md#observation-selection) | `core.obs_spec`, `lmt_base.obs_spec`, concrete contexts | `obs_spec.md` | ObsSpec grammar, context and columnar-query tests |
| Search declared layouts without opening files or applying positional selection. | [File stores](spec.md#file-stores) | `core.file_store`, concrete stores | `input-discovery.md` (Python search of a tree) | Store and store-view tests |
| Populate/query each configured input; combine selected tables with diagnostics. | [Input discovery](spec.md#input-discovery) | `core.input_discovery`, root assembly, concrete entries | `input-discovery.md` (recipe and command line) | Discovery, registration and integrated-store-query tests |
| Select capabilities without changing filename parsing; permit direct concrete use. | [Data-model capabilities](spec.md#data-model-capabilities), [Instrument extensions](spec.md#instrument-extensions) | `core.data_model`, root assembly, dummy extension | `file-tables.md` (capabilities), `datamodels-architecture.rst` | DataModelSet, neutral-import and dummy workflow tests |

The 2026-11-02 pass superseded the original layout at the user's direction. The
README keeps only the task table, because the package is a suite and a
filename-only example does not belong there. The domain page `datamodels.md` is an
overview with a levels table (one file, many known files, a data tree, a recipe or
command line) that points to one guide per level: `read-acquisition.md`,
`file-tables.md`, `file-stores.md` and `input-discovery.md`. The cross-domain walkthrough `guide.md`
is retired; its real-data examples moved into those guides, and its probe
`design/probes/check_observation_guide.py` is replaced by the executed examples
(`just doc-test`). The "Observation data" chapter lists these pages at one level
under its caption.

## Remaining scope

- The rest of the package specification still needs domain-by-domain review;
  this pass does not claim that product or pipeline documentation converges with
  its contracts.
- `datamodels-api.rst` combined substantial usage and extension guidance with
  API declarations. The 2026-11-02 pass moved the usage prose and examples to
  `file-tables.md` and the extension guidance to `datamodels-architecture.rst`;
  the API page keeps the declarations. The workflow contracts did not change.
- This pass does not re-audit the existing test matrix or reader implementations.

## Verification

Commands, baseline copies and build logs are under
`run/datamodels-docs-organization-20261101/`, relative to the Python workspace.
The focused diff compares with that saved starting state, not Git HEAD; it
excludes concurrent changes outside the observation-data specification.

- Both new Python examples execute successfully without local data files.
  The discovery YAML validates through `InputDiscoveryConfig`.
- Fresh strict HTML: `sphinx-build -M html tolteca/docs tolteca/docs/_build
  -E -a -n -W --keep-going`; zero warnings.
- Fresh strict Sphinx doctest: 29 tests pass, with no setup/cleanup failures
  and zero warnings. An earlier isolated build directory could not fetch ten
  external inventories under the sandbox; the final build uses the existing
  documentation cache without disabling any checks.
- All 452 local links and fragments on the index, README and domain page resolve.
- Browser inspection confirms the grouped navigation, inline README link,
  rendered examples and domain-page tables.
- No implementation or test expectations changed. Installation commands and
  the full scientific test suite were not rerun for this documentation-only pass.

## Clean-checkout review — 4 November 2026

The four observation guides are `read-acquisition.md`, `file-tables.md`,
`file-stores.md` and `input-discovery.md`. The extension guide is
`datamodels-architecture.rst`. Software terms remain with these workflows.
The [handbook glossary](../../tolteca_helpdesk/handbook/glossary.md) owns shared
scientific definitions. TolTECA's [terminology index](../docs/terminology.md)
links to the glossary and to the software guides.

The reviewed publication includes the observation-data specification, guides,
API, recipe downloads and product terminology. Additional product, reduction
and pipeline sections in the shared working tree belong to their respective
domain reviews. A checkout built from the reader commits exercised the typed,
composite and discovery entry points with the reference tree. Fresh strict HTML
and all 57 executable examples in the scoped checkout passed.
