# Observation data

Observation data access makes recorded measurements and their context usable
without requiring scientific reduction. Source access owns how data is reached.
Instrument interpretation owns its meaning. Scientific processing owns derived results.

Individual acquisition access and collection preparation are independent entry
paths. Collections can start with filenames, explicit sources or declared stores.
Their consumers select records before scientific processing.

## Sources and acquisition access

Source access supports individual inspection and batch interpretation without
repeating storage-specific work for each file.

### Source and identification locations

Locations distinguish the resource to read from the naming evidence used to identify it.

| Term | Meaning |
| --- | --- |
| Source | Resource to inspect: a location or an already-open object (`source`). |
| Source location | Location from which the data is accessed. |
| Identification location | Filename or path used to interpret acquisition naming conventions (`source_loc`). It may differ from the actual source location. |
| Filename information | Preliminary acquisition information inferred from a naming convention. |
| Acquisition metadata | Recorded information describing the measurement and its observing context. |

- Single-source preparation validates the resource and identification location,
  then parses the filename eagerly. Invalid input raises. This operation does
  not collect history or apply a batch error-recording policy.
- A supported source remains accessible when its filename is unfamiliar or
  absent, including when supplied as an open object.

### Filename recognition

Filename information supports preliminary selection without opening the source.

- The universal factory uses all registered naming conventions independently of
  collection scope. Source preparation always invokes this universal factory.
- A factory invoked on a concrete filename model considers only that model and
  its descendants. It never returns an ancestor or sibling. Concrete models also
  support annotations and instance checks.
- The most specialized eligible matching model wins. Unresolved ambiguity raises
  before field conversion, including during batch error-recording mode.
- The basic parser follows the same registration and specialization rules as
  instrument parsers. An unfamiliar filename supplies its extension and an empty
  immutable set of identified data-model memberships.
- General telescope names establish LMT production and LMT membership. Registered
  instrument conventions add membership for their telescope companions.
- Filename parsing supplies extension information. Format opening must not derive
  it independently. Filename models and parser registration remain independent
  of [file-table projection](#preparation-and-current-values).

### Format opening

Opening acquires a format-level resource independently of instrument identification.

- Opening policies match logical record attributes and pass the explicit source
  unchanged to the selected callable. Matching alone acquires no resources.
  Format callables own location support and preparation.
- Standalone source opening uses filename information with an opening policy.
  A caller may supply a policy for that operation without subclassing the source
  model or selecting data models. It yields a resource without identifying a
  reader or collecting metadata, and applies the [ownership contract](#data-access-and-resource-ownership).
  Newly opened resources close at context exit. Borrowed resources remain caller-owned.
  Single-source opening returns or raises without collecting history.
- Policies compose without changing shared defaults. Rules within a policy use
  first-match ordering, including explicit overrides. No match raises for direct
  opening. A selected callable's failure propagates without trying another rule.
- Batch opener iteration selects callables from table values, opens in row order
  and transfers each yielded object's ownership before opening the next row.
  Failure or early termination must neither close already-yielded objects nor
  open later sources. [File-table scopes](#metadata-reads-and-resource-scopes)
  provide the separate managed-resource workflow.
- Typed opening combines a known format opener with its known accessor, without
  reader selection or full metadata reading. Failed binding or classification
  closes the newly opened resource. Success transfers ownership to the caller.
  An already-open object with a known reader uses that accessor directly.

### Reader identification

Reader identification chooses an interpretation without requiring complete
acquisition metadata or signal arrays.

- Concrete workflows supply ordered reader rules whose results are accessor
  classes. The rules consume current query values, including membership tests
  on `data_models`. Construction applies the selected accessor's validation.
- When current values cannot select a class, content identification extracts
  routing values through the workflow's mappers and applies the same rules.
  It must not trial accessor construction, read full metadata or use parallel
  acceptance predicates.
- Content identification can recover unfamiliar filenames. Once a reader is
  selected, construction or metadata failure must not trigger another
  interpretation. Fallback addresses a missing rule result, not a failed reader.
- Workflows constrain accepted reader scopes. These scopes can overlap:
  general telescope access accepts telescope acquisitions, while TolTEC accepts
  its telescope companions, KIDs and HWPR data. Both use this resolution contract.

### Batch layouts and semantic access

Files from the same writer share stable layout conventions. Semantic access uses
those conventions without assuming equal identities, acquisition values or sample counts.
Layout identifies the kinds and locations of recorded entities.

| Concept | Responsibility in acquisition access |
| --- | --- |
| Schema | Give recorded entities a consistent scientific vocabulary. |
| Resolved mapper | Bind that vocabulary to a source layout for reuse across compatible sources. |
| Accessor or view | Apply a mapper to a particular source and expose its data through a domain API. |

- Reader design supports and encourages resolved-mapper reuse across compatible
  files. Compatibility depends on the bindings required by the requested operation,
  not complete layout equality. Extra fields or omitted unused fields permit reuse.
- Reuse preserves each source's own values and identity. Layout bindings must not
  substitute data from the source used for resolution.
- Resolution and reuse follow
  [Tollan's mapping contract](../../../tollan/design/spec.md#schemas-and-mappings).

### Data access and resource ownership

Data access permits inspection of selected measurements without unnecessary
whole-acquisition reads.

- Selection and reordering preserve correspondence to the source acquisition.
  Local array or table positions do not replace source identifiers.
- Deferred reads require the source to remain available. The operation that opens
  a resource owns its cleanup, including on failure. Supplied resources remain borrowed.
- Views provide read-only access to their bound source and descendants, excluding
  parent and sibling data. The component that changes a source manages affected
  view lifetimes. Views do not detect edits, refresh cached state or rebind themselves,
  under [Tollan's view contract](../../../tollan/design/spec.md#accessors-and-views).
- A reader describes an acquisition with stable identity and layout. Tracking
  later edits to recorded metadata is outside that reader's scope.

## Observation metadata

Metadata supplies validated observing facts for selection and interpretation.
It connects related acquisitions independently of their storage locations.

### Authority and validation

Recorded facts take precedence over preliminary naming evidence.

- Recorded metadata supplies authoritative values. Filename information remains
  separate provenance. A recorded original acquisition path is provenance too,
  distinct from the current source location.
- Missing required acquisition facts are reported. Guesses must not complete a
  metadata record. Metadata inspection does not require reading unrelated scientific data.
- Shared data models validate values before interpretation or selection. They use
  strict types, are frozen by default and validate defaults and default-factory
  results. Models needing a different input or mutation protocol declare exceptions.
- Reusable field types own validation, normalization and serialization. Models
  enforce additional constraints and relationships separately.

### Observation and interface identity

LMT conventions relate observations, acquisitions and producing interfaces.
Shared identity does not require a shared file or location. Independent sampling
means joint analysis, not identity matching, owns time alignment.

| Term | Identifier or field | Meaning |
| --- | --- | --- |
| Observation | `uid_obs` | Acquisitions sharing one observation number within the master context. Master-qualified form: `{master}-{obsnum}`. |
| Acquisition, or raw observation | `uid_raw_obs` | One recorded acquisition within an observation. Master-qualified form: `{master}-{obsnum}-{subobsnum}-{scannum}`. |
| Quartet | `master`, `obsnum`, `subobsnum`, `scannum` | Acquisition controller, observation number, sub-observation number, and scan number. |
| Acquisition file | `uid_raw_obs_file` | A file belonging to an acquisition. Its identity includes the constituent interface. |
| Acquisition master | `master` | Controller associated with the acquisition sequence. |
| Producing instrument | `instru` | Instrument that generated the data. |
| Data-model membership | `data_models` | Applicable data-model names. Membership order establishes no precedence. |
| Interface class | `interface_kind` | Kind of producing interface, such as `roach`, `tel`, or `hwpr`. |
| Interface instance | `interface` | Particular producing interface, such as `toltec7`, `tel`, or `hwpr`. |

- One observation can contain several acquisitions and interfaces. Producing
  instrument identity differs from applicable data models, whose memberships
  can overlap. Canonical observation identifiers are independent of storage location.
- An unknown master may be omitted, never inferred by default. Such identifiers
  are context-relative and not unique across masters.
- An observation UID requires `obsnum`. A raw-observation UID requires all three
  numeric components. Its file UID additionally requires `interface`. Missing
  required components yield no UID.
- Quartet formatting requires `obsnum` but permits an omitted master and omitted
  trailing numeric components. Numeric gaps yield no identifier. Later components
  must not shift position. UID constructors enforce completeness for their scope.
- Database ingestion and observation product identity require a known master
  and the complete numeric identity for their scope.
- Acquisition metadata supplies complete interface identity: producing instrument,
  applicable data models, interface class and instance. Identity fields require
  explicit values, including single-value constraints. Mappers supply those values,
  and type constraints do not imply defaults.

### Telescope telemetry

Telescope access supplies observing context and pointing for inspection and joint analysis.

- Telescope recognition is independent of receiver-specific restrictions.
  Instrument data models identify their telescope companions from recorded metadata.
- Telescope records have `instru="lmt"` and include `"lmt"` in `data_models`.
  The receiver header supplies additional membership, such as `"toltec"`,
  without changing the producing instrument.

### TolTEC observations

TolTEC access relates its acquisition streams using the shared source-access,
metadata and LMT identity contracts.

| Area | Role in a TolTEC observation |
| --- | --- |
| KIDs readout | Detector data for sweep and timestream inspection and processing. |
| Half-wave plate readout | Modulation data for joint analysis with detector observations. |
| Telescope companion | LMT-produced telescope data belonging to both the LMT and TolTEC data models. |

### Native processed KIDs metadata

- Native processed KIDs files expose the acquisition source recorded by the writer.
  Reading this reference does not access the acquisition. Its filename can supply
  preliminary identity independently of the processed file's name or location.
- Raw acquisition settings require the acquisition headers. When no acquisition
  source was recorded, the processed reader still supplies its classification and
  the table records unavailable metadata while retaining earlier values.

## Data-model capabilities

Capability selection chooses interpretation support independently of a record's
producing instrument or its applicable data-model memberships.

### Identity, specifications and sets

Specifications associate named identities with reusable capabilities. Sets select
those specifications without defining another processing workflow.

- A data-model identity owns its name. A specification associates it with accessor
  and store capabilities without an independent registration alias. Capabilities
  need not carry an identity binding or depend on their specification.
- A component's affiliation differs from a record's applicable data models.
  Several models can support one record without inheritance between definitions
  or duplicate shared readers.
- Store requirements identify acquisition and companion stores, not extra
  interpretation handlers. The file-accessor capability owns enrichment, readers
  and observation-query interpretation.
- A set is not a composite identity. One-model and multi-model sets use the same
  operations and resolution rules. Selections are read-only. Narrowing creates a
  new selection and retains identity-to-capability associations, including shared capabilities.

### Registration and import boundaries

Application entry points assemble capabilities. Core building blocks and direct
concrete use remain independent of that assembly.

- Constructing a specification or defining an accessor does not register a data
  model. Direct concrete use requires neither a specification nor registration.
- Instrument component imports may load identity information but must not assemble
  or register workflow capabilities. Definition modules assemble completed components
  without registration. Public data-model, file-table and input-discovery entry
  points enable built-ins. Optional extensions activate through their own explicit entry points.
- Registering an existing data-model name raises without changing the registry.
  Root lookup reflects available registrations. Named selections snapshot them,
  and later registration does not enlarge existing sets.
- Core declarations and explicit selections must not import first-party assembly
  or consult its registry. Shared extension interfaces live in `tolteca_datamodels.core`.
  Importing them must not load instrument packages or register built-in discovery inputs.
- Feature registries belong to their core features. Instrument packages supply
  concrete registrations. The assembled named data-model registry belongs to the
  root data-model entry point.
- Root modules expose task entry points, not extension building blocks. Core
  supplies specifications and sets of explicitly supplied specifications. Root
  assembly adds named lookup, registration and selection.
- Core file access uses every available filename registration without importing
  instruments. The filename entry point enables parsers without assembling data-model
  definitions. Root file access adds selected concrete capabilities to the shared workflow.

## File tables

File tables collect information for inspection, enrichment and selection. A row
represents a collected source, not necessarily complete acquisition metadata.
Columns expose selected attributes and need not reproduce a complete source record.
File, observation and product collections retain their distinct record meanings.
Generic projection follows [Tollan's model-table contract](../../../tollan/design/spec.md#model-tables).

### Preparation and current values

Construction starts with preliminary information obtained without opening sources.
Reading is a later stage or explicit operation. Both consume current values while
retaining the original naming evidence.

- Construction validates source/location pairs and retains one row per input.
  Failed pair validation leaves missing information. A filename failure retains
  the validated pair. Both follow [batch error handling](#failures-and-traces).
- Guessing consumes locations already installed in the table. It accepts neither
  replacement sources nor separate source-to-row alignment. Filename guessing is
  an internal construction stage, with no public re-guess operation.
- Each row retains its filename parse result, or absence of a successful parse,
  through selection, reordering and enrichment. The original parse records filename
  recognition, not current scope.
- File-table integration owns filename projections. Marked metadata fields initialize
  current columns before capability matching, independently of selected capabilities.
  Markers select columns, not rows. Each alias copies its filename column, preserving
  existing current columns, including nulls. Filename models do not own table mappings.
- Matching, queries, row iteration and subsequent computations use current values. They must not
  initialize metadata or substitute original filename values for missing or null current cells.
  Filename-only attributes, such as extension, can supply values where no current
  column exists. Metadata replacement preserves the original file columns.
- Memberships are immutable sets. A nonempty set supplies scope evidence. An empty
  filename-derived set supplies none. Missing cells after preparation failure are
  not another allowed model-field value.
- Binding an accessor or narrowing capabilities does not populate or mutate its
  table. Row positions must remain stable while an operation uses them.

### Capability matching and composition

Selected capabilities determine interpretation, without discarding explicit sources
or suppressing registered filename information.

| Collection | Interpretation |
| --- | --- |
| Mixed | Use all enabled data-model support, including overlapping memberships. |
| Known scope | Use selected data-model support, including its supported companion data. |

- Shared workflows use registered support without requiring a caller-supplied
  context. A specialized workflow can select a data-model set up front. Direct
  concrete access uses declared support without a discovery context.
- Concrete rules match current memberships and any required prepared interface
  fields. Recognized rows accepted by no selected policy are out of scope.
  Empty membership evidence leaves content identification available.
- Before opening, explicit precedence between selected capabilities assigns matched rows to
  disjoint concrete groups. An instrument can supersede LMT processing on shared
  telescope rows without enabling LMT processing for other rows.
- Direct concrete and composite access use the same processing contracts.
  Overlapping memberships must not duplicate derivation, classification, opening
  or projection. One reader can supply shared telescope metadata and its associated
  instrument classification independently of registration order.
- Matching groups are recomputed from current values for each operation, not retained
  as another routing state. Filename assignment does not establish an exclusive
  live reader. [Content identification](#reader-identification) remains a separate operation.
- Composites and their views select capabilities. Individual concrete policies
  do not. Explicit selection belongs to a reusable view, not the table-bound
  accessor's constructor. A view retains its originating accessor and selection,
  invokes the shared workflow and can only narrow further. It does not mutate
  the table when bound.
- Nested groups expand to distinct declared concrete policies. Grouping adds no
  parent processing pass and enables no undeclared subclasses. An empty selection
  performs no matching and records no match issues.
- Composite group iteration yields disjoint integer row positions and concrete
  accessors over the original full table. Concrete accessors do not expose
  composition iteration.

### Projection and classification

Projection exposes values with consistent physical meaning and storage across
heterogeneous source models. Concrete policies own scientific derivation.
Construction performs the initial derivation.

- Scientific columns use declared units, time representations and coordinate
  components. Different attribute paths or conversions may supply a shared column
  if its meaning and storage agree. Filename registration does not merge projection
  readers. Incompatible destination writes fail during population.
- Concrete filename derivation prepares values without changing the table.
  Computations consume current values, with the filename-only exception in
  [preparation](#preparation-and-current-values).
- Scientific classifications retain instrument meaning. Equal numeric codes from
  different domains are not interchangeable. Each enabled classifier populates its
  own column for every row, including empty tables and foreign or unidentified inputs.
  Unmatched rows receive that classifier's unknown flag. Direct and composite
  access use the same names. There is no combined classification flag.
- Each classifier prepares full-length defaults and computed values for matched
  groups. The workflow stages only destination columns and assigns them after
  every preparation and destination conversion succeeds. Failure leaves those
  destinations unchanged. Filename projection and matching diagnostics remain available.
  This guarantee does not cover source preparation or the whole workflow.
  Unselected classifier columns remain unchanged.
- Successful header identification replaces contradictory filename classifications
  with the corresponding instrument's unknown flag. Other collected classifications
  remain until an accepting reader supplies their replacement.

### Metadata reads and resource scopes

Metadata enrichment replaces preliminary values with authoritative information.
Opening provides live resources and readers without implicitly reading metadata.

| Construction mode | Rows eligible for metadata reading |
| --- | --- |
| `guess` | None. |
| `guess_or_read` | Rows whose filename information identifies no memberships. |
| `read` | Rows accepted by the selected capabilities and rows without identified memberships. |

- Recognized out-of-scope rows retain guesses without opening in either reading
  mode. Each later explicit metadata read processes all currently eligible rows,
  regardless of earlier population. Eligibility comes from current columns,
  never trace history. Construction and explicit reads share one read/projection operation.
- Table opening acquires resources from each row's source. It uses current values
  with the selected concrete policy's opener and reader rules. These policies must
  agree on supported object types. A location opens once per shared callable within
  a scope. Borrowed objects bypass opening.
- Unidentified rows try selected policies in declared precedence order, reusing
  successfully opened resources. An opener failure does not try another rule
  within that policy. The [reader stopping rule](#reader-identification) applies
  once a class is selected.
- The scope exposes temporary format-level payloads and live domain accessors.
  Metadata reads inside it reuse those accessors. Reads outside it acquire and
  close owned resources themselves. Table operations construct their selected
  accessors directly, retaining those instances' caches. Separately obtained
  named xarray accessors are not implicitly shared.
- On exit, owned resources close. Temporary resource/reader columns are removed
  or restored to enclosing-scope values. Readers in an enclosing scope can be
  reused. Reader selection does not persist beyond the resource scope. Reopening
  selects and binds again from current table values. Metadata and traces remain.
- Metadata supplies complete observation values, including computed UIDs. Additional
  live-reader values supply classifications where needed. Together they replace
  the guessed subset, including derived columns, for recognized, selected rows.
  Replacements include authoritative nulls.
  Existing locations, filename provenance and unselected rows remain unchanged.
- Each update covers only destinations declared for the returned metadata and
  the concrete reader-column hook. Omitted destinations retain their values.
  There is no filename derivation after metadata projection. Processed KIDs metadata
  projects its recorded parent reference under the [existing provenance contract](#native-processed-kids-metadata).
- Metadata and classification replacement form one update. Identification, reading,
  projection and destination conversion must succeed before the row's update
  becomes visible. Only destination columns are staged. Failed rows retain existing
  values, including guessed classifications. Diagnostics are independent of that update.
- Projection and dtype-contract failures raise before destination columns change,
  without projection retry. Fail-fast metadata inspection leaves destination values
  unchanged. Traces remain available. A successful shared-reader result can correct
  membership and classification without reopening or reidentifying the payload.

### Queries

Queries select from the current collected information, including useful partial rows.

- Membership queries match complete data-model names. Missing or empty memberships
  do not match. [Preparation](#preparation-and-current-values) distinguishes current
  memberships from retained filename evidence.
- Queries combine instrument-specific classification masks without mixing their
  flag domains. Observation queries follow [actor-owned interpretation](#actor-owned-interpretation).

### Failures and traces

Batch operations retain per-source issues for diagnosis without treating trace
history as current routing or readiness information.

| Condition | Batch behavior |
| --- | --- |
| Source validation, recognized filename conversion, resource opening, reader creation or metadata reading fails | Record per row by default and continue, retaining collected information. Explicit raise mode propagates the failure. |
| Filepath, file-accessor or opener rules have no match | Informational issue, including in raise mode. Unknown and foreign memberships use the same file-accessor miss kind. |
| Reader value rules have no class | Try content identification. Successful fallback records no issue. |
| No live reader can be obtained | One reader-creation failure covers exhausted selection, inspection errors and construction errors. Apply record-or-raise policy. |
| Registration ambiguity, projection/dtype-contract failure or an unexpected programming failure | Always raise. Do not convert it to an identification miss. |

- This policy applies to construction, explicit metadata enrichment and scoped
  opening. Single-source operations retain their [direct-call behavior](#format-opening).
  Successful identification does not guarantee readable metadata. An unfamiliar
  filename alone is not a failed metadata read.
- Opening, reader creation and metadata reading have distinct failure kinds.
  Reader failures follow the [stopping rule](#reader-identification). An identification
  miss cannot replace previously collected values.
- A trace retains the latest issue of each kind. Repetition replaces its details,
  and successful retries do not clear earlier issues. Successful results belong
  in information columns, not duplicate trace entries.
- Entries carry an enumerated kind, optional explanation and optional structured
  context. Failure context may describe exceptions, but must not retain live
  resources. Context supports in-memory inspection and logging without requiring JSON compatibility.
- Selection and reordering retain each row's trace object. Recording an issue
  updates its failure status without rescanning entries. All issues and failures
  alone can be iterated in order of each kind's first occurrence. Summaries count
  selected rows containing each kind. Batch operations own error policy, not traces.

## Observation selection

Observation selectors state constraints before expensive reading or processing.
Parsing is context-free. Filesystem, collection and database consumers preserve
the same observation constraint meanings through their own interpretations.

### Constraints and positions

Constraints describe values, alternatives, ranges and positions in an observation hierarchy.

- Alternatives within a field combine with OR. Constraints across fields combine
  with AND. All value constraints apply before positional constraints.
- Positions select eligible ordered distinct values, not file arrival order or
  observation timestamps. Candidate discovery supplies records for the full selection.

### Actor-owned interpretation

The actor applying a selector supplies the field layout and label interpretations
needed for that operation.

- Instrument label-to-number declarations are programmatic, not user configuration.
  Combining contexts creates a new context without changing its inputs and rejects
  conflicting meanings for the same field and label. Known interpretations do not
  prohibit other literal labels.
- Selected file workflows supply selector classes and label interpretations.
  Direct and composite queries use their common selector class when one exists,
  including validation of parsed selectors. Without a common class, queries require
  a parsed selector that supplies its own layout. Different classes do not prevent
  table construction or source access.
- Table positional selection runs once with the combined observation context.
  [Stores](#candidate-discovery) interpret only local values and do not apply positions.

## File stores

Stores expose organized acquisition trees without requiring knowledge of instrument
directory layouts. They return candidate paths without opening data or collecting metadata.

### Declared locations and substores

Each store owns explicit local locations and delegates explicit child locations.

- Substore registration keys a child by its concrete class and assigns one relative
  descendant subpath. Typed lookup reuses that child instance. Parent construction
  creates registered children without filesystem discovery and exposes a read-only mapping.
- Later registrations affect new stores only. Existing instances and views retain
  their constructed trees. Registering an existing class raises, including at the
  same normalized path, without changing paths or descriptors. Path validation is
  lexical and neither inspects the filesystem nor resolves symlinks.
- A store searches its root locally only when explicitly managed. Local search
  can coexist with delegated descendants but must exclude files owned by those
  substores. Traversal resolves declarations without filesystem inspection.

### Candidate discovery

Coarse searches preserve candidates required by the full observation selector.

- The entry store parses text once and passes the parsed selector unchanged to
  descendants. Registration does not constrain child selector types. Unsupported
  stores skip local search but still visit descendants. Invalid text or failure
  handling a supported selector raises rather than returning an empty result.
- Each store owns its selector type and local value interpretation. Delegated
  stores use their own interpretations. Stores neither combine contexts nor apply
  positional selection.
- Concrete stores should use fixed-depth globs without recursive wildcards for
  owned layouts. TolTEC manages its root with fixed-depth master/interface globs,
  constrained by ObsSpec. Views select declared paths or whole substores without
  multiplying equivalent glob searches. Glob enumeration order is retained within locations.
- Capability requirements select matching stores' local locations independently
  of children. Substore descriptors instead select whole child branches. Selected
  descendants remain reachable through unselected or selector-incompatible parents.
  Unrestricted searches visit all registered stores. Stores own their directory layouts.

### Partial views, patterns and paths

Named selections expose the declared tree without requiring callers to reproduce its paths.

- Class descriptors provide unbound selections and instance descriptors provide
  bound views. Partial selection includes a union or subtracts exclusions, never
  both in one call. A call must specify one form. Chained descriptors intersect
  constraints and cannot restore excluded paths. Views do not mutate stores.
- Views expose candidate glob patterns relative to the entry-store root without
  filesystem access. Only a complete concrete directory selection exposes a path.
  An arbitrary union need not identify one directory.
- Candidate globbing preserves the entry-root spelling. Results retain the first
  occurrence of each path without resolving filesystem aliases.

## Input discovery

Configured discovery builds populated, selected file tables from stores and explicit
sources. Entries can span instruments and stores while preserving their selection meanings.

### Input forms and configuration

Discovery settings define how each independent entry obtains candidate sources.

- Literal locations and glob searches are distinct input forms.
- Extensions register a settings model under one identifying key before consumers
  compile enclosing configuration schemas. Reusing a key or model raises, even
  for identical registration, without damaging the previously usable schema.

### Capability selection

A data-model set defines capabilities available to the consuming task.

- Global and per-entry selections can narrow that set but cannot enable support
  outside it. Without an explicit set, discovery uses registered capabilities.
- Discovery searches the union of selected models' required store classes, including
  companions. A shared store is searched once per tree location. Store selection
  does not determine record interpretation or filter explicitly supplied sources.
- An empty selection searches no stores but can still populate filename information
  for explicit sources. Each entry populates its table with its selected capabilities.

### Per-entry selection and combination

Each entry completes preparation and observation selection before combination.

- The configured accessor interprets the populated candidate table. Each entry
  applies its full selector once to that combined table, including positions
  after all value constraints.
- Combining entries retains their capability choices. Unselected classifiers stay
  unpopulated for those entries, without recomputation after concatenation.
- Selected duplicates retain their first selected occurrence. Rows without validated
  locations remain distinct and keep their diagnostics.
- Discovery returns the combined, populated and selected model table. Its consumer
  owns subsequent processing selection and execution.

## Instrument extensions

Extensions supply domain interpretation for the shared observation-data workflow.
Each instrument owns its naming conventions, classifications, recorded-data interpretation
and store conventions.

### Concrete and composite policies

Extensions implement shared contracts through distinct policy responsibilities.

| Component | Responsibility |
| --- | --- |
| Core file accessor | Source validation, universal filename projection, marked metadata initialization and table-local traces. It does not evaluate capability predicates or derive instrument values itself. |
| Concrete workflow | One interpretation's matching, opener/reader rules, metadata mappings, derivation and ObsSpec interpretation. It operates independently of composition and data-model registration. |
| Composite workflow | Component enumeration, precedence, disjoint row assignment, and combined query symbols and ObsSpec contexts. |
| Shared opening and metadata operations | Dispatch, resource lifetime and staged updates for both direct and composite use. Composition adds no second operation path. |

- Data-model registration is the sole registry for workflow capabilities.
  Specifications name accessor classes, and public capability selection uses
  accessor types. [Assembly](#registration-and-import-boundaries) must not introduce
  a reverse dependency from concrete capabilities to specifications.
- Extension policies follow the same interpretation contracts whether selected
  by registry or concrete class. They implement [file-table processing](#file-tables),
  [reader identification](#reader-identification), [selection](#observation-selection)
  and [store ownership](#file-stores) without redefining those guarantees.
- Shared workflows require no classification columns, flag enums or particular
  scientific column dtype. Concrete policies supply these when needed.
- Shared field types follow [value validation](#authority-and-validation). Extension
  setup uses supported building blocks under the [import contract](#registration-and-import-boundaries).
  [Architecture](../architecture.md#observation-access) records implementation wiring.
