# TolTECA specification

## Purpose and scope

- TolTECA supports scientific inspection and processing of LMT and TolTEC
  observations, from recorded acquisitions to reusable results.
- Observation identity, physical meaning, and provenance remain interpretable
  across data access, analysis, and publication.
- Scientific libraries support interactive use independently of file discovery
  and configured workflow execution.
- Database, Web, deployment, and archive application behavior is outside this
  scope. TolTECA defines the data contracts it supplies to those applications.

## Observation data

- Observation data access makes recorded measurements and their context usable
  without requiring scientific reduction.
- Source access owns how data is reached. Instrument interpretation owns what
  the recorded data means. Scientific processing owns derived results.
- Individual acquisition access and collection preparation are independent entry
  paths. A collection may start with filenames, explicit sources, or a declared
  file store. Its consumer selects records before scientific processing.

The [observation-data guides](../docs/datamodels.md) follow this domain's progression:
individual data and metadata, model tables, observation constraints, file stores,
configured discovery, and capability selection and extensions. They connect these
contracts to pipeline input.
Software concepts are defined within their owning sections below and introduced
at their point of use in the guide. The [shared scientific terminology](../../tolteca_helpdesk/handbook/glossary.md)
distinguishes observing and scientific scopes across packages; it does not own
API availability or replace the domain contracts.

### Sources and metadata

- Source access supports inspecting individual acquisitions and interpreting
  batches without repeating storage-specific work for each file.

| Term | Meaning |
| --- | --- |
| Source | Resource to inspect: a location or an already-open object (`source`). |
| Source location | Location from which the data is accessed. |
| Identification location | Filename or path used to interpret acquisition naming conventions (`source_loc`); it may differ from the actual source location. |
| Filename information | Preliminary acquisition information inferred from a naming convention. |
| Acquisition metadata | Recorded information describing the measurement and its observing context. |

#### Source recognition and access

- Source recognition supports finding and inspecting data before reading its
  scientific contents.
- Filename information supports preliminary selection without opening the source.
- Single-source preparation validates the resource and its identification
  location, then parses the filename eagerly. It raises on invalid input;
  an unfamiliar filename still provides its extension and an empty set of
  identified data-model memberships. Batch
  preparation records per-input failures by default and retains one row per
  input. Invalid resource/location pairs have missing information and a failure
  trace; filename failures retain the validated pair. Explicit raise mode stops
  on these failures. Registry ambiguity and projection errors remain fatal.
- The batch workflow owns tracing and error-recording policy. Single-source
  preparation and opening return results or raise; they do not collect history.
- Standalone source opening uses filename information with format-opening rules
  and manages newly opened resources until context exit. Borrowed objects remain
  caller-owned. A caller may supply an opening policy for an individual operation;
  this requires no source-model subclass or data-model selection. Opening yields
  a format-level resource without reader identification or metadata collection.
  Table opening uses current table values with those rules rather than delegating
  to a single-source object's original filename information.
- The universal filename factory uses all registered naming conventions,
  independently of collection scope. A factory invoked on a concrete model
  considers only that model and its descendants, never ancestors or siblings.
  The most specialized eligible matching model wins; unresolved
  ambiguity raises before field conversion, including in batch error-recording
  mode. Concrete filename models also support annotations and instance checks.
  Source preparation invokes the universal factory, not a concrete model's factory.
  The basic filename parser participates in the same registration and
  specialization rules as instrument parsers.
  General telescope names establish LMT production and LMT data-model membership.
  Registered instrument conventions add membership for their telescope companions.
- Data in a supported format remains accessible when its filename is unfamiliar
  or absent, including data supplied as an open object.
- Format access is independent of instrument interpretation. Opening a supported
  source does not require identifying its instrument first.
- Tollan supplies generic record-rule matching. Tolteca specializes callable
  rules for source opening and supplies default format callables. File accessors
  supply matching records and sources. Filename parsing alone supplies extension
  information; opening must not derive it independently.
  Format callables own location support and preparation. Table acquisition
  passes the original source to the selected callable and manages its lifetime.
- Source-opening policies pass the explicit source unchanged to the selected
  callable. Matching alone must not open sources.
- Batch opening selects callables from table values without acquiring resources.
  Iteration opens in row order and transfers ownership of each yielded object
  to the caller before opening the next. Failure or early termination must not
  close already-yielded objects or open later sources.
- No match raises when direct opening is requested. A selected callable's
  failure propagates without falling through to another rule.
- Table construction, explicit metadata enrichment, and scoped opening apply
  the same error policy to acquisition failures, including unsupported sources.
- Each data model may supply its own format-opening policy to obtain objects
  supported by its readers. Policies compose without modifying shared defaults.
  Rules within a policy use first-match ordering, including explicit overrides.
- Each matched row uses its selected concrete policy's opener and reader rules.
  The policy's opener must support the object types its readers accept. A shared
  callable opens a location once within a scope. Borrowed objects bypass opening.
  Unidentified rows try the selected concrete policies in their declared
  precedence order, reusing successfully opened resources. Recognized foreign
  rows are not opened. An opener failure is recorded without trying another rule
  within that policy.
- Format rules consume logical attributes independently of instrument handling.
  Table routing uses current query values. Filename-only attributes, such as
  the extension, use their projected file columns; original duplicates do not
  override current values, including current nulls.
- Typed opening combines a known format opener with its known accessor, without
  reader selection. Failed binding or classification closes the newly opened
  resource; success transfers ownership to the caller without requesting full
  metadata. Already-open objects with a known reader use that accessor directly.
- Current collected information may select a reader directly. Construction
  applies the accessor's own validation; metadata reads remain authoritative.
- Recognition does not require complete acquisition metadata or signal arrays.
- Each data model declares ordered reader-selection rules whose values are
  accessor classes. Rules use current query values, not original filename
  snapshots; collection constraints test membership in `data_models`.
- Content identification extracts routing values through the data model's
  mappers and applies the same rules. It must not trial accessor construction,
  read full metadata, or use a parallel set of acceptance predicates.
- Instrument workflows constrain the accepted reader scopes. These workflow
  scopes may overlap: general telescope access accepts telescope acquisitions,
  while TolTEC access accepts its telescope companions, KIDs and HWPR data.
  Both use the same reader-resolution operation. Current membership controls
  applicable data models. Content identification can recover unfamiliar filenames.
- Matching partitions rows into disjoint concrete-policy groups before opening.
  Each group selects, binds and reads its rows through that policy. Composite
  workflows use the same operation as direct concrete access; they do not repeat
  projection or classification for overlapping data-model memberships.
- Once a unique reader is selected, binding or metadata failure must not trigger
  another interpretation. Content identification is the fallback for a missing
  rule result, not recovery from a selected reader's failure.

#### Acquisition metadata

- Metadata access supplies validated records for selection and interpretation.
- Recorded metadata supplies authoritative values; filename information remains
  available as separate provenance.
- A recorded original acquisition path remains metadata provenance, distinct
  from the current source location.
- Missing required acquisition facts are reported; guesses do not complete a
  metadata record.
- Successful identification does not guarantee readable metadata. Batch access
  records data-reading failures by default, retains the affected rows and their
  previously collected information, and continues with other rows. Explicit
  fail-fast access propagates those failures. Neither mode tries a different
  reader after the selected reader fails. Registration ambiguities and unexpected
  programming failures propagate in both modes.
- A row's metadata enrichment and classification replacement form one update.
  Identification, reading, projection, and destination conversion must succeed
  before the table publishes that update. Failed rows retain their collected
  values. Recording a diagnostic is independent of that update. Data-reading failures are recorded per row. Projection and dtype-contract
  failures always raise before destination columns change; no projection retry
  is performed. Explicit fail-fast metadata inspection leaves destination values
  unchanged. Traces remain available in either mode.
- An identification miss retains previously collected values, including guessed
  classifications, in error-recording mode. Explicit fail-fast access raises on
  an unrecognized opened object during construction, metadata inspection, or
  scoped opening. An unrecognized filename alone remains a nonfatal diagnostic.
  Only a recognized reader result can replace collected values.
- Projection replaces only the destinations declared for the returned metadata
  type and the concrete reader-column hook. Omitted destinations retain their
  existing values. No filename derivation runs after metadata projection.
- Each explicit metadata read processes eligible rows, including rows populated
  during construction or by an earlier read. An open scope exposes live resources
  and readers temporarily without reading metadata. Metadata reads inside that
  scope reuse its readers; outside it, they open the required resources and close
  owned resources afterward. After the scope, metadata and trace values remain.
- Metadata inspection does not require reading unrelated scientific data.
- A reader describes an acquisition with stable identity and layout. Tracking
  subsequent edits to its recorded metadata is outside that reader's scope.

#### Native processed KIDs metadata

- Native processed KIDs files expose the acquisition source recorded by the writer.
  Reading this reference does not access the acquisition. Its filename can supply
  preliminary identity independently of the processed file's name or location.
- Raw acquisition settings require the acquisition headers. When no acquisition
  source was recorded, the processed reader still supplies its classification and
  the table records unavailable metadata while retaining earlier values.

#### Batch layouts and semantic access

- Files in a batch from the same writer share stable layout conventions. The
  schema/mapper/view system uses those conventions for semantic access across
  the batch.
- Layout identifies the kinds and locations of data entities. Shared layout does
  not imply equal acquisition values, identities, or sample counts.

| Concept | Responsibility in acquisition access |
| --- | --- |
| Schema | Give recorded entities a consistent scientific vocabulary. |
| Resolved mapper | Bind that vocabulary to a source layout for reuse across compatible sources. |
| Accessor or view | Apply a mapper to a particular source and expose its data through a domain API. |

- Reader design supports and encourages resolved-mapper reuse across a batch.
  A mapper resolved for one file can serve other files with the same layout or
  compatible subsets and extensions.
- Compatibility depends on the bindings required by the requested operation.
  Complete layouts need not be identical; extra fields or omitted unused fields
  do not prevent reuse.
- Reusing a mapper preserves each source's own values and identity. Layout
  bindings do not substitute data from the file used to resolve them.
- Resolution and reuse follow
  [Tollan's mapping contract](../../tollan/design/spec.md#schemas-and-mappings).

#### Data access and resource ownership

- Data access supports inspecting selected measurements without loading the
  entire acquisition when that is unnecessary.
- Selection and reordering preserve each selected element's correspondence to
  the source acquisition. Local array or table positions do not replace source
  identifiers.
- Deferred reads require the source to remain available. The operation that opens
  a resource owns its cleanup, including on failure; supplied resources are borrowed.
- Views provide read-only access to their bound source and descendants. Parent
  and sibling data are outside that view's scope.
- The component that changes a source manages affected view lifetimes. Views do
  not detect edits, refresh cached state, or rebind themselves, under
  [Tollan's view contract](../../tollan/design/spec.md#accessors-and-views).

### LMT observations

- LMT observation conventions connect telescope state and acquisition files
  across participating instruments.
- An observation can contain several acquisitions and producing interfaces.
  Shared identity relates their records without requiring a shared file or location.
- Related interfaces can have independent sampling. Shared observation identity
  does not establish time alignment; joint analysis owns that alignment.

#### Observation and interface identity

- Observation identity connects records from the same observing activity.
- Interface identity distinguishes the producing instrument from the data models
  applicable to interpretation. Data-model memberships may overlap.

| Term | Identifier or field | Meaning |
| --- | --- | --- |
| Observation | `uid_obs` | Acquisitions sharing one observation number within the master context. Master-qualified form: `{master}-{obsnum}`. |
| Acquisition, or raw observation | `uid_raw_obs` | One recorded acquisition within an observation. Master-qualified form: `{master}-{obsnum}-{subobsnum}-{scannum}`. |
| Quartet | `master`, `obsnum`, `subobsnum`, `scannum` | Acquisition controller, observation number, sub-observation number, and scan number. |
| Acquisition file | `uid_raw_obs_file` | A file belonging to an acquisition; its identity includes the constituent interface. |
| Acquisition master | `master` | Controller associated with the acquisition sequence. |
| Producing instrument | `instru` | Instrument that generated the data. |
| Data-model membership | `data_models` | Applicable data-model names; membership order establishes no precedence. |
| Interface class | `interface_kind` | Kind of producing interface, such as `roach`, `tel`, or `hwpr`. |
| Interface instance | `interface` | Particular producing interface, such as `toltec7`, `tel`, or `hwpr`. |

- Identifiers may omit an unknown master. Such identifiers are context-relative,
  not unique across masters. No default master is inferred.
- An observation UID requires `obsnum`. A raw-observation UID requires all three
  numeric components. A raw-observation file UID also requires `interface`.
  Missing required components yield no UID.
- Quartet formatting permits an omitted master and omitted trailing numeric
  components, but still requires `obsnum`. A numeric gap yields no identifier;
  later components must not shift position. UID constructors enforce the
  completeness required for their scope.
- Database ingestion and observation product identity require a known master
  and the complete numeric identity for their scope.
- Acquisition metadata supplies complete interface identity: producing
  instrument, applicable data models, interface class, and interface instance.
- Metadata models require explicit identity values, including fields constrained
  to a single value. Mappers supply those values when constructing metadata;
  type constraints do not imply defaults.
- Canonical identifiers describe observation and acquisition identity independently
  of where the data is stored.

#### Telescope telemetry

- Telescope access supplies observing context and pointing measurements for
  inspection and joint analysis with data from other instruments.
- Telescope recognition is independent of receiver-specific restrictions.
  Instrument data models identify their telescope companions from recorded metadata.
- Telescope records have `instru="lmt"` and include `"lmt"` in `data_models`.
  The receiver header supplies additional membership, such as `"toltec"`,
  without changing the producing instrument.

### TolTEC observations

- TolTEC observation access supports inspecting related acquisition streams under
  the shared [source and metadata](#sources-and-metadata) and
  [LMT identity](#observation-and-interface-identity) contracts.

| Area | Role in a TolTEC observation |
| --- | --- |
| KIDs readout | Detector data for sweep and timestream inspection and processing. |
| Half-wave plate readout | Modulation data for joint analysis with detector observations. |
| Telescope companion | LMT-produced telescope data belonging to both the LMT and TolTEC data models. |

### Model tables

- Model tables provide a columnar workspace for collecting information,
  enriching it, and selecting records for further work.
- Generic projection and enrichment follow
  [Tollan's model-table contract](../../tollan/design/spec.md#model-tables).
- Factories prepare source records; metadata reads update their current values.
  Binding an accessor or narrowing its capabilities does not repopulate the table.
  Row iteration and queries support inspection and downstream processing of the
  prepared records. Row positions must remain stable while an operation uses them.
- A row represents a collected record. Columns expose the attributes needed for
  inspection and selection; they need not reproduce a complete source record.
- File, observation, and product collections retain their own record meanings.

#### Mixed and scoped collections

- Source collections support mixed data and work within selected data models.
  Data-model scope selects interpretation capabilities, not producing instruments.

| Collection | Interpretation |
| --- | --- |
| Mixed | Use all enabled data-model support, including overlapping memberships. |
| Known scope | Use selected data-model support, including its supported companion data. |

- These are the collection forms of the
  [shared and specialized workflows](#instrument-extensions). Both follow the
  same projection, enrichment, and selection contracts.
- Explicit interpretation selection belongs to a reusable file-accessor view,
  not the table-bound accessor's constructor. The view retains its originating
  accessor and selected capabilities. It invokes the same shared workflow with
  that selection, rather than implementing a second workflow. Binding either an
  accessor or a view does not mutate or populate the table. Further view selection
  cannot widen its capabilities.
- Scientific classifications retain their instrument meaning in mixed collections.
  Equal numeric codes from different domains are not interchangeable.
  Each enabled instrument classifier populates its own column for every row.
  Unmatched rows receive that classifier's unknown flag, including foreign and
  unidentified inputs. Shared and specialized workflows use the same column names.
  There is no combined classification flag; queries combine per-instrument masks.
  Successful header identification replaces contradictory filename classifications
  with the corresponding instrument's unknown flag.
- Collection scope controls interpretation; it does not itself discard
  explicitly supplied sources or suppress registered filename information.
- Current data-model memberships determine scope at every processing stage.
  Parsed memberships are immutable sets. Rows with nonempty memberships are
  matched through the selected concrete policies' rules, which may also constrain
  prepared interface fields. Rows no selected policy accepts are out of scope.
  An empty filename-derived set supplies no membership evidence. Missing table
  cells after preparation failure are not an additional model-field value.
- Each row retains its original filename parse result, or absence of a successful
  parse, through selection, reordering and metadata enrichment. The original
  parse result records filename recognition, not current scope.
- Construction validates supplied source/location pairs and installs them in
  a new table, retaining one row per input. Guessing consumes the table's existing
  source locations and retains the resulting filename parses. It does not accept
  replacement sources or a separate source-to-row alignment. Concrete operations
  receive disjoint positions from capability matching on the current table.
- Filename-only population reads no sources. Conditional population attempts
  metadata reads only when filename information identifies no memberships.
  Full population attempts reads for in-scope rows and rows without identified
  memberships. Recognized out-of-scope rows
  retain their guesses without being opened in either reading mode.
- Later metadata inspection uses the current membership and identity columns.
  Filename snapshots remain provenance. A successful shared-reader result may
  correct membership and classification without reopening or reidentifying the
  payload. Projection and classification still publish together.

#### Projection and queries

- Selected workflows supply observation label interpretations and a selector
  class. Queries use their common selector class when one exists, including
  validation of parsed selectors. Direct and composite access follow the same
  rule. A selection without a common class requires a parsed selector, which
  supplies its own field layout. Different selector classes do not prevent table
  construction or source access. Positional selection runs once using the
  combined observation context.
- Projection makes source information available for comparison and selection
  across a collection.
- File-table integration owns filename projections. Filename models and parser
  registrations must remain usable without constructing table mappings.
  Filename fields may mark values that represent data metadata. Projection retains
  filename snapshots and initializes the marked metadata columns before capability
  matching, independent of the selected capabilities. Markers select columns,
  not rows: each alias copies its corresponding filename column.
  Existing current columns,
  including nulls, are preserved. Filename models do not own table mappings.
- Scientific columns use declared units, time representations, and coordinate
  components so that query values retain a defined physical meaning.
- A shared filename or metadata column has consistent meaning and storage.
  Separate input projections may supply it through different attribute paths
  and conversions. Filename registration does not merge projection readers;
  incompatible destination writes fail during population.
- Capability matching consumes prepared current columns. It must not initialize
  metadata or substitute filename snapshots for missing or null current values.
  Concrete filename derivation prepares column values without modifying the
  table. Subsequent computations consume current values. A concrete classifier
  may explicitly use filename-only attributes for which no current column exists.
- Each enabled classifier prepares its full-length default column, including
  for empty inputs and unmatched rows. Matched groups supply their computed
  values. The shared workflow stages only destination columns and assigns them
  after all defaults and group updates succeed. A preparation or dtype-assignment
  failure leaves derived destinations unchanged; filename projection and matching
  diagnostics remain available. This is not a transaction around source
  preparation or the entire workflow. Unselected classifiers are unchanged.
- Queries and row iteration use current column values.
- Membership queries match complete data-model names. Missing and empty
  memberships do not match. Filename memberships remain available separately
  when header metadata replaces the current memberships.
- Partial rows remain useful for preliminary selection, including resources
  that have not been opened. A partial table row is not complete acquisition metadata.

#### Metadata enrichment

- Enrichment adds authoritative information to a collection prepared from
  filenames, discovery results, or other supplied records.
- Guessed acquisition information, including derived columns, is a subset of
  the authoritative information that metadata enrichment supplies.
- Enrichment replaces preliminary values with authoritative metadata for the
  selected records, including authoritative absence of a value.
- Each metadata projection applies only to records recognized by its concrete
  accessor. Successful metadata projection replaces values covered by the selected
  accessors, including values absent from the recovered metadata.
- Metadata supplies complete observation values, including computed UIDs.
  Additional live-reader values supply classifications where needed. Both
  projections are staged using only their destination columns and published
  together after the complete row succeeds; failures leave it unchanged.
  Initial derivation runs during table construction, not after metadata projection.
  There is no public re-guess operation. TolTEC classification is a concrete policy:
  existing collected classifications remain until an accepting reader supplies
  a replacement. Processed KIDs metadata projects the recorded parent reference.
- Records outside the selection retain their existing values.
- Enrichment preserves filename provenance and actual/identification locations.
- Opening a file table acquires and identifies resources from each row's source.
  Metadata population is a separate operation. The resource scope
  exposes both the format-level payload and the selected domain accessor. A matched
  row uses its assigned workflow. An unidentified row tries the selected workflows
  in precedence order until a reader is obtained. A selected reader's construction
  failure stops that search; it is not a missing match.
- Explicit metadata enrichment reuses live accessors when available. Construction
  modes and later enrichment share the same live-accessor read and projection
  operation. A failed read or projection leaves the row's information unchanged.
- File-table accessors are constructed directly from their selected classes.
  Live `accessor_obj` values retain per-accessor caches. A separately obtained
  named xarray accessor is not implicitly shared with a new table operation.
- Each explicit metadata read processes all eligible rows, regardless of earlier
  table population. Table construction chooses rows according to its mode:
  guess reads none, guess-or-read reads unidentified rows, and read includes
  matching and unidentified rows. Both entry paths use the same read and
  projection operation. Failed reads retain existing metadata values.
- Per-source traces retain the latest processing issue of each kind, including
  non-error identification misses. Repeated issues replace the same kind's
  details; successful retries do not clear recorded issues. Each entry has an
  enumerated kind, optional
  explanatory text and optional structured context. Failure context may describe
  exceptions; entries must not retain live resources. Successful results remain
  in the information columns rather than duplicated trace entries.
  Context supports in-memory inspection and logging; it need not be JSON-compatible.
- A missing match among filepath, file-accessor or opener rules is
  informational, including in raise mode. File-accessor matching reports the
  same kind of miss for unknown and foreign data-model memberships. Read-mode
  routing uses current information columns, not trace history.
- Reader creation matches current values, falls back to content identification
  when no class matches, and constructs the selected accessor. Failure to obtain
  a live reader records one `reader_creation_failed`, including exhausted
  selection, inspection errors and construction errors. The operation follows
  `on_error`; successful fallback records no issue. Construction failure does
  not try another reader. Resource opening and metadata reading retain their
  separate `source_open_failed` and `metadata_read_failed` kinds.
- A row retains its trace object when selected or reordered. The trace exposes
  whether its entries contain a failure; recording an issue keeps that value
  consistent without rescanning other entries. All issues and failures alone
  can be iterated separately in order of each kind's first occurrence.
  Error handling belongs to the
  batch operation, not the row trace. Trace entries are internally constructed
  records, not external-input validation models.
  Summaries count rows containing each issue kind among the selected rows.
- Owned resources close when their scope ends. Temporary resource and reader
  columns are removed on exit, or restored to an enclosing scope's values.
  Live readers may be reused within that enclosing scope. Reader selections
  are not retained after the resource scope ends; reopening selects and binds
  readers again using current table values.

### Observation selection

- Selection expresses which observations or interfaces are relevant before
  expensive reading or processing.
- An observation selector states constraints on observation and interface fields.
  Candidate discovery finds the sources to which those constraints apply.
- The same observation constraint retains its meaning across collection,
  filesystem, and database consumers.

#### Constraints and interpretation

- Constraints describe a desired subset using field values, alternatives, ranges,
  and positions within the observation hierarchy.
- Alternatives within a field combine with OR; constraints across fields combine
  with AND.
- All value constraints apply before positional constraints.
- Positions select eligible ordered distinct values, not file arrival order or
  observation timestamps.
- Instrument extensions supply label-to-number interpretations for selection.
  These are code-defined declarations, not user configuration. Composition
  creates a new context without changing its inputs and rejects conflicting
  interpretations of the same field and label.
- Known label interpretations do not prohibit matching other literal labels.

### File stores

- File stores expose organized acquisition trees without requiring callers to
  know each instrument's directory layout. They return candidate paths without
  opening data or collecting metadata.

#### Candidate discovery

- Coarse store searches retain candidates needed by the full selector.
- A substore registration identifies a child by its concrete class and assigns
  that class one relative descendant subpath within the parent. Class-based
  lookup returns the corresponding concrete store type and reuses its instance.
  Parent construction creates its registered child stores without filesystem
  discovery. The child mapping is read-only. Later registrations affect only new
  stores; existing instances and their views retain the constructed tree.
  Registering an existing class must raise, including at the same normalized
  subpath, without changing its path or descriptors. Subpath validation is lexical
  and does not inspect the filesystem or resolve symlinks.
- Registration does not constrain the child's selector type. Discovery parses
  text at the entry store and passes the parsed selector unchanged to descendants.
  Each store searches local files only if it supports that selector; unsupported
  stores still visit their descendants. Invalid text and failures while handling
  a supported selector raise rather than produce an empty result.
- Each concrete store owns its selector type and local value interpretations.
  Delegated stores resolve candidates using their own interpretations. Stores
  do not combine interpretations or apply positional selection.
- Each store declares directly managed locations and delegated substores.
  The store's root is searched locally only when explicitly managed. A store
  may manage its root while also delegating descendants; local searches must
  exclude files owned by those substores. Concrete stores should use fixed-depth
  globs without recursive wildcards to express their owned layouts.
  Traversal resolves these declarations without inspecting the filesystem.
- Python callers select declared locations and substores through named
  descriptors, without spelling their filesystem layout. Class descriptors
  provide unbound selections; instance descriptors provide bound views.
  Partial selections include a union or subtract exclusions, never both in one
  call. A partial selection must specify one form.
  Chained descriptors intersect constraints and cannot restore excluded paths.
  Views do not mutate stores.
- TolTEC manages its root and renders fixed-depth master/interface globs.
  ObsSpec supplies master/interface constraints; store views select declared
  paths and whole substores. Views must not multiply equivalent glob searches.
  Glob enumeration order is retained within each location.
- Store views expose candidate glob patterns without filesystem access, relative
  to the entry store's root. Only a complete concrete directory selection exposes
  a filesystem path; an arbitrary union need not identify one directory.
  Candidate globbing preserves the entry-root spelling and deduplicates paths
  by first occurrence without resolving filesystem aliases.
- Data-model discovery requirements select each matching store's local locations
  independently of its children. Python substore descriptors select the whole
  child branch. Selected descendants remain reachable through unselected or
  selector-incompatible parents. An unrestricted search visits all registered
  stores. Directory layouts remain the stores' responsibility.

### Input discovery

- Configured input discovery builds a model table from acquisition stores and
  explicit sources for inspection or pipeline input selection. Each entry prepares
  a [mixed or scoped collection](#mixed-and-scoped-collections).
- Literal locations and glob searches are distinct input forms.
- Discovery extensions register a settings model under one identifying key.
  Reusing a key or model must raise, including an identical registration.
  Failed registration must preserve the previously usable configuration schema.
  Extensions must register before consumers compile enclosing configuration schemas.
- Each discovery entry applies its complete selector before entries are combined.
  Selected duplicates retain their first selected occurrence.
  Rows without a validated location remain distinct and retain their diagnostics.
- A discovery collection may span stores and instruments while preserving each
  domain's selection meanings.
- Discovery uses a data-model set to define the capabilities available to
  the consuming task. Global and per-entry selections may narrow that set but
  cannot enable capabilities outside it. Without an explicit set, discovery
  uses registered capabilities.
- Each entry populates its own table with its selected capabilities. Combining
  entries retains those choices: an unselected classifier remains unpopulated
  for that entry, rather than being recomputed after concatenation.
- Each data model declares the store classes needed to discover its inputs,
  including companion data. Discovery searches the union of those requirements
  for the selected models. A shared store is searched once per tree location.
  Store selection does not change the interpretation of discovered records or
  filter explicitly supplied sources. An empty capability selection searches
  no stores but can still populate filename information for explicit sources.
- The configured file accessor owns interpretation of the populated candidate
  table. Each discovery entry applies the full selector once to its combined
  table, including positional selection after all value constraints.
- Discovery returns the combined, populated and selected model table. The
  consuming workflow owns its subsequent processing selection.

### Data-model capabilities

- A data-model specification owns one identity and associates it with reusable
  accessor and store capabilities. Those capabilities need not own that identity.
  Constructing a specification must not enable it globally.
- A capability set selects specifications for a workflow. It does not create a
  composite identity or define a separate multi-model workflow. A one-model set
  uses the same operations and resolution rules as a multi-model set.
- Application entry points assemble first-party specifications. Optional
  extensions activate explicitly; an existing name must not be replaced.
- Core declarations and explicit selections must not load first-party assembly
  or consult its registry. Named application selection snapshots available specs;
  later registrations do not enlarge an existing selection.
- A capability selection is read-only. Narrowing produces a new selection and
  retains each selected identity's association with its capabilities, including
  when multiple identities share an accessor or store.

### Instrument extensions

- Instrument extensions let shared observation-data workflows handle additional
  data models and let specialized workflows select their interpretation support.
- A data-model identity owns its name. A spec associates that identity with
  concrete capabilities; registration and selection use its name without an
  independent alias. Capabilities need not carry an identity binding.
  A component's affiliation is distinct from a record's applicable data models.
  Records may support several data models without requiring inheritance between
  those definitions or duplicating a shared reader.
- A named data-model spec declares its workflow capabilities.
  Its store requirements identify acquisition stores and companion stores;
  they do not enable additional data-model handlers. Its file accessor owns
  table enrichment, metadata readers, and observation
  query interpretation. Defining an accessor subclass does not register a
  data model; direct typed use remains independent of registration.
- A data-model set selects definitions. Registering an existing name
  must raise without modifying the registry. Additional registrations do not
  change existing sets. Importing instrument components may load data-model
  identity information but must not assemble or register workflow capabilities.
  Explicit definition modules assemble their
  completed components without registration. The public data-model, file-table
  and input-discovery entry points register built-in definitions. The filename
  entry point enables parsers without assembling data-model definitions.
  Optional extensions control their own registration entry point.
- Shared extension interfaces live in `tolteca_datamodels.core`. Importing them
  must not load instrument packages or register built-in discovery inputs.
  Feature registries, such as filename parsers, belong to their corresponding
  core feature; instrument packages supply concrete registrations. The assembled
  named data-model registry belongs to the root data-model entry point.
- Root assembly modules expose task entry points, not extension building blocks.
  The data-model entry point provides live named lookup of registered capabilities
  and workflow selection. Core provides specs and sets of explicitly supplied
  specs; the root adds named lookup, registration and selection. Core file access
  uses every available filename registration without importing instruments.
  Root file access adds selected concrete capabilities to the shared workflow.
- Shared workflows require no data-model context from the caller. They use
  registered support to identify applicable layouts and invoke the corresponding
  domain operations for supported inputs.
- Specialized workflows receive a data-model set up front and use it to
  select support within the same workflow. Direct concrete file accessors select
  their declared support without requiring a discovery context.
- Each specialization retains the same interpretation and operation contracts
  whether selected through the registry or by its concrete accessor class.
- Compatible support may contribute to the same collected source. Shared
  telescope metadata and the associated instrument's classification remain
  available together through one selected reader, independently of registration
  order. Filename derivation assigns each matched row to one selected capability,
  with explicit precedence for overlapping support. That assignment does not
  establish an exclusive live reader; content-based identification is separate.
- Each instrument owns its naming conventions, acquisition classifications,
  recorded-data interpretation, and store conventions.

#### File-accessor workflows

- File-accessor extensions supply policies for a single interpretation or a
  composition of interpretations. Both use the shared source preparation,
  opening and metadata-projection contracts.
- The generic core accessor owns source validation, universal filename projection,
  marked metadata initialization and table-local traces. It does not evaluate
  capability predicates or derive instrument values itself. Its public methods
  delegate processing to the configured workflow.
  Filename guessing is an internal construction stage. Public factories prepare
  preliminary information; subsequent enrichment reads authoritative source metadata.
  The base workflow handles one policy: its matching rules, opener/reader rules,
  metadata mappings, derivation hooks and ObsSpec interpretation. Concrete
  workflows specialize that policy and run independently of composition.
  The composite owns component enumeration, precedence and assignment of disjoint
  row subsets over the original table. It also combines the selected policies'
  query symbols and ObsSpec contexts. Public accessor selection still
  uses `accessor_types`; data-model specs continue to name accessor classes.
  `CompositeFileAccessor` adds `partial()` and `iter_accessors()` without
  instrument settings. Group iteration is not part of `FileAccessorBase` or
  concrete accessors; the composite workflow uses its own private matcher
  to yield positions and accessors. Workflows do not expose public iteration.
  An empty capability selection performs no matching and records no match issues.
  Shared row diagnostics and staged column updates retain the same behavior for
  direct and composite use. The assembled root
  accessor selects capabilities from registered data-model specs. Composite
  accessors declare groups of concrete or composite capabilities. Selection
  belongs to composites and their views, not to individual concrete policies.
  Nested groups expand to distinct concrete policies before row matching;
  grouping does not add a parent processing pass or enable undeclared subclasses.
  Concrete workflows supply matching and processing policies and work without
  constructing or registering a spec. Specs associate identities with accessor
  and store classes; those classes do not depend on their specs.
- Matching uses current record values. Explicit precedence between selected
  accessor classes assigns each matched row to one class. An instrument accessor
  can replace LMT processing on their shared telescope rows without enabling
  LMT processing on other rows. Iteration yields disjoint integer row positions
  and concrete accessors over the original full table. Groups are recomputed
  for each operation, not stored as a second source of routing state.
- Shared opening and metadata projection consume the selected row groups and
  policies. They own opener and reader dispatch, resource lifetime, and staging
  metadata before updating the table. Both single and composite workflows use
  these operations. Composition does not add a second opening or projection path.
  Data-model registration remains the sole registry for workflow capabilities.
  The shared workflow does not require classification columns, flag enums, or
  any particular scientific column dtype.

#### Shared data models

- Data models validate values before they are used for interpretation and
  selection. They use strict type validation, are frozen by default, and validate
  declared defaults and default-factory results.
- Models with a different input or mutation protocol declare their exceptions
  explicitly.
- Reusable Pydantic field types own value validation, normalization, and
  serialization. Models enforce additional constraints and relationships
  separately.

## Data products

Users retain product descriptions, artifact references and execution records for
reuse across observing-site, archive and local workspaces. The portable core owns
these records and filesystem namespace/run persistence. Applications supply their
product metadata, identity rules and publication operations. Database projection
and scientific processing remain separate consumers.

### Portable records

- `tolteca_datamodels.core.data_prod` provides reusable records without importing
  instrument registrations, concrete product packages or a database schema.
- A product has a nonempty string UID and an optional namespace UUID. References
  and association targets use the same identity. Directed associations carry
  application-defined relationship labels and facts; database registration is a
  consumer constraint. An empty tag has no selected reference.
- Product records hold identity, typed metadata, items, associations, tags and
  audit facts. Index documents hold the supporting locations, namespace, actors,
  execution snapshot and nested documents. Document context does not become
  product identity. Applications can specialize record metadata and index types.
- Editable records retain strict field and default validation. Shared Tollan
  fields own path normalization and validation. Run names add the store's
  reserved-name constraints.
- Each item references a declared location. Repeated location/path pairs within
  one product are invalid; different products may reference the same source.
  An included namespace must match the product's namespace UUID. An included
  store run must reference the included execution. These checks do not require
  the payload or a database entry to exist.
- Actor and execution mapping keys must equal their records' UUIDs. Namespace
  store-run keys must equal their execution UUIDs, and each store run must
  reference an included execution. Reserved names and assigned counter values
  are unique within one namespace document.
- An inventory identifies an independently maintained set of reported records.
  It has no storage root and does not replace product or namespace identity.

### Sources and persistence

- Each item stores a location key and a relative `SubPath` with literal filename
  characters. A location declares a storage root. Relative roots, including
  parent directories, resolve against the opened index directory; external roots
  use FileLoc's location formats. Application configuration roots do not apply.
- Resolution returns a FileLoc or an encoded absolute source URI without opening
  the payload. Reading, resolving and writing preserve the serialized relative
  locations and item paths. Reopening a relocated index establishes its new
  resolution base. Writing an opened index does not change its existing base.
- YAML persistence uses Pydantic's round-trip representation for declared fields.
  It preserves explicit nulls and restores field types through their existing
  validation contracts. Computed fields are reconstructed, not serialized as
  inputs. Persistence must not relax strict validation. YAML formatting and
  comments are not preserved; Python-specific object tags are rejected.

### Actors and record attribution

- Actors retain client and environment facts reused across executions.
  Invocation-specific configuration, commands, inputs and timing belong to runs.
- Runs and stores share process-local default actors for matching stable runtime
  facts. Namespace creation reuses a matching recorded actor. Supplied and
  imported actors keep their UUIDs and facts, even when another actor has matching
  facts. Run creation remains store-independent; persisted actor reuse is explicit.
- Audit records preserve creation and modification times. Unknown creators and
  modifiers remain independently null; known attribution must not be discarded
  or replaced by the importing actor. Actor records have their own creation time,
  without recursive creator/updater records.

### Namespaces and executions

- A namespace identifies a persistent output scope shared by many executions.
  Each execution has a stable portable UUID independent of destinations. A
  filesystem store operates on one namespace and does not execute the workflow.
- One namespace document carries namespace metadata, actors and optional selected
  run records. Reading namespace metadata does not load run history. Full history
  collection is an explicit export operation.
- The execution owner determines its outcome. Saving a supplied execution does
  not finish it or require captured runtime metadata. Conflicting execution or
  actor facts are rejected before replacing saved facts. A saved completed
  execution is retained when its earlier running snapshot is supplied.
- A store-run record describes one execution's output in a namespace. Its local
  name, counter and per-product outcomes/provenance do not become execution
  identity. Independently obtained writers must preserve each other's updates
  to different outputs of that store run.
- Run reservation uses direct identity, name and counter lookups. Lifecycle and
  output-record updates load and write only the affected run. Per-update work
  must not grow with the number of prior runs.

### Run names and interrupted allocation

- A namespace owns one authoritative allocator and durable counter state.
  Allocation is serialized. Saved reservations are reused for the same execution;
  a conflicting name or naming policy is rejected. Recording an execution alone
  does not require a name reservation.
- Callers may request a fixed name or counter generation. Names and allocated
  counter values are unique within the namespace. Explicit unused counter values
  are allowed; they must not rewind the allocator or reuse a claim. Counter
  values need not be continuous.
- Interrupted allocation may leave claimed names or counter values without a
  saved reservation. Claims remain unavailable; the store does not recover or
  reclaim them. A claimed generated name must also have its counter claimed.
  A retry reuses a saved reservation; otherwise automatic naming allocates a new
  one and a claimed fixed name fails. Allocation does not scan run history.
- A user may remove an abandoned fixed-name claim when no writer is active and
  no saved reservation or published output uses it. This releases only the name;
  it does not rewind the counter or remove completed reservations.
- Copied allocator state does not grant independent allocation authority.
  Imports must not rewind state or duplicate reservations.
