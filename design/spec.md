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

### Shared data models

- Data models validate values before they are used for interpretation and
  selection. They use strict type validation, are frozen by default, and validate
  declared defaults and default-factory results.
- Models with a different input or mutation protocol declare their exceptions
  explicitly.
- Reusable Pydantic field types own value validation, normalization, and
  serialization. Models enforce additional constraints and relationships
  separately.

### Sources and metadata

- Source access supports inspecting individual acquisitions and interpreting
  batches without repeating storage-specific work for each file.

| Term | Meaning |
| --- | --- |
| Source | Resource containing the data to inspect. |
| Source location | Location from which the data is accessed. |
| Identification location | Filename or path used to interpret acquisition naming conventions; it may differ from the source location. |
| Filename information | Preliminary acquisition information inferred from a naming convention. |
| Acquisition metadata | Recorded information describing the measurement and its observing context. |

#### Source recognition and access

- Source recognition supports finding and inspecting data before reading its
  scientific contents.
- Filename information supports preliminary selection without opening the source.
- Filename parsing uses all registered naming conventions, independently of
  collection scope. The most specialized matching model wins; unresolved
  ambiguity raises before field conversion, including in batch error-recording
  mode. Concrete filename models are public result types for annotations and
  instance checks. Parsing remains universal;
  using a concrete result type does not restrict the registered naming conventions.
  General telescope names establish LMT production and LMT data-model membership.
  Registered instrument conventions add membership for their telescope companions.
- Data in a supported format remains accessible when its filename is unfamiliar
  or absent, including data supplied as an open object.
- Format access is independent of instrument interpretation. Opening a supported
  source does not require identifying its instrument first.
- Tollan supplies generic record-rule matching. Tolteca specializes callable
  rules for source opening and supplies default format callables. File accessors
  supply matching records and sources, including filename fallback for routing.
  Format callables own location support and preparation. Table acquisition
  passes the original source to the selected callable and manages its lifetime.
- Source-opening policies pass the explicit source unchanged to the selected
  callable. Matching alone must not open sources.
- Batch opening matches and collects sources before returning an iterator.
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
- File accessors resolve the applicable data models' opener choices before
  acquiring a resource. A shared callable opens once; distinct choices for the
  same row are an ambiguity.
  No matching opener leaves the source unsupported. Registration order must not
  choose a backend. Borrowed objects bypass opener selection.
- File accessors retain the data models supporting each selected opener or reader.
  Reader content inspection may disambiguate reader candidates after opening;
  opener resolution must not perform speculative I/O.
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
- Data models selecting the same accessor class share its construction and
  metadata read for the row. The class identifies the reader, not a wrapper
  callback or specification object.
  Each model applies its own projection and classification to that result.
  Distinct rule-selected readers require content disambiguation. Multiple
  remaining readers are an error; registration order must not resolve them.
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
  values. Recording a diagnostic is independent of that update. Error-recording
  access continues with other rows; explicit fail-fast metadata inspection
  leaves the input table unchanged.
- An identification miss retains previously collected values, including guessed
  classifications, in error-recording mode. Explicit fail-fast access raises on
  an unrecognized opened object during construction, metadata inspection, or
  scoped opening. An unrecognized filename alone remains a nonfatal diagnostic.
  Only a recognized reader result can replace collected values.
- Metadata inspection does not require reading unrelated scientific data.
- A reader describes an acquisition with stable identity and layout. Tracking
  subsequent edits to its recorded metadata is outside that reader's scope.

#### Native processed KIDs metadata

- Native processed KIDs files remain openable and classifiable while their
  metadata support is incomplete. Acquisition metadata must not be fabricated
  from filename guesses to fill this gap.
- A reader returning no metadata may supply a classification. That update must
  not claim metadata completion or replace collected observation values. Its
  trace records metadata as skipped, preserving earlier unresolved failures.
- **Deferred gap:** the native processed KIDs reader does not yet expose shared
  acquisition identity metadata or a reliable link back to acquisition metadata.
  Closing this gap requires one of those representations; the choice remains
  open. Processed metadata need not satisfy the complete raw-acquisition schema.

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
- Table contents are populated and updated only by TolTECA's defined methods
  and procedures. Direct user edits to rows or columns are outside supported use.
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
  Null memberships mean unknown scope. A collection is in scope when it
  intersects the selected data-model names; otherwise it is out of scope,
  including an empty collection. Successful filename parsing supplies a
  collection, possibly empty; unsuccessful parsing initially leaves it unknown.
- Each row retains its original filename parse result, or absence of a successful
  parse, through selection, reordering and metadata enrichment. The original
  parse result records filename recognition, not current scope.
- Filename-only population reads no sources. Conditional population attempts
  metadata reads only for unrecognized filenames. Full population attempts
  reads for in-scope and unrecognized filenames. Recognized out-of-scope rows
  retain their guesses without being opened in either reading mode.
- Later metadata inspection uses the current membership and identity columns.
  Filename snapshots remain provenance. A successful shared-reader result may
  correct membership and classification without reopening or reidentifying the
  payload. Projection and classification still publish together.

#### Projection and queries

- Projection makes source information available for comparison and selection
  across a collection.
- Scientific columns use declared units, time representations, and coordinate
  components so that query values retain a defined physical meaning.
- A shared filename or metadata column has consistent meaning and storage.
  Separate input projections may supply it through different attribute paths
  and conversions. Filename registration does not merge projection readers;
  incompatible destination writes fail during population.
- Derived-column handlers compute output without modifying their input rows.
  Population applies their collected results after all computations succeed;
  records outside each handler's scope retain their existing values.
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
- Each metadata projection applies only to records recognized by its owning
  handler. Successful enrichment replaces values covered by the selected
  handlers, including values absent from the recovered metadata.
- Covered inputs and their derived columns agree after enrichment, without a
  separate derivation pass by the caller.
- A derived column outside enrichment's coverage may depend only on inputs
  that enrichment leaves unchanged. This includes indirect dependencies.
- Records outside the selection retain their existing values.
- Enrichment preserves filename provenance and actual/identification locations.
- Opening a file table acquires and identifies resources. Full metadata
  enrichment is explicit or requested as part of opening. The resource scope
  exposes both the format-level payload and its unique domain accessor. Shared
  data models may use that same accessor; distinct viable physical readers remain
  an ambiguity error.
- Explicit metadata enrichment reuses live accessors when available. Construction
  modes and later enrichment share the same live-accessor read and projection
  operation. A failed read or projection leaves the row's information unchanged.
- File-table accessors are constructed directly from their selected classes.
  Live `accessor_obj` values retain per-accessor caches. A separately obtained
  named xarray accessor is not implicitly shared with a new table operation.
- Each row records whether source metadata has successfully populated its
  values. Later enrichment skips completed rows by default; callers can force
  another read. Failure does not claim completion or erase a previous successful
  result. Reopening alone does not reset this state.
- Per-row processing traces retain successes, scope decisions, skipped work,
  and failures in order. Successful retries resolve issues without deleting
  prior events. Summary counts use each row's latest outcome per stage; skipped
  work does not resolve a previous failure for input-validation purposes.
- Owned resources close when their scope ends. The table may retain successful
  reader selections for the same source and routing choices, but must not retain
  live accessors bound to closed owned resources. Reopening binds the retained
  selection to the new payload; it does not reuse the previous live accessor.

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
- Instrument domains own the correspondence between scientific labels and numeric
  codes. Combined domains must not assign conflicting meanings to the same label.
- Known label interpretations do not prohibit matching other literal labels.

#### Candidate discovery

- Discovery assembles candidates from acquisition stores and explicit sources
  into a [mixed or scoped collection](#mixed-and-scoped-collections).
- Literal locations and glob searches are distinct input forms.
- Discovery extensions register a settings model under one identifying key.
  Reusing a key or model must raise, including an identical registration.
  Failed registration must preserve the previously usable configuration schema.
  Extensions must register before consumers compile enclosing configuration schemas.
- Coarse store searches retain candidates needed by the full selector.
- Each discovery entry applies its complete selector before entries are combined.
  Selected duplicates retain their first selected occurrence.
- A discovery collection may span stores and instruments while preserving each
  domain's selection meanings.
- Discovery uses a data-model context to define the capabilities available to
  the consuming task. Global and per-entry selections may narrow that set but
  cannot enable capabilities outside it. Without an explicit context, discovery
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
- The configured file accessor owns interpretation of the populated candidate
  table. Each discovery entry applies the full selector once to its combined
  table, including positional selection after all value constraints.
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
- Discovery produces a candidate collection. The consuming workflow owns its
  subsequent processing selection.

### Instrument extensions

- Instrument extensions let shared observation-data workflows handle additional
  data models and let specialized workflows select their interpretation support.
- A data-model definition owns its name. Components bind to that definition;
  registration and workflow selection use its name without an independent alias.
  A component's affiliation is distinct from a record's applicable data models.
  Records may support several data models without requiring inheritance between
  those definitions or duplicating a shared reader.
- A named data-model definition owns registration of its workflow capabilities.
  Its store requirements identify acquisition stores and companion stores;
  they do not enable additional data-model handlers. Its file accessor owns
  table enrichment, metadata readers, and observation
  query interpretation. Defining an accessor subclass does not register a
  data model; direct typed use remains independent of registration.
- A workflow context selects registered definitions. Registering an existing name
  must raise without modifying the registry. Additional registrations do not
  change existing contexts. Importing instrument components may load data-model
  identity information but must not assemble or register workflow capabilities.
  Explicit definition modules assemble their
  completed components without registration. The public data-model, file-table
  and input-discovery entry points register built-in definitions. The filename
  entry point enables parsers without assembling data-model definitions.
  Optional extensions control their own registration entry point.
- Shared extension interfaces live in `tolteca_datamodels.core`. Importing them
  must not load instrument packages or register built-in discovery inputs.
  Registries belong to their corresponding core feature; instrument packages
  supply the concrete registrations.
- Root assembly modules expose task entry points, not extension building blocks.
  The data-model entry point provides live named lookup of registered capabilities
  and workflow selection; definitions and registration interfaces belong to core.
  Core file access uses available registrations but must not import built-in
  instruments itself. Root file access enables the built-ins for that same workflow.
- Shared workflows require no data-model context from the caller. They use
  registered support to identify applicable layouts and invoke the corresponding
  domain operations for supported inputs.
- Specialized workflows receive a data-model context up front and use it to
  select support within the same workflow. Direct concrete file accessors select
  their declared support without requiring a discovery context.
- Each specialization retains the same interpretation and operation contracts
  whether selected through the registry or through explicit context.
- Compatible support may contribute to the same collected source. Shared
  telescope metadata and the associated instrument's classification remain
  available together through one selected reader, independently of registration
  order. Filename-derived computations may contribute cooperatively; they do
  not establish an exclusive reader or owning table accessor.
- Each instrument owns its naming conventions, acquisition classifications,
  recorded-data interpretation, and store conventions.

## KIDs signal processing

- `tolteca_kidsproc` defines a canonical, instrument-independent signal layout
  and reusable analysis recipes for kinetic inductance detector data.
- The canonical signal contract is the handoff from instrument processing to
  generic analyses.
- Callers can also supply signals directly, independently of acquisition files
  and instrument-specific workflows.

### Signal representation

- Signal representation supports KIDs analysis of individual detector channels
  and channel collections through a common semantic API across source layouts.
- The canonical layout specifies the signal structure and semantics presented to
  analyses. Mappings can present that contract over different source layouts.
- Signal representations use Tollan's schema/mapper/accessor/view framework. Its
  [data access and representation contracts](../../tollan/design/spec.md#data-access-and-representation)
  govern coordinate and unit handling, mapper reuse, and view scope and lifetime.

| Term | Meaning |
| --- | --- |
| Sweep | Detector signal represented against probe frequency. |
| Multi-sweep (`multi_sweep`) | Channel-indexed collection of sweeps. |
| Timestream | Detector signal represented against time. |
| Multi-timestream (`multi_timestream`) | Channel-indexed collection of timestreams. |

- Multi-channel representations support collective analysis and access to
  individual channels through the corresponding sweep or timestream interface.

### Analyses and results

- Analyses support interactive inspection and repeated diagnostics of selected
  signals.
- Analyses consume the [canonical signal contract](#signal-representation)
  without requiring instrument identity or acquisition metadata.
- KIDs analyses use Tollan's analysis framework and follow its
  [analyses and results contract](../../tollan/design/spec.md#analyses-and-results).
- Numerical calculations remain usable independently of analysis objects and
  workflow orchestration. D21 and PSD expose classmethods accepting numerical
  arrays and explicit parameters. Configured instance methods adapt views or
  xarray inputs and return standalone results; full calls install analysis
  namespaces under the shared Tollan contract.
- Splitting work for memory or parallel execution must not change the scientific
  calculation.

| Analysis class |
| --- |
| `D21Analysis` |
| `PSDAnalysis` |

## TolTEC KIDs processing

- TolTEC KIDs processing turns native acquisitions into usable signals and
  scientific results for interactive inspection and operational diagnostics.
- `tolteca_kids` owns TolTEC-specific transformations, diagnostics, and workflow
  recipes.
- A KIDs processing pipeline operates on one block. Each processing step receives
  that block and retains its invocation context there. Acquisition preparation
  and dispatch across blocks belong to the enclosing workflow. Operations that
  combine blocks belong to separately defined higher-level pipelines.
- Instrument processing consumes [recorded measurements and acquisition context](#toltec-observations)
  to prepare signals that satisfy the [canonical signal contract](#signal-representation).
  Acquisition timing checks and missing-sample handling belong to this preparation.
- Domain diagnostics and recipes compose [KIDs analyses](#analyses-and-results)
  with TolTEC-specific scientific operations.
- Domain recipes use the shared [processing workflow contracts](#processing-workflows)
  for configuration and execution.

### Drive fitting

Drive fitting selects per-channel driving levels from an attenuation sequence;
commit transfers a saved result to a current target comb. The
[snapshot drive-fit workflow specification](../delivery/drive-fit-normalization/drive-fit-workflow-spec.md)
records configuration, scientific results, publication, and the Tlaloc handoff.
It distinguishes file
installation, tone upload and hardware attenuation application, and marks the
proposed OT sequence separately from observed source behavior.
The [production integration note](../delivery/drive-fit-normalization/production-integration.md)
records the subsequent evidence, algorithm checks, and handoff to the combined
detector-setup specification. An undersampled nonlinear phase window must become
a per-power fit failure before evaluating window statistics; it must not abort
otherwise usable channels or alter their scientific selection rule.

### Acquisition blocks

- Blocks isolate coherent acquisition intervals for independent processing while
  retaining the context needed to interpret them.
- A block is a selected acquisition interval with its applicable channels and
  probe settings. Its extent follows the recorded acquisition structure.
- Collections retain acquisition organization and support selecting blocks for
  independent pipeline invocations.
- Block boundaries describe recorded samples. Allocated storage and requested
  acquisition counts do not establish that those samples exist.
- A block retains the acquisition, model, and probe context needed for use
  independently of its collection.
- Organizing measurements into blocks is separate from their scientific
  transformation.

| Identity | Meaning |
| --- | --- |
| Channel identity | Identifies a channel within one acquisition and producing interface. |
| Sample identity | Identifies a recorded sample within one acquisition stream. |
| Detector identity | Identifies a physical detector independently of acquisition-specific channel numbering. |

- A channel identifier is local to its acquisition and readout interface. It is
  not a persistent detector identity.
- Individual blocks and all subsets retain the original acquisition's channel
  IDs and sample IDs through selection and reordering. These IDs locate each
  selected channel and sample in the original acquisition, independently of
  local array or table positions.
- [Detector matching](#cross-acquisition-analysis) tracks detector identity by
  establishing channel correspondence across acquisitions.

### Signal preparation and diagnostics

- Signal preparation supplies canonical inputs for generic analyses; diagnostics
  apply TolTEC context to their results.
- A sweep acquisition's native samples may also supply time-domain diagnostics.
  Sweep averages cannot substitute for those samples.
- Sampling interpretation uses the recorded sample rate, not a rate estimated
  from file duration.
- Domain diagnostics follow the [analysis-result contract](#analyses-and-results)
  and remain accessible independently of plotting and invocation contexts.
- A failed optional correction does not itself invalidate its input. Consumers
  choose acceptable alternatives under [workflow failure handling](#sequential-execution-and-failures).

### Detector modeling

| Feature |
| --- |
| Detector models |
| Model-based signals |

### Cross-acquisition analysis

- Cross-acquisition analysis compares detector response and follows tone identity
  across tuning and observing runs.
- Matching results retain both acquisitions' interface and channel identities.
  Equal channel numbers across acquisitions do not establish a detector match.
- Matching records refer to the contributing data without requiring copies of
  their raw arrays.

### Processing steps

- TolTEC KIDs processing steps adapt domain operations to configured recipes
  using Tollan's `Step` framework.
- Block processing steps do not enumerate an acquisition's blocks. The enclosing
  workflow identifies and creates blocks, selects them, and invokes the processing
  pipeline for each block. Publication collects the resulting block outputs.
- They follow the shared contracts for
  [configuration composition](#configuration-composition),
  [step invocation and scientific-result ownership](#steps-and-scientific-results),
  [execution and failure handling](#sequential-execution-and-failures), and
  [invocation diagnostics](#inspect-execution-diagnostics).

| Workflow layer | Step classes |
| --- | --- |
| Acquisition preparation | `Input`, `BlockIdentify`, `BlockCreate` |
| Block processing | `SweepReduce`, `SweepCheck`, `SweepCheckPlot`, `KidsFind`, `KidsFindPlot`, `TonePrepare`, `TimestreamReduce`, `NoOp` |
| Product publication | `DataProdOutput` |
| Instrument file installation | `TlalocOutput` |

### Sweep finding and next-tone preparation

Sweep reduction prepares the next tone comb using the production v2 Tlaloc
contract. Finding consumes the checked sweep and transfers recorded input-probe
amplitudes through the v2 matcher. The baseline preserves the matcher's
many-to-one assignments; it does not reinterpret them as missing matches.

- Finding publishes both the detection and input-probe ECSV tables with their
  units, identities, flags, match distances and detection statistics.
- Optional finding plots use the same checked samples as finding. The product
  retains each block's figures independently of installing instrument files.
- Tone preparation preserves v2 selection, fixed-count Tune behavior,
  variable-count standalone sweep behavior, phase assignment and common FFT row
  ordering. The four files retain the existing names and numerical columns.
- Product publication retains prepared tone values. Installing a reopened
  product must reuse those values without matching or generating phases again.
  Installation follows successful enabled publication and registration, and
  selects only the last acquired sweep block.
- Ordinary sweep output must not create or modify `atten_drive.dat`. The current
  observing sequence has no attenuation-update hook at ordinary tone upload.
  Amplitude changes therefore allocate the available power among tones; they
  cannot independently preserve absolute power for every tone as the comb changes.

#### Optional amplitude fallback at fixed attenuation

The output policy owns amplitude filling during tone-file preparation. It leaves
finding and its correspondence tables unchanged. A null fallback preserves v2.
An explicit `median` or `lut` policy fills every detection in an ambiguous
connected match group and every unavailable matched amplitude. Only a group
containing one probe and one detection is unambiguous. Tune's retained input
probes and zero-amplitude masked placeholders keep their existing assignments.

The LUT policy reads the network's `amps_correction_lut.csv` from the configured
Tlaloc etc root, even when installation is disabled. It evaluates absolute RF
frequencies using the same natural cubic spline and linear endpoint extrapolation
as Tlaloc. Let $u_j$ be a positive enabled input-probe coefficient and $L(f_j)$
the LUT coefficient at that probe frequency. The set $U$ contains unambiguous
input probes, counted once by their acquisition channel IDs. The user-selected
global scale is

$$
\delta = \operatorname{median}_{j\in U}
  \left[20\log_{10}\frac{u_j}{L(f_j)}\right],\qquad
k = 10^{\delta/20}.
$$

An ambiguous or unavailable output coefficient at frequency $g_i$ becomes
$b_i=kL(g_i)$; a unique match retains its input coefficient. Multiplying the
entire LUT by a constant must not change these resulting coefficients. Missing
LUTs, nonpositive interpolated values, and absence of usable anchors must fail
preparation before publication or installation. The median policy instead uses
the median finite nonnegative coefficient of enabled input probes.

After tone selection and padding, divide all output coefficients by the common
factor $M=\max(1,\max_i b_i)$ to satisfy the [0, 1] file range. Do not clip
individual tones or alter drive attenuation. This common scaling preserves
relative powers, including under Tlaloc's waveform peak normalization.

Products retain the original matched coefficients, ambiguous and filled masks,
LUT hash and source path, $k$, $\delta$, anchor count and $M$. Replay exports the
saved final coefficients without accessing the LUT. Each preparation context
retains the effective output policy for reproducing that invocation.

The diagnostic power fraction uses drive-fit commit's shared fixed-total model:
$p_i=b_i^2/\sum b_j^2$, with masks and Tlaloc's last-write ownership of duplicated
DAC bins applied before the sum. A changed comb redistributes this budget. The
actual DAC waveform is peak-normalized and its mean power also depends on phase
crest factor; the fraction model is not an absolute detector-power calibration.
The observed benefit across a long series of sweeps remains unvalidated.

### Sweep quality diagnostics

- Sweep checks support quality inspection and downstream use of corrected data.
  They operate on the supplied block after reduction and preserve the original
  reduction. Each block retains its
  corrected sweep, flags, statistics, D21 diagnostics and effective check settings
  as scientific data. Checking another block requires another pipeline invocation.
- The core calculation preserves production v2 results. Questions about that
  algorithm remain review findings; they do not authorize numerical changes.
  Readout-power modeling and raw-sample noise PSD are outside the current port.
- A saved check is usable without a live execution context. Its view exposes
  scientific results; it does not rerun the check or reconstruct execution state.
- Configured visualizations construct presentations from completed check results
  for standalone inspection and live Web panels. The plot step selects results
  within its supplied block and invokes its configured artifact specifications.
  Its artifact context belongs to that block's step invocation. The paired
  configuration and artifact context follow Tollan's
  [visualization and artifact contract](../../tollan/design/spec.md#visualizations-and-artifacts).
  Product output owns artifact publication and inventory; Web components own
  selection state and source lifetime.
- Each retained artifact carries its effective Viz configuration and native
  figure. Product output invokes the step's artifact spec to materialize that
  context; it does not depend on a plotting step or figure backend. Artifacts are
  scoped by interface, block path and step invocation key. Export failure prevents
  run commit and tag advancement.
- Processing records are stored beneath each block's `tolteca_kids` node in Zarr.
  Acquisition preparation records remain at the interface root's `tolteca_kids`
  node. Records persist configuration, completion and failures; live artifact
  objects are excluded. Figures can be reconstructed from the saved check view.
- The runner owns acquisition preparation, block iteration and publication.
  `Kids` provides configured steps and sequential sweep/timestream pipelines;
  those pipelines contain only per-block operations.
  Existing standalone reducers' acquisition convenience calls remain available;
  the configured workflow uses block-local invocation records.
- Check and plot steps are independently optional in the configured sweep
  pipeline. Disabling plots retains scientific check results. Recomputing a
  check requires regenerating dependent live figures before publication.

## Data products

- Users work with related observations and results across observing-site,
  archive, and local workspaces. Data-product management lets them locate
  available data and reuse selected results without rerunning the producing
  workflows.
- Scientific domains determine product contents and provenance. This domain
  defines how those products are published and made available to subsequent
  analysis, visualization, and archive consumers.
- TolTECA owns the shared product model and reusable product APIs.
  `tolteca_db` ingests observations and published products using this model.

### Logical model and stored representations

- The product model defines products, runs, associations, artifact ownership and
  named selections independently of a storage representation.
- DPDB represents queryable identities and relationships through relational
  tables, keys and constraints. Typed configuration values remain metadata of
  their owning records.
- Every realized scientific product is a first-class DataProd with its own UID,
  metadata, associations and sources. A run records the producing activity; it
  does not supply a missing part of a product's database identity.
- Every non-singleton DP belongs to a version group that manages alternative
  realizations. The group declares their shared DP type. Its membership
  associations identify versions; its tags select associations.
- The product-store directory layout and YAML indexes are materialized views of
  those relationships. The root index describes the version group and its
  inventory; each version index describes a distinct realized product.
- DPDB must retain enough information to recreate the defined `dataprod_toltec`
  projection at a selected output root: directories, complete indexes, version
  associations, current tag selections, allocation state and artifact placement.
  Rebuilding indexes must not require the original YAML files. Scientific payloads
  are materialized from registered artifact sources, without rerunning processing.
  Namespace descriptors preserve reservations and per-output publication requests,
  including failures without product rows. Tag-selection history is deferred.
- The filesystem projection is optimized for directory exploration and portable
  exchange. Other projections may present the same records differently, such as
  a tolteca_web DP viewer. Product identity, associations and tag selection retain
  the same meaning across projections; filesystem hierarchy does not constrain
  the Web presentation.
- Database registration and disk publication map the same logical relations to
  their respective representations. Filesystem publication remains usable
  without a database; this separation does not require a live database to export
  or reopen a product. Ingest must preserve the information needed to regenerate
  that projection, including resolved producing configuration and provenance.
- Source URIs describe placement. Moving a representation does not redefine the
  product, run or association that it represents. Run identity and tag/allocation
  scope must be expressed in the model rather than inferred from a parent path.

### Product schema

- The schema supports discovery and provenance queries across products held in
  different locations. It separates scientific identity, stored sources, and
  product relationships.
- The [schema review](../../tolteca_db/design/data-product-schema-review-20261013.md)
  specifies the implemented columns and application constraints. Its
  [browser report](../../tolteca_db/design/schema-review-20261013/build/html/index.html)
  is generated from the runtime ORM in an empty SQLite database. Existing deployed
  databases still require a migration; none is applied by this work.

| Record | Schema name | Responsibility and relationships |
| --- | --- | --- |
| `DataProd` (DP) | `data_prod` | One independently identified singleton, version group, named group or realized result. |
| `DataProdType` | `data_prod_type` | Registry of DP types; add `dp_version_group`. |
| `DataProdSource` | `data_prod_source` | Stored resource owned directly by one DP and located at one `Location`. |
| `Location` | `location` | Registered data root against which source URIs are resolved. |
| `DataProdAssoc` (DPA) | `data_prod_assoc` | Directed, typed relationship between two DPs; primary marker and relationship facts. |
| `DataProdAssocType` | `data_prod_assoc_type` | Registry of DPA types. |
| `DataKind` | `data_kind` | Registry of data classifications. |
| `DataProdDataKind` | `data_prod_data_kind` | Many-to-many assignment of kinds to DPs. |
| `Flag` | `flag` | Registry of quality labels; its `namespace` is unrelated to a product namespace. |
| `DataProdFlag` | `data_prod_flag` | Many-to-many assignment of flags to DPs with assertion provenance. |

| Execution record | Schema name | Responsibility and relationships |
| --- | --- | --- |
| `Actor` | `actor` | Immutable client/environment facts reused by runs and attributed records. |
| `DataProdNamespace` | `data_prod_namespace` | Portable identity scope and run-counter owner. |
| `DataProdInventory` | `data_prod_inventory` | Independently maintained registered facts; portable identity for the local journal. |
| `DataProdNamespacePlacement` | `data_prod_namespace_placement` | Namespace path in one inventory. |
| `Run` | `run` | One invocation in one output namespace; full resolved workflow and exact selected inputs. |
| `DataProdAssocRun` | `data_prod_assoc_run` | Many-to-many link between runs and DPAs, with edge-specific execution context. |
| `DataProdTag` | `data_prod_tag` | Product-owned label selecting one outgoing DPA, or empty. |

- A run has no owning-product FK. Run provenance links to DPAs; no `run_data_prod`
  table or version-group extension table is required by this model.
- Generic database auditing belongs to deployment infrastructure. The unused
  `EventLog` ORM has been removed; existing databases have not had that table
  dropped. Tag-selection history remains deferred.

| Field category | Owner | Role and boundary |
| --- | --- | --- |
| Entity identity | `uid_obs`, `uid_raw_obs`, `uid_raw_obs_file` | Observation, acquisition and file identity under the [observation identity contract](#observation-and-interface-identity). |
| Product identity | `DataProd.uid` and its identity scope | One product; several products can concern the same entity. Namespace is separate from UID text. |
| Source placement | `DataProdSource.location`, `source_uri`; `Location.root_uri` | Locates a stored representation independently of product identity. |
| Typed product metadata | `DataProd.meta` | Product facts, including a version group's contained product type. |
| Typed source metadata | `DataProdSource.meta` | Facts about the stored resource, including interface-specific metadata. |
| Relationship facts | `DataProdAssoc.context` | Facts about the relationship itself. |
| Resolved execution configuration | `Run.context` | Full effective workflow configuration and exact selected inputs. |
| Relationship execution details | `DataProdAssocRun.context` | Execution-specific details for this run and association; not another copy of the full workflow. |

- Persistence preserves typed metadata and context contracts. Moving existing
  DPA process context to the run/link owners requires an explicit migration.

#### Metadata and database fields

- Typed product metadata owns descriptive and scientific facts, including the
  product name. Product names need not be unique. The ORM's `name` property reads
  the metadata value; there is no independently writable name column.
- Relational columns own identity, foreign keys and operational state with
  database constraints. The run name is a column because namespace-wide name
  reservation requires uniqueness. Actor display naming uses `name`.
- Querying a metadata field does not require promoting it to a column. Physical
  optimizations must preserve one authoritative value and the typed metadata API.

#### Namespace identity and portability

- A namespace has a stable portable UID. Each inventory places it at a relative
  subpath, such as `ql` or `archive/taco/ql`. Renaming a placement preserves
  scoped product/run identity. Placement and `Location.root_uri` are independent.
- `DataProdNamespace` uses table `data_prod_namespace` and FK
  `data_prod_namespace_fk`. Python relationships and configuration use `namespace`.
- Realized products have namespace-scoped identifiers. Version groups are global.
  Portable product references carry separate `namespace_uid` and `uid` values.
  A run has a globally unique portable UUID and a namespace-unique readable name.
  Product UIDs and run UIDs do not embed namespace text.
- `dp_obs`, `dp_raw_obs`, intrinsic `dp_cal_group` components and
  `dp_version_group` are global logical products. Their indexes can occur in
  several namespace projections without creating additional products.
- `pk` and `_fk` values are local database keys. Import matches portable namespace
  actor and run UIDs, then remaps references to destination PKs. UUID is the
  portable identity type for namespaces, actors and runs.
- A Taco namespace with UID N can have path `ql` in Taco's inventory and
  `taco/ql` in Unity's copied inventory simultaneously. Imports preserve both
  placements and each inventory's product, run, source and tag presence.
  Imports must not rewind allocation. Import placement configuration owns
  destination path mapping.
- `data_prod.data_prod_namespace_fk` is nullable for global singletons and required
  by the ORM for scoped products. Uniqueness is `(data_prod_namespace_fk, uid)`;
  a partial unique index enforces global UID uniqueness on SQLite/PostgreSQL.
  PostgreSQL supports shared services; SQLite supports local workflows.

#### Actors and record attribution

- An actor owns immutable client, runtime and deployment facts reused across
  runs, such as hostname, user and environment. Changed recorded facts select a
  different actor. Invocation-specific configuration, commands, inputs and timing
  belong to the run.
- Mutable records identify their creator and latest modifier through
  `created_at`, `created_by_fk`, `updated_at`, `updated_by_fk`. Actor FKs reference
  `actor.pk`; Python relationships are `created_by` and `updated_by`.
- Creation sets both actor references to the creator. Import preserves recorded
  attribution while remapping local keys. Idempotent import does not change
  modification timestamps; destination edits record their actual modifier.
- Actor records have their own creation time, without recursive creator/updater
  references. This permits actor creation independently of attributed records.
- Writers reuse an actor with matching stable runtime facts. Native writes record
  the current actor; imports preserve recorded UUIDs and timestamps. Unknown
  historical creators remain null. Existing quality-flag assertion fields retain
  their separate contract.

#### Product types

- `dp_obs` and `dp_raw_obs` let users find an observation's acquisitions and
  their files before or after processing.

| DP type(s) | Design role |
| --- | --- |
| `dp_obs` | Represents one observation. Lists its known acquisitions through associations to their `dp_raw_obs` products. |
| `dp_raw_obs` | Represents one acquisition. Lists its known files through `DataProdSource` records. |
| `dp_raw_obs_proc` | One realized acquisition-level result, shared by its interfaces. Each producing run creates a distinct product in a version group. |
| `dp_version_group` | Stable logical product managing versions of one declared DP type. Its metadata declares the managed non-singleton DP type. |
| `dp_cal_group` | Intrinsic connected component of the per-file calibration graph projected onto raw-acquisition nodes. May contain multiple VNAs; the earlier VNA supplies its UID, with absorbed UIDs retained as redirects. |
| `dp_drivefit`, `dp_focus_group`, `dp_astig_group`, `dp_oof_group`, `dp_pointing`, `dp_map`, `dp_catalog`, `dp_named_group` | Follow the shared [association](#product-associations-and-primary-products) and [publication](#publication-ownership-and-atomicity) contracts. |

- Observation and acquisition inventories support partial discovery; additional
  acquisitions or files can be registered later.
- Raw-product registration and discovery are independent of scientific processing.
- `dp_obs` and `dp_raw_obs` are singleton products per observation and acquisition,
  respectively. Their identities remain the same across output namespaces and
  storage locations.
- A `dp_raw_obs_proc` product holds acquisition-level transformations and
  diagnostics from its producing run. A version group organizes these products
  without merging their identities or producing configurations. Observation-level
  science reduction remains a separate product.

Calibration grouping follows the adopted
[calibration-component contract](../../tolteca_db/design/association-rules.md#calibration-components).
Each VNA calibrates itself; this semantic relation does not require a stored
self-edge. The graph must retain known VNA roots and isolated nodes independently
of that choice. Unsynchronized interface lineages can join several VNAs into one
acquisition-level component. Membership follows calibration edges, including
timestreams; observation order does not define a boundary.

The [persistent interface-lineage layer](../../tolteca_db/design/association-rules.md#persistent-per-interface-lineage)
separates logical calibration references from physical source copies and derives
the raw-acquisition associations from those references. Portable indexes and DB
refresh preserve known and unresolved references independently of source relocation.
VNA roots are stored as self-references without generating self-DPAs.

A VNA opens an unscoped `dp_cal_group_<VNA quartet>` on first sight. Merges keep
the earlier VNA's UID; absorbed groups remain addressable and resolve to the
current group. A component without a known VNA remains queryable without an
invented group anchor. These are intrinsic products, not version-group members.

Only each group's original VNA primary association is materialized as a DPA.
Full calibration membership is projected from persistent connectivity. Updates
union affected components without rewriting all members; descendant enumeration
is a read/export operation. No new per-file facts are invented by this projection.
The [connectivity contract](../../tolteca_db/design/association-rules.md#persistent-calibration-connectivity)
defines identity ordering, transactional state and transfer behavior.

#### Product associations and primary products

- DPAs record product membership and dependencies, including processing inputs
  and calibration products. The direction follows the operation: its resulting
  product points to the products used to construct it. Processing links a derived
  result to its inputs; grouping links a group to the products it groups.
- Source and destination describe the directed relation, not a universal
  parent/child hierarchy. Run-specific input selection refers to exact products,
  including their namespace identity for scoped inputs.

| DPA type | Source DP → referenced DP |
| --- | --- |
| `dpa_obs_raw_obs` | `dp_obs` → `dp_raw_obs` for an acquisition belonging to that observation. |
| `dpa_raw_obs_cal_obs` | `dp_raw_obs` → calibration `dp_raw_obs`. |
| `dpa_raw_obs_proc_raw_obs` | `dp_raw_obs_proc` → its acquisition's `dp_raw_obs`, selected as primary. |
| `dpa_cal_group_raw_obs` | `dp_cal_group` → `dp_raw_obs`; original VNA primary edge stored, complete membership projected by the calibration query. |
| `dpa_drivefit_raw_obs` | `dp_drivefit` → `dp_raw_obs`. |
| `dpa_drive_fit_commit_raw_obs` | `dp_drive_fit_commit` → target `dp_raw_obs` (primary). |
| `dpa_drive_fit_commit_drivefit` | `dp_drive_fit_commit` → exact namespaced `dp_drivefit` realization. |
| `dpa_focus_group_raw_obs` | `dp_focus_group` → `dp_raw_obs`. |
| `dpa_astig_group_raw_obs` | `dp_astig_group` → `dp_raw_obs`. |
| `dpa_oof_group_raw_obs` | `dp_oof_group` → `dp_raw_obs`. |
| `dpa_named_group_data_prod` | `dp_named_group` → any DP. |

- Exact metadata associations use `AssocRules`: observation membership matches
  `(master, obsnum)`; processed-to-raw matches the quartet for every realization;
  calibration associations match per-interface `cal_*` quartets and retain the
  contributing interfaces in edge context. Namespace and version belong to the
  product endpoints, not to a separate association scope.
- Live association maintenance must exploit exact identities, uniqueness,
  indexed references and acquisition order. `AssocPool` collects new product
  identities and incoming interface evidence; exact indexed lookups include
  waiting dependants. `AssocRules` configures `AssocGenerator`, whose collators
  maintain only relationships affected by those facts. Metadata summaries, copies
  and tags do not trigger identity reprocessing. Explicit `reconcile` is separate
  from incremental generation. Portable import registers supplied DPAs and their
  provenance, then uses the same generator for changed/waiting portable calibration
  references; it does not reconstruct lineage from physical source metadata.
  Full-history processing per update is a release blocker. Query, hydration and
  write costs must scale with affected facts and match full reconciliation.
  See the [performance contract](../../tolteca_db/design/association-rules.md#live-update-performance-contract).
- File ingestion consumes opened accessors after model-table selection and
  preserves their complete primary metadata. ToltecDB import retains partial
  record metadata when a file is unavailable. A known calibration quartet can
  establish a logical raw product without a file or selected target row. An
  unknown calibration master must remain unresolved; ingestion must not invent
  its product identity or filename. See the
  [association contract](../../tolteca_db/design/association-rules.md).

- Association browsing presents persisted edges in both directions, with exact
  endpoint identities, primary roles and per-interface calibration evidence.
  Logical products remain visible without registered or locally available files.
  Browsing does not construct edges or open scientific payloads. Rule ownership,
  enrichment and read-query boundaries are defined in the association contract.

- A product's **primary** is the `dp_obs` or `dp_raw_obs` used to group it for
  storage and browsing.
- `dp_obs` and `dp_raw_obs` are their own primary. Their membership or calibration
  associations do not change that assignment.
- For every other DP, the producer selects one association as primary. Resolution
  follows the selected associations until it reaches a `dp_obs` or `dp_raw_obs`.
- `is_primary` marks the selected DPA without changing its relationship type.
- Version membership and primary placement have separate relation types. An empty
  version group must resolve its primary without treating it as a version.
  Proposed labels are `dpa_version_group_data_prod`, `dpa_version_group_obs` and
  `dpa_version_group_raw_obs`; their exact registry names remain for review.
- Resolution reports missing targets, missing or ambiguous primary selections,
  and cycles.
- DPAs relate products. Source ownership remains `DataProd` → `DataProdSource`;
  `Location` and the source URI identify where the contents can be accessed.

#### Product identifiers

- Every version is independently addressable, can be an association target and
  owns its sources. Scoped portable references combine namespace UID and product
  UID as separate values; global singletons use their entity-based UID.
- Entity identifiers follow [observation and interface identity](#observation-and-interface-identity).
  `primary_uid` is the UID of the [primary product](#product-associations-and-primary-products)
  (`primary.uid`).

| Product | Identity scope | UID |
| --- | --- | --- |
| Singleton `dp_obs` | Global | `{uid_obs}` |
| Singleton `dp_raw_obs` | Global | `{uid_raw_obs}` |
| Version group for any non-singleton product | Global | `{logical_product_identifier}` |
| Any realized non-singleton product | Producing run namespace | `{version_group.uid}/{run_uid}` |

- The producer defines the logical product identifier. Examples are
  `{uid_raw_obs}_proc` for processed acquisitions and
  `{dp_type}_{primary.uid}_g{N}` for grouped analysis products.
- Namespace N1 may contain `U_proc/R1` and N2 `U_proc/R2`, with distinct run
  UUIDs R1 and R2. Both runs may use the name `0007`. A namespace rename
  changes neither UID. A DPA targets the exact product; its
  portable reference retains the namespace UID separately.
- For grouped analysis results, `N` counts contributing observations or
  acquisitions as defined by the producer. Associations identify the exact
  contributors. Producers must distinguish different contributor sets with the
  same primary and count when these represent different logical products; the
  additional naming rule remains producer-specific.
- One run can consume A and B from different namespaces and produce C and D in
  its single output namespace. Output UIDs include their distinct group identifiers
  and the same run UID. Naming allows one realization per group per run.
- Consumers use recorded run relations for provenance rather than parse product
  UIDs. Copying an index or artifact preserves its recorded identity.
- Global run UUIDs make generated realization UIDs distinct across namespaces.
  Namespace ownership remains separate from UID uniqueness: a realization belongs
  to its producing run's namespace, and portable product references retain that namespace.

#### Version groups

- Every non-singleton DP has a version group, including acquisition-level
  processing, grouped analyses and maps. Each realization remains a first-class DP.
- A group has type `dp_version_group`, stored in the ordinary `data_prod` table.
  Its typed `VersionGroupMeta` requires `content_data_prod_type`, a registered
  type value. This belongs in `meta`, without a special extension table or type FK.
- Every version member must have the contained product type. Members may belong
  to different namespaces; the group's own type differs from the type it manages.
- The group has stable identity without a producing-run suffix and does not
  recursively require another version group. `dp_named_group` retains its separate
  generic grouping role and may contain mixed types.
- Each tag is a label owned by one group DP. It selects at most one outgoing
  version-membership DPA, or is empty. The selected edge must originate at the
  owner and target a completed version of its declared product type. Several
  labels may select the same edge.
- Tag uniqueness is `(data_prod_fk, label)`. One DPDB has one `main` selection
  per version group, across its namespaces. Publishers may choose explicit labels
  such as `ql-main`; namespace paths neither generate nor constrain tag labels.
  Local qualifiers conventionally use hyphens (`ql-main`, `nightly-20260708`).
  Reserve colons by convention for explicitly mapped remote-origin qualifiers
  (`taco:main`, `taco:ql-main`). This is a publishing convention, not a hard
  character restriction or a source-location resolver.
  No additional namespace or run field belongs on the tag.
- Moving a tag changes selection without changing product identity, contents or
  producing configuration. Producing runs remain accessible through provenance.
- Tag-selection history is deferred. Creator/latest-modifier attribution records
  the current selection's provenance; it is not a complete selection history.

### Scientific contents and representations

- Product contents preserve reusable scientific data separately from formats
  intended for display, inspection, or external ingestion.
- A representation is a form in which those contents are made available.
  Representations share the product's identity and provenance.
- An artifact is a stored representation of product contents.
- Stored representations preserve the producing domain's data and analysis
  contracts when reopened.

#### Storage format and access patterns

- Parquet and Zarr are parallel options for persisting scientific contents.
  Neither is restricted to discovery catalogs, individual acquisitions or group
  products. Parquet may hold a directly published scientific result, not only an
  export of another stored representation.
- Choose the format and data layout for the intended consumer's queries and
  selections. Optimize against representative access patterns and measured
  performance rather than imposing one layout across products.
- Scientific processing owns computation and alignment across inputs. Published
  results can include precomputed group tables, aligned arrays and summaries for
  selective consumption without repeating that processing.
- A product may provide multiple representations when consumers benefit from
  them; using Parquet and Zarr does not require duplicating every result.
- Tabular snapshots may expose selected DPDB records or model-table projections
  for consumers. They do not replace DPDB's ownership of product identity,
  relationships, provenance and registered locations, and are not a mandatory
  intermediate step for every consumer.

#### Processed-acquisition tree

- A `dp_raw_obs_proc` Zarr artifact spans one acquisition. Its root children must
  correspond one-to-one to that acquisition's enabled interfaces, including KIDs
  readouts and HWPR. Each interface retains its input identity, metadata, data and
  processing results.
- Every file remains under its interface node, even when only one interface is
  enabled. Interface count must not change the stored tree's structure.
- Storage uses `array.zarr/<interface>/`, with interface names directly beneath
  the product root. The root is the product scope; each interface child
  owns its acquisition metadata and interface-specific contents. KIDs subtrees
  contain block collections, analysis results and saved step records; HWPR retains
  its own data layout. A separate `interfaces` container adds no role to this layout.
- Product readers select an interface subtree and return it as a scientific tree.
  The stored product root and the returned interface root are different scopes.
  Selecting the only available interface automatically does not change storage.

#### Tables, plots, and diagnostic exports

- Export representations make stored results usable by their intended consumers.
- Tables and plots present the same scientific results used by programmatic
  consumers. Export preserves those results' numerical meaning.
- Invocation records support execution diagnostics under the
  [workflow diagnostics contract](#inspect-execution-diagnostics). They have a separate
  purpose from scientific-data persistence.

### Product sources and storage

- Product stores organize independently produced results for discovery, selection,
  and publication across processing runs.
- Storage organization makes related products discoverable without making their
  current location part of their scientific identity.

#### Product indexes and organization

- Product indexes support discovery and inspection without loading scientific
  contents. A portable index describes its DP, associations, provenance and
  sources without requiring a database connection.
- An index-only product may reference existing products and sources without
  copying their scientific data.
- Namespace paths let independently produced trees coexist under one
  `dataprod_toltec` root. Names such as `scratch`, `ql` and `taco/ql` describe
  purpose or origin without imposing processing semantics. Paths may be renamed
  hierarchically; namespace identities are preserved rather than consolidated.
- Version groups, observations, acquisitions and intrinsic calibration groups
  retain their unscoped UIDs wherever their indexes appear. A namespace directory
  projects the versions produced in that namespace. Registering copies adds
  representations of the same DP.
- Matching portable namespace/product identities identify the same originating
  namespace and products. They do not imply equal inventories at different
  locations or synchronization of their mutable state. A transferred tree may
  contain only a subset of the source's products. Relocation preserves recorded
  identities and relative source references.
- DataProdStore owns placement and publication. A primary-entity directory can
  contain several products or version groups of the same DP type; its child
  labels must distinguish their logical identities.
- The layout below defines one filesystem projection of the product graph.
  Projection to a new root preserves product/run identity and selection while
  producing location-relative references appropriate to that destination.

| Product form | Index path relative to the store root |
| --- | --- |
| Namespace inventory | `{namespace.path}/namespace.yaml` |
| Logical observation | `{namespace.path}/{primary_label}/dp_obs.yaml` |
| Logical acquisition | `{namespace.path}/{primary_label}/dp_raw_obs.yaml` |
| Any version group | `{namespace.path}/{primary_label}/{logical_product_directory}/index.yaml` |
| Processed-acquisition version group | `{namespace.path}/{primary_label}/dp_raw_obs_proc/index.yaml` |
| Grouped-analysis version group | `{namespace.path}/{primary_label}/{dp_type}_{primary_label}_g{N}/index.yaml` |
| Realized product | `{version_group_directory}/{run.name}/index.yaml` |

- `primary_label` omits a leading `tcs-` from the canonical primary UID; other
  master prefixes remain. Index UIDs retain canonical entity identifiers.
- The root index describes a version group, declared product type, version
  associations and current tag selections. Each completed version has its own product index,
  run provenance and artifact sources; it remains interpretable after detachment.
- `{namespace.path}/namespace.yaml` owns the portable namespace record, actors and
  run inventory. Its namespace record carries `counter_next`; group indexes carry
  namespace references on their realized members, never an independent allocator.
- A group's `products` embeds its realized product records; `assocs` identifies
  membership and primary placement; `tags[label].current` selects an exact member
  through its UID and namespace UID.
  Embedded records permit metadata-only ingestion of the complete inventory.
- A singleton or version group remains global. Its registered index representation is a source with
  `meta.format=data_prod_index` and `meta.namespace_uid`, making namespace visibility
  reconstructable without assigning a namespace to the singleton itself.
- Logical observation and acquisition inventories can be refreshed as files
  arrive. Their discovery lifecycle is distinct from immutable result publication.

#### Runs and tags

- A run identifies one invocation in one output namespace. It may produce several
  independently identified products there and consume inputs from other namespaces.
  Version groups organize outputs; tags provide stable named selection.

| Term | Meaning and scope |
| --- | --- |
| Product | One independently identified DP, with its own sources and relationships. |
| Version group | Global logical DP managing versions of one declared DP type across namespaces. |
| Run | One processing invocation; no owning-product FK. |
| Run UID | Portable UUID identifying an invocation across stores and databases. Re-registering the UID addresses the same run. |
| Run name | Namespace-unique path component, supplied explicitly or generated by the namespace counter. |
| Tag | Group-owned label selecting an outgoing version association, or empty. |
| Counter key | Optional context in a generated name, such as `{value:04d}-{key}`; not execution identity on its own. |

- `data_prod_assoc_run` links runs and exact DPAs using composite key
  `(run_fk, data_prod_assoc_fk)`. For `A + B -> R -> C`, R links to derivation
  edges `C -> A` and `C -> B`. Relation types distinguish derivation, grouping
  and primary placement; not every linked DPA source is a scientific output.
- `run.context` retains the full resolved workflow and selected inputs.
  `data_prod_assoc_run.context` holds only edge-specific execution details.
- Publication selects only completed products. Promotion may select an older
  completed version without changing its artifacts. Untagged publication leaves
  all selections unchanged; an empty tag has no current result.
- Readers select a group/tag or pin an exact product reference/index. Run-name
  ordering does not imply the selected result. The default tag is `main`.
- Products commit independently. A completed invocation does not establish that
  every requested output was published. Re-registration preserves identity and
  does not authorize replacing completed scientific contents.
- Run states are `running`, `completed`, `failed`, and `cancelled`. Reservations are
  durable before work starts. Per-output requests record group UID, declared
  product type, primary UID, requested tag, status and failure message in
  `run.context.publications`.
  A failed reservation creates no completed-product placeholder.

#### Metadata ownership

- Stored metadata supports product discovery, workflow replay, execution
  diagnostics and scientific interpretation. Each record retains the information
  owned by its producer at the corresponding scope.
- Paths below describe a KIDs product directory. Scientific paths are relative
  to the selected interface subtree at `array.zarr/<interface>/` for both one
  and multiple interfaces; see the
  [processed-product tree contract](#processed-acquisition-tree).

| Metadata | Relational / logical owner | Filesystem or scientific representation |
| --- | --- | --- |
| Stable client/environment facts | `actor.meta` | `namespace.yaml.actors`, plus referenced actors in product indexes. |
| Namespace identity and counter; inventory path | `data_prod_namespace`; `data_prod_namespace_placement` | `namespace.yaml.namespace`: UUID, projected path, `counter_next`, metadata and audit. |
| Version-group facts and selection | `data_prod.meta`, membership DPAs, `data_prod_tag` | Group root `index.yaml`: declared product type, version inventory and current selections. |
| Realized product facts and relationships | `data_prod`, DPAs, owned sources | `<run.name>/index.yaml`: exact product, inputs and artifacts. |
| Run reservation and invocation facts | `run` and `run.meta` | `namespace.yaml.runs[run_uid]`; realized index carries a publication-time `run` snapshot. |
| Full resolved workflow and selected inputs | `run.context` | Namespace run `context.inputs` and `context.outputs`; realized index `provenance.inputs[interface].config` retains output-specific resolved values. |
| Edge-specific execution details | `data_prod_assoc_run.context` | `assocs[].runs[run_uid]`: execution context and attribution. |
| Acquisition and block metadata | Scientific artifact | Zarr acquisition root, `blocks`, `blocks/<block>` attrs; block metadata includes acquisition fields. |
| Block index and shared axes | Scientific artifact | Zarr `blocks/index`, `blocks/chan`, `blocks/probe` datasets with variable units. |
| Step execution records | Scientific artifact | `tolteca_kids` attrs beneath the owning acquisition or block scope: step/alias, config, completion, failures and selections; excludes live `context.data` and `context.artifacts`. |
| Reducer calculation snapshots | Scientific result dataset | Result attrs such as `blocks/<block>/sweep`; invocation controls remain in step records. |
| Analysis configuration snapshots | Scientific analysis namespace | Namespace attrs under `blocks/<block>`, exposed by its analysis view. |
| Creator and latest modifier | Record audit columns | Portable actor references; no database-local FKs in indexes. |

- Import/export maps these portable records to the relational owners above.
  A realized index is immutable and may record a still-running multi-output run;
  the namespace descriptor records its final invocation state. The member's audit
  creation time identifies its independent publication completion.
- Run provenance belongs to each realized product, including its embedded record
  in the group inventory. Scientific nodes retain interpretation and result
  snapshots. No separate manifest file is required.
- A workflow request does not establish which results were actually produced.
  Saved execution records describe performed steps; installed datasets and their
  snapshots describe available results. Reused results retain their producing
  settings.
- KIDs publication saves the available step records except `DataProdOutput`.
  The run index records publication settings; an output step's live return value
  is not part of the exported scientific tree.
- Publication must not reconstruct a purported full workflow from partial step
  records. Independently invoked steps may have no full workflow snapshot.
- Execution depends on resolved configuration values, input data and the software
  environment. It must not depend on the original recipe or on whether a value
  was explicitly supplied or defaulted. A saved default is an explicit snapshot
  value; changes to later defaults must not replace it during replay.
- Replay uses the recorded input and resolved snapshot. Numerical reproduction
  additionally requires the same input contents and compatible software. This
  contract does not capture arbitrary Python preprocessing or guarantee bitwise
  reproduction from a configuration alone.
- Loading step metadata does not restore live views, open handles or a resumable
  invocation. Scientific views use the stored datasets. Runtime/package records
  currently describe the publication environment; they do not archive input
  contents or a complete software environment.

#### Run identity and name allocation

- A namespace owns one authoritative allocator and durable `counter_next` state.
  A run is reserved once there before processing. The reservation and counter
  advance are atomic; allocator state outlives the lock and process.
- Every run has a stable portable UUID supplied by the caller or generated once
  for the invocation. Import matches this UUID and remaps the database-local PK.
- A run name is a namespace-unique path component. Callers may supply a name or
  request counter generation. An omitted name uses the run UUID as text.
- A different UUID with the same namespace and name is a conflict. Reimporting
  one UUID merges its compatible snapshots; it does not authorize output overwrite.
  A run retains its output namespace identity.
- The counter supports automatic advancement and explicit unused `value` requests.
  Allocated `(namespace, counter_value)` pairs are unique. An explicit value cannot
  rewind the counter or reuse a reservation; explicit non-counter names have no
  counter value. An optional key can produce `{value:04d}-{key}` labels.
- Failed or abandoned reservations remain consumed. Adding another output of the
  same invocation reuses that run and does not allocate another name or UID.
- Copied allocator state does not grant an independently writable replica
  allocation authority. Import must not rewind state or duplicate reservations.
- `DataProdStore.invocation()` reserves once under the namespace lock. The KIDs
  batch driver shares that invocation across acquisitions; publication sessions
  record individual output requests and commits. The namespace descriptor persists
  failed reservations as well as successful runs.

#### Publication ownership and atomicity

- Publication lets cooperating writers share a store while readers continue to
  use completed results. The publication session owns its reservation and staged
  output; the store owns durable inventory and tag updates.
- A tagged publication holds exclusive version-group/tag ownership from reservation
  through completion or failure. Contention fails before the requesting session
  starts, without waiting. Different tags and different products may compute
  concurrently.
- Run reservation serializes namespace allocation. Product commit preserves
  concurrent changes to group associations and selections by rereading current
  state under the publication lock/transaction, not replacing it with a stale
  inventory snapshot. These are separate ownership scopes.
- A producer completes and validates its artifacts before committing them. The
  completed source inventory and requested tag selection become visible together.
- Failed computation or uncommitted publication retains the run reservation and
  leaves the previous tag selection usable. Uncommitted artifacts are not sources
  in the published group inventory. A first unsuccessful attempt leaves no current
  selection. The run's typed publication-request inventory records each requested
  output/group/tag and failure without requiring a completed-product FK.
- Completed products remain unchanged and selectable after later publications
  or tag promotions. Updating a tag does not rewrite the selected product index.
- Cooperating writers must address the same shared store with cross-machine
  exclusion and atomic publication guarantees. Independent filesystem copies do
  not coordinate; publication belongs at their shared destination.
- Filesystem publication does not require a database and is separate from database
  registration. No filesystem/database or cross-product atomic transaction is
  promised.

### Reopening and external use

- Reopening exposes selected scientific results and their provenance without
  rerunning the producing workflow.
- Product consumers select a version through a group/tag or an exact product,
  then an interface within it. A pinned product remains fixed when a tag moves.
  A tag with no completed selection cannot supply scientific data.
- Reopened scientific data retains its own interpretation without recreating the
  invocation that produced it.
- Database registration and Web discovery preserve the same DP identity, DPAs,
  and source ownership.
- Each realized product owns its artifact sources directly. Moving a group's tag
  does not remove earlier products, sources or producing-run records. Incomplete
  reservations do not register completed artifact sources.
- Publication, DPDB ingest and the processed KIDs Web service use scoped group and
  realized-product identities. Web queries can filter by namespace and tag or pin
  a realization/location. `write_data_prod_projection` rebuilds the namespace,
  group, realization and registered singleton indexes without the original YAML;
  local payload copies are optional. A general DP graph viewer remains separate work.
- Archive export uses a pinned run and supplies the archive's required
  metadata and file inventory.
- Product indexes, LMT metadata blocks, and archive dataset indexes are distinct
  interfaces. Detector readout frequency is not an astronomical observing band.

#### Discovery across deployment locations

Two accepted discovery paradigms share portable product identities and artifact
resolution. Neither requires operational databases to modify one another.

| Paradigm | Query owner | State and failure boundary |
| --- | --- | --- |
| Query federation | Web queries attached DPDB providers | Providers remain authoritative and unchanged; responses identify provider freshness and unavailable providers. |
| Materialized unified inventory | Web queries a separate unified DPDB | Taco and Unity operational DPDBs are independent sources. The unified inventory is a disposable, rebuildable read projection. |

- A deployment location owns a software environment and operational services.
  A DPDB storage location identifies an artifact root. A query provider identifies
  an attached inventory. These are separate concepts.
- Each independently maintained artifact tree has its own operational inventory.
  Taco processing, Unity processing, the transferred Taco tree on Unity, and
  contributed user trees have separate DPDBs. A transfer registers completed local
  copies in the copied-tree DPDB, not in Unity's native processing DPDB.
- A copied-tree DPDB preserves product, run and namespace identities. It owns a
  separate registration journal because its subsequent writes and availability
  are independent of the source. A contribution may supply its DPDB or rebuild
  an inventory by ingesting its portable indexes.
- The unified DB consumes all participating inventories. It must not own unique
  operational facts: purging it must not discard a processing result or transfer
  registration. A newly published remote product does not imply a local copy.
- Federation qualifies DB-local keys by provider. A displayed provider qualifier
  is separate from the namespace path and does not specify artifact transport.
  Bounded provider queries, global pagination/deduplication and partial failures
  require explicit contracts before exposing a federated Web endpoint.
- Inventory-specific namespace placements remain distinguishable: `taco:ql`
  denotes Taco's `ql` inventory; `unity:taco/ql` denotes Unity's independently
  registered copy at `taco/ql`. The colon qualifies the inventory, not the
  namespace UUID or artifact URI. Neither inventory implies the other's contents,
  availability or current selections. A unified projection must retain these
  distinctions even when portable product identities coalesce.
- A materialized inventory replays portable registration facts. It does not rerun
  workflows, reproduce SQL mutations or depend on historical Python contexts.
  Bootstrap/rebuild exports current source inventories; live ingestion must select
  changed facts and affected association owners without scanning full history.
- Unified inventory construction is a workflow whose inputs are participating
  DPDB inventories and whose output is a separate DPDB. It may derive more complete
  records than any individual input. Source inventories remain unchanged. For
  compatible contributions, incremental construction must agree with a fresh
  rebuild from the same facts, independently of provider arrival order.
- `DataProdRegistration` carries product and namespace indexes grouped by reporting
  inventory. Plain indexes belong to the receiving database's local inventory;
  wrapped inventory records preserve their reporting UUID through further imports.
  Registration creates supplied endpoints before relationships, accepts graph
  cycles, and resolves portable identities to destination-local keys. The caller
  owns one transaction. Unknown targets and incompatible identities fail that batch.
- Compatible contributions retain sources and associations; calibration interface
  evidence accumulates. Absence from a batch does not delete destination records.
  This does not establish arbitrary cross-provider metadata precedence. Known
  calibration identity conflicts fail; source-metadata and availability authority
  remain unresolved. Arrival order is not an authority rule.
- `dp_raw_obs` and `dp_obs` aggregate partial knowledge. Merging inventories must
  retain the union of their source/interface contributions and raw-observation
  membership. Aggregate metadata belongs to the aggregation type and must be
  derived from the combined contributions. Different partial summaries are not
  integrity conflicts; audit time must not choose one summary over another.
- Acquisition start is the earliest known interface start and may change as more
  interfaces become known. A starting-time summary can be merged with another
  such summary by taking their minimum. Acquisition-kind flags similarly combine
  by union. Other aggregate fields require their own collation rules; non-null
  replacement is not a general aggregation contract.
- Observation start is the earliest known start among its member acquisitions.
  Adding a member or enriching an existing member's start must update this
  summary. Updates must preserve observation annotations and existing membership,
  and must only revisit observation owners affected by the contributed acquisitions.
- Telescope-header and LMTMC metadata currently use different representations.
  Their enrichment is permissive: supplied non-null raw-observation fields update
  the summary without an integrity conflict, even when values differ. The explicit
  time and kind aggregation rules still apply. This interim behavior does not
  establish source precedence or order independence for the other telescope fields;
  a normalized metadata bridge and their collation rules remain deferred.
- Integrity reporting compares corresponding underlying facts about the same
  entity. Conflicting facts must produce a warning with identity, differing
  values and available attribution/source evidence. Skip the conflicting
  contribution while preserving independent compatible contributions. Source
  enrichment and representation/precision differences must be distinguished
  from contradictory facts before rejecting an update.
- Integrity issues are distinct from realization lifecycle and version selection.
  The first implementation reports them through Tollan logging. A persistent
  issue store, Web presentation, record markers and their resolution lifecycle
  remain proposals; no lifecycle state or version tag represents a conflict.
- Namespace prefix and storage-root mappings describe destination placement.
  Namespace UUIDs, run UUIDs, product UIDs and input provenance stay unchanged.
  Copies add accessible sources to existing products; independent reductions
  retain separate namespace/run identities and join the same logical version group.
- Import may explicitly rename source tag labels, for example `main` to
  `taco:main`. Unlisted labels remain unchanged; qualification is not automatic.
  Duplicate destination labels in one imported index are an error. The unified
  workflow may maintain `main` from a selected source label, such as `unity:main`,
  by selecting its membership DPA directly. An empty selection clears the derived
  label; an absent source label leaves it unchanged. Source arrival order does not
  determine this configured preference.
- Operational commits must not depend on availability of the unified DB.
  `Database.session()` commits affected portable records to a source-local
  registration journal in the same transaction as the inventory changes. The
  journal has a stable UUID and monotonically increasing committed sequence.
  Dagster can schedule imports; CLI and Python invoke the same workflow.
- Implemented: index-file and DB-refresh registration share the batch boundary;
  explicit selected-product export is bounded by that selection. Full refresh is
  for bootstrap and reconciliation, not a recurring live job.
- Capture includes changed rows with their owning product metadata from native
  ORM transactions, including source-only, association-only and run-only updates.
  Updating a tag or adding one membership must not export unchanged memberships.
  Capture must include earlier acquisitions affected by newly resolved calibration
  references. It must not scan unrelated product or invocation history.
- A captured batch becomes available only after its transaction commits.
  Rollbacks discard uncommitted capture; later failures retain earlier committed
  batches. Releasing a nested transaction must not publish inventory changes
  independently of the outer transaction and its batch. The optional in-memory
  capture and workflow return values remain
  available; durable batches are the restart boundary.
- `data_prod.unify` owns a destination bind and a typed list of local source binds
  or remote registration APIs with placement mappings. First contact imports a
  consistent source snapshot and
  records its journal sequence in the destination transaction. Later invocations
  apply only subsequent committed batches, in order, using bounded pages. Each
  page and its applied sequence commit together. Duplicate delivery is skipped;
  failure leaves the last committed sequence available for retry. A missing
  sequence or a source restored behind applied progress is an error.
- Source snapshots explicitly establish a SQLite read transaction; inventory and
  journal boundary must describe the same committed state. Destination importers
  serialize progress checks and registration using SQLite's writer lock. Workers
  overtaken by a newer import confirm current source progress before
  treating a lower snapshot sequence as lost history.
  Import transactions do not re-journal the unified projection. Placement is
  recorded with source progress; changing it requires rebuilding a fresh destination.
  Rebuild uses a new initialized DB and the same configured sources. The workflow
  never purges an existing DB. Journal retention/compaction and
  arbitrary SQL/bulk/deletion capture are outside this slice. A journal identifies
  one writable history: independently writable forks require a new inventory.
- Remote delivery exposes a configured operational DPDB through read-only status,
  consistent snapshot and bounded contiguous journal-page operations. The transport
  uses the existing portable registration model. Requests bind to the expected
  journal UUID and a fixed committed horizon; replacement, missing history and
  backward progress must fail. Network reads must finish before the destination
  acquires its writer lock. A competing importer may advance during a request;
  the importing transaction must recheck progress and discard stale fetched work.
- CLI, Python and Dagster invoke the same import workflow. Dagster provides a
  source-partitioned asset and readiness sensor. Destination DPDB progress is the
  delivery acknowledgement; scheduler cursors and run success are not substitutes.
  The sensor must detect manual source commits, retry unapplied work after failed
  runs or scheduler restart, and continue other sources during a source outage.
  Source inspection may read current sequence and destination progress only; it
  must not export an inventory or scan historical products per tick.
- Quality annotations and general lifecycle/deletion state are not in the portable
  inventory. A production rebuild cannot claim full DPDB equivalence until those
  domains and mutable-record authority are covered. Calibration connectivity and
  redirects are derived from portable per-interface facts.

#### Remote discovery and access

- Users discover products across the observing site, Unity storage and local
  workspaces, then read selected scientific data without downloading an entire
  product. Local and remote readers preserve the same domain views, values,
  units and resource-ownership contracts.
- Discovery queries provide product identities, relationships, available runs
  and accessible source references. DB-backed readers obtain these descriptors
  from database records without fetching `index.yaml`. A reader can also open
  a known published index directly without database access.
- Remote discovery uses a query API. Published indexes and artifacts are
  independently readable through HTTPS. The API and stored indexes use the
  shared product models; scientific selections use the domain readers.
- Tag resolution selects one association and its completed target product before
  opening artifacts. That exact product remains pinned throughout the read, even
  if the tag advances.
- Artifact references resolve through their registered locations. For a
  filesystem index, relative location roots resolve against that index's URI.
  Recorded input provenance and local cache paths do not determine the current
  artifact location.
- Filesystem-index ingestion accepts the same explicit namespace, tag and
  storage placement as portable inventory import. Registering transferred
  indexes through Unity's operational DB must retain source product/run UUIDs,
  add the copied locations, and publish those additions to the unified workflow.
  Repeating unchanged registration must not append a journal batch.
- Namespace paths organize products independently of artifact access. Neither
  `:` nor `/` in a namespace name denotes remote or filesystem storage. Source
  choices expose their registered location and access method; filesystem access
  is relative to the reader. A transferred copy adds a source without changing
  product identity. Cached remote objects remain part of the remote source.
- Product identity and source references cross deployment boundaries;
  database-local primary keys do not identify products or sources portably.

#### Metadata availability and read caches

- The operational site processes incoming acquisitions, materializes products
  and populates its DPDB as observing proceeds. Unity must present these timely
  site results together with products available through NESE storage.
- Site discovery is independent of bulk artifact transfers. Federation queries
  the site provider; a materialized unified inventory retains imported metadata.
  Both can expose a completed site product before its payload reaches NESE.
- Retained discovery metadata can support browsing during site outages. A
  federated view must report an unavailable provider explicitly; it may only show
  cached metadata with its freshness. Neither retained metadata nor a reachable
  unified DB guarantees access to uncached remote artifact chunks.
- Site and Unity ingestion use the shared product identities and associations.
  Transferred artifacts become accessible sources after publication at their
  destination; registering a source does not copy its payload. A NESE copy of
  an existing result adds a source to that product, not a new realization.
  Independently computed Unity results retain their own run and product identities.
- The defined filesystem projection places transferred site products beneath
  `dataprod_toltec/taco/{ql,scratch}` and independent Unity results beneath
  `dataprod_toltec/unity/{ql,scratch}`. Import maps a site's namespace path to
  its destination prefix while preserving its namespace UUID. Unity processing
  owns different namespace UUIDs. Both paths associate with the same global
  observation/acquisition products. Prefixes do not enter product or run UIDs.
- Transferring site products and transferring raw files for Unity processing are
  independent operations. The combined catalog must support site-only products,
  locally copied site products, and independently computed Unity products
  simultaneously. Copy registration preserves the site's run identity and uses
  explicitly qualified import tags. Version groups remain global; destination
  preference selects `main` independently of the artifact's access location.
- Refresh of site-owned products and tag selections preserves Unity-local source
  registrations and independent results. Resolving a tag pins the observed
  completed product; later refreshes do not change an active selection.
- Browser visualization reads scientific data through the Web service. Local
  Python clients can use the shared remote reader without first importing a
  local database. Offline work can use locally ingested products.
- A cache stores disposable copies of retrieved objects. It does not create a
  new product, registered source or complete local replica.
- Mutable inventories and tag selections require refresh. Cached completed-run
  artifacts remain tied to their selected source and run. Cache identity must
  not collapse different sources or runs into the same tag or acquisition key.
- Read preparation may load small discovery metadata and block indexes eagerly;
  numerical arrays remain lazy until selected by their consumer.
- Processes sharing a local HTTP object cache must see only completed downloads.
  An interrupted read must remain retryable without exposing partial cached
  arrays. Cached selections remain readable when their origin is unavailable;
  unavailable uncached objects remain errors.
- KIDs product publication owns channel/time chunk settings. They bound the
  signal data fetched for channel pages and time windows while preserving
  scientific values, axes and units. Stored output settings describe the chosen
  layout; changing them affects only a newly published run.

The [remote-access design](data-product-remote-access.md) records deployment,
transport and cache choices, including the accepted division between run-index
provenance and scientific metadata. Publication has no separate manifest artifact.

## Processing workflows

- Processing workflows combine input selection, scientific operations, and output
  for repeatable interactive or operational runs.
- Scientific domains own recipe composition and operation prerequisites.
  Shared workflow execution supplies configuration and invocation control for
  those recipes; scientific operations remain independently callable.
- TolTECA workflows define their generated artifacts and product contents.
  Shared publication infrastructure materializes those definitions; database
  registration and scheduling do not define alternative scientific outputs.
- Each workflow accepts its required input collection and effective settings.
  Raw-acquisition ingestion and quicklook must both be callable from Python and
  CLI without Dagster. Their computation does not depend on who invokes them or
  which discovery source supplied compatible inputs.
- Quicklook defines processing and feedback artifacts for selected acquisitions.
  Input preparation, execution scheduling and result registration have separate
  owners; raw-acquisition ingestion is not a prerequisite for quicklook execution.
- A workflow invocation executes on selected inputs and may produce several
  [products in a run](#runs-and-tags). Execution diagnostics, scientific outputs and
  publication state have distinct lifecycles.
- Processing supports in-memory results without publication or DPDB access.
  When output is enabled, publication writes the product contents, metadata and
  portable indexes to the configured store. DPDB registration is an optional,
  explicitly configured output operation. The same registration operation must
  also accept saved output indexes without recomputing the product.
- Publication and registration have separate completion states. Registration
  failure must preserve committed files and their identity. Batch completion
  updates the registered run after the batch exits, including failure exits.
- Registration recovery selects the producing run UUID. A moving version tag
  must not substitute another realization, import unrelated runs, or alter a
  newer tag selection. Filesystem inventories and DB commits remain separate;
  the output context records publication and registration outcomes.
- Service-owned recipes provide integrated entry points for external use cases
  by composing application workflows and settings. Those recipes accompany the
  owning service's API and tests. Deployment projects them into a location's
  `run/<service>/`; it does not own their scientific or application logic.
  The Web service's [deployment contract](../../tolteca_web/design/kids-block-viewer-spec.md#deployment-profiles)
  defines that integration for `run/tolteca_web`.

### Raw acquisition discovery and DPDB workflow inputs

- Raw acquisition discovery accepts file-store queries, globs and explicit file
  locations, or acquisition records from ToltecDB. Live acquisition discovery
  uses ToltecDB `Valid` to identify upstream completion.
- DPDB is the authoritative hub for data-product discovery and higher-level
  workflow input selection. It is one supported route for preparing quicklook
  inputs, alongside file discovery and ToltecDB acquisition discovery.
- Raw-acquisition ingestion is a workflow that registers or enriches discovered
  acquisition records, sources and intrinsic associations in DPDB. Workflows
  using TolTECA subpackages populate DPDB with their generated product records.
- Input preparation converts discovery results into the collection required by
  the selected workflow. The same quicklook workflow accepts inputs prepared from
  explicit files, file-store selections, ToltecDB records or DPDB `dp_raw_obs`
  records. It must not query DPDB merely to rediscover inputs already supplied.
- Workflows retain their domain-specific prerequisites. Ingestion writes DPDB;
  quicklook processing and publication do not require prior raw ingestion.
  Registration makes published products discoverable through DPDB.
- Web and the quicklook API discover available results through DPDB. Raw arrays
  and generated artifacts remain in their registered storage locations.

### Operational acquisition and product triggers

- Dagster models durable data assets and their partitioned dependencies. Sensors
  observe upstream systems; declarative automation schedules dependent assets.
  Application workflows must remain callable without Dagster. An op graph may
  implement a graph-backed asset when independently retriable internal stages
  serve the materialization of that asset.
- A partition identifies a repeatable unit of data, not a processing attempt.
  Run UUIDs distinguish attempts. A processed acquisition publishes its selected
  interfaces together; adding interface partitions must not split that atomic
  product into competing writers.

- Operational quicklook responds promptly to completed raw acquisition files.
  The site retains a Dagster sensor on ToltecDB's `Valid` state as the upstream
  acquisition-completion signal, following production `toltecpipe`.
- Dagster sensors identify changed records, prepare workflow input requests and
  dispatch the applicable workflows. A ToltecDB readiness event can independently
  trigger raw-acquisition ingestion and quicklook. Quicklook must not wait for
  that ingestion workflow or for a subsequent DPDB sensor tick.
- Valid acquisition records must remain eligible for logical ingestion while
  their files are unavailable. File readiness gates quicklook, not ingestion.
  Later file arrival must enrich the ingested record with authoritative header
  metadata and source availability without duplicating its identity.
- Enabling either automation path must account for ready, unmaterialized
  partitions whose readiness events already exist. Service startup order must
  not discard work. Restarting services must not recompute completed partitions
  without a new input event or an explicit materialization request.
- DPDB sensors independently dispatch workflows from committed product changes,
  including normal CLI and Python writes. Producers must not require a Dagster
  call. Discovery must resume across restarts and bound work to unread changes
  and affected acquisitions, rather than rescan retained product history.
- DPDB-driven quicklook uses registered source availability. Missing files become
  eligible when ingestion or refresh commits their availability. Every currently
  known KIDs interface must have an available source; subsequent interface
  arrival can request the acquisition again. This route must not schedule raw
  ingestion as a prerequisite or feedback loop.
- Duplicate delivery, aggregate-metadata updates and result registration must not
  rerun unchanged inputs. Explicit materialization remains available for retries
  and changed processing settings. A replaced source identity or missing delivery
  history must fail visibly without acknowledging unseen changes.
- Switching quicklook between ToltecDB, DPDB and manual file discovery changes
  input preparation and triggering, not the quicklook computation.
- Workflow operations remain callable from Python and CLI independently of
  Dagster. Sensors and jobs invoke those operations; they do not own a separate
  implementation of input resolution, artifact construction or registration.
- Transfer runs have an independent job and readiness sensor. A transfer commits
  its copied-tree inventory after successful copying; unified-inventory imports
  consume that DPDB's committed journal. A failed transfer must not acknowledge
  completion or advertise uncopied artifacts as locally available.
- One operational deployment can register multiple code locations in one Dagster
  instance. Code locations own workflow definitions and execution environments;
  they do not establish DPDB ownership. Taco and Unity operate independently.
  Separate jobs, sensors and execution capacity keep quicklook independent of
  transfer and general DPDB work.
- Operational quicklook has an independent execution path from general
  DPDB-driven workflows. Code-location, worker and instance placement are
  deployment choices; they must not introduce a general-ingestion dependency
  into the workflow's input contract.
- Existing logical acquisition records remain eligible for enrichment when more
  interfaces or authoritative metadata become available. An existing product
  identity alone must not suppress ingestion updates.
- Cross-site discovery consumes operational DPDB records, through federated
  queries or registration into a separate unified inventory. The site's ToltecDB
  acquisition sensor is not a product-discovery source. The two paradigms follow
  [discovery across deployment locations](#discovery-across-deployment-locations).
- Acquisition progress, derived-product execution and cross-deployment import
  progress have separate lifecycles. Dispatching a job does not establish that
  its ingestion or synchronization transaction committed.

### Unity service and compute placement

- Unity's service container hosts continuous services, sensors, and workflow
  supervision. Significant data reduction runs in Slurm allocations on compute
  nodes. Manual QA follows the same placement rule after login through Unity's
  submission host.
- Automated and manual processing invoke the same application workflows and
  publication operations. Dagster is not required to produce or register a result.
- Shared `/nese/toltec` paths provide access to inputs and committed artifacts.
  Database access must additionally satisfy the backend's concurrency and storage
  requirements; a shared mount alone does not establish those requirements.
- An external attempt retains its resolved inputs/settings, producing run UUID
  and scheduler reference. Reattachment preserves the attempt; recomputation
  creates a new one. Completion requires scheduler success and committed outputs
  belonging to that attempt. Registration can resume from those outputs.
- Explicit cancellation stops remote computation. Interrupted supervision alone
  must not cancel the job or authorize another submission with an unknown outcome.
- Deployment execution policy owns placement, resource bounds and permitted
  overrides. Scientific workflows retain their processing contracts independently
  of that policy. Persist resolved execution settings with the attempt so queued
  work and recovery do not silently follow changed deployment defaults.
- Workflow admission limits must distinguish live processing, bulk work and QA
  where they compete. Cluster enforcement must cover manual submissions and
  external jobs whose supervisor is no longer running; Dagster concurrency alone
  does not establish a cluster-wide resource limit.
- Unity execution transport, the tested Docker Slurm prototype, and remaining
  deployment/QA questions are recorded in
  [Unity workflow execution through Slurm](unity-slurm-execution.md).

### Configure and run processing

- Configured processing runs a domain recipe over a selected batch with settings
  matched to each input.
- **Handoff:** [candidate collection](#candidate-discovery) and configuration
  sources → [selected inputs](#processing-input-selection) with
  [effective settings](#configuration-composition) →
  [domain steps](#steps-and-scientific-results) → scientific results and
  [invocation contexts](#inspect-execution-diagnostics).

#### Processing input selection

- A workflow selects its inputs from a prepared candidate collection.
- Header-dependent queries use authoritative metadata.
- Batch selection preserves the shared collection and its resource ownership.
- Input selection defines the batch to process. Per-input conditions choose
  settings for members of that batch; they do not define the batch itself.

#### Configuration composition

- User-configuration models follow
  [Tollan's validation and persistence contract](../../tollan/design/spec.md#runtime-context-and-component-settings).
- Configuration supports shared defaults, settings from multiple sources, and
  invocation overrides through [Tollan's configuration contract](../../tollan/design/spec.md#application-configuration).
- Effective settings are the settings applicable to one selected input after
  composition and overrides.
- Selected input metadata supplies evaluation context for per-input settings.
  Live source handles remain in the input resource scope; configuration and
  retained input metadata must not carry them.
- Scientific operations own their numerical settings and defaults. Workflow
  configuration selects and composes those settings through the operations'
  configuration contracts.
- DB-backed workflows use the root sections `db`, `data_prod`, and `web`, with
  one component handler per section. Nested models configure each operation;
  named references select connections. Handlers compose the existing
  `input_discovery` and `kids` components without relocating their settings.
  Ownership and resource lifetimes follow the
  [DB/Web configuration contract](db-web-configuration.md).

#### Python and command-line entry points

- Python and command-line entry points support interactive and operational use
  of the same configured workflow.
- Input resources follow the shared
  [resource-ownership contract](#data-access-and-resource-ownership).

#### Check a run without processing

- A no-op run exercises input preparation, selection, configuration, and invocation
  without numerical processing or product publication.
- No-op execution preserves selected input order and reports completion for each
  input, including those sharing an acquisition.

#### Steps and scientific results

- Steps connect independently usable scientific operations into a workflow over
  selected acquisition or block data.
- Steps adapt domain operations to [Tollan's invocation contract](../../tollan/design/spec.md#steps-and-pipelines).
  Scientific results remain governed by the operation's domain contract.
- Downstream steps can use available results and invocation outcomes to determine
  whether their own input requirements are met.

#### Sequential execution and failures

- Sequential execution provides a lightweight way to order dependent operations.
  Operation contracts do not require that particular scheduling driver.
- The configured sequence determines execution order.
- A recorded failure may leave usable inputs for later operations. Each consumer
  defines which available results or alternatives it accepts.
- An unhandled error stops the sequence under
  [Tollan's step contract](../../tollan/design/spec.md#steps-and-pipelines).
- Completion is assessed against the requested operations; it does not require
  unrelated scientific stages to have run.
- Sequential execution does not infer a dependency graph, propagate downstream
  recomputation, or merge partial results from earlier product runs. An alternative
  DAG driver and partial reprocessing policy remain outside this contract.

### Publish and reuse scientific results

- `DataProdOutput` publishes an invocation's available scientific results for
  subsequent inspection and analysis.
- **Handoff:** scientific results and producing provenance → `DataProdOutput` →
  published product → [run and interface selection](#reopening-and-external-use)
  → domain data and result views.
- It follows the [scientific-content](#scientific-contents-and-representations)
  and [publication](#publication-ownership-and-atomicity) contracts.
- Required results are validated before publication. Optional analyses contribute
  their available outputs without imposing unrelated stage requirements.
- The KIDs acquisition workflow publishes selected interfaces together as one
  `dp_raw_obs_proc` run, preserving every identified block's reduced data and the
  acquisition metadata, block index and shared axes needed to interpret it.
  Interface-specific contents remain within that product.
- KIDs runners reserve the output and acquire the tag before block processing.
  Direct output-step use starts publication when that step is invoked; it does
  not claim tag ownership during prior caller-managed computation.
- Reuse and force settings act on scientific results in the working data. They
  do not authorize replacing completed products or reusing output reservations.
- An existing product enters at reopening. Database, Web, and archive consumers
  use the [external handoff contracts](#reopening-and-external-use).

### Inspect execution diagnostics

- Invocation diagnostics support inspecting an operation after it has run.
- **Handoff:** step invocation → recorded context → diagnostic inspection in the
  current or another process.
- An invocation context records its effective settings, outcome, failures, and
  diagnostic data.
- Contexts can be saved and loaded in another process for further diagnostics.
- Diagnostic transport does not establish workflow checkpoint or resume support.
