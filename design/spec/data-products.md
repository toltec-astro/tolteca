# Data products

Users retain product descriptions, artifact references and execution records for
reuse across observing-site, archive and local workspaces. The portable core owns
these records and filesystem namespace/run persistence. Applications supply their
product metadata, identity rules and publication operations. Database projection
and scientific processing remain separate consumers.

## Portable records

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

## Sources and persistence

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

## Actors and record attribution

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

## Namespaces and executions

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

## Run names and interrupted allocation

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
