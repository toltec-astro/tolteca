# TolTECA architecture

Settled implementation mechanisms behind the specification
(`spec/index.md`). Describes the current code.

## Data product store

- A live filesystem group commits its member count and tags by replacing the
  root index under its publication lock. Each `.members` record identifies a
  realized run directory and its membership association. Publishing a new
  version requires no earlier realization reads or rewrites. A full portable
  export expands the committed records into `products` and `assocs`. A pinned
  registration reads only the requested members. `tags[label].current` retains
  the exact product and namespace UIDs.

- A committed group header must describe a version group. Available members
  identified by its membership associations must have the group's declared
  product type. Supporting products are not members unless an exact membership
  reference identifies them; referenced members need not be embedded for transport.
- Membership records are written twice: by position, `.members/<n>.yaml`, and
  by the member's product UID, `.members/by_uid/<percent-encoded UID>.yaml`. A
  tag finds its member through the UID record; a selection by run composes the
  member UID `{version group UID}/{run UID}`, the store's own identity rule. A
  UID record is committed only when it equals the positional record within the
  header's member count.
- A committed lookup must identify the requested member and agree with its
  realized product, which records the same version group. Inconsistent committed
  records raise an error. Absent or uncommitted requested members are omitted
  without scanning other runs.
- Index documents carry their kind in `document`: `data_prod_index` for a
  product index, `version_group_header` for a live group header with its
  `version_count`. One parser validates either kind.

- The counter supports automatic advancement and explicit unused `value` requests.
  Allocated `(namespace, counter_value)` pairs are unique. An explicit value cannot
  rewind the counter or reuse a reservation; explicit non-counter names have no
  counter value. An optional key can produce `{value:04d}-{key}` labels.
- Interrupted allocation may leave claimed names or counter values without a
  saved store-run record. These claims remain unavailable; the store does not
  recover or reclaim them. A claimed generated name must also have its counter
  value claimed. A retry reuses a saved reservation; otherwise automatic naming
  allocates a new one and a claimed fixed name fails. Allocation does not scan
  run history.
- A user may remove an abandoned fixed-name claim when no writer is active and
  no saved reservation or published output uses it. This releases only the name;
  it does not rewind the counter or remove a completed reservation.

- Run reservation serializes namespace allocation. Product commit preserves
  concurrent changes to group associations and selections by rereading current
  state under the publication lock/transaction, not replacing it with a stale
  inventory snapshot. These are separate ownership scopes.

- The current index schema nests product content under `product`; supporting
  records remain at the document level. Development indexes use format 1;
  obsolete test outputs may be regenerated without compatibility readers.
  Core imports must not load the
  concrete product package, instrument registrations or `tolteca_db`.

- `{namespace.path}/namespace.yaml` stores the namespace record and actors, without
  embedded run history. Its namespace record carries `counter_next`; independent
  run files retain invocation states. A namespace index can include selected runs
  for transport. Group indexes carry namespace references on their realized
  members, never an independent allocator.

- A singleton or version group remains global. Its registered index representation is a source with
  `meta.format=data_prod_index` and `meta.namespace_uid`, making namespace visibility
  reconstructable without assigning a namespace to the singleton itself.

- A singleton descriptor locates its file under the namespace root, expressed
  relative to the descriptor’s directory on disk. Publishing
  it in DPDB and importing that file use the same location and filename. Publishing
  the singleton in another namespace adds a distinct representation.

- Run states are `running`, `completed`, `failed`, and `cancelled`. Reservations are
  durable before work starts. Per-output requests record group UID, declared
  product type, primary UID, requested tag, status and failure message in
  `run.context.publications`.
  A failed reservation creates no completed-product placeholder.

| Metadata | Relational / logical owner | Filesystem or scientific representation |
| --- | --- | --- |
| Stable client/environment facts | `actor.meta` | `namespace.yaml.actors`, plus referenced actors in product indexes. |
| Namespace identity and counter; inventory path | `data_prod_namespace`; `data_prod_inventory_namespace` | `namespace.yaml.namespace`: UUID, projected path, `counter_next`, metadata and audit. |
| Version-group facts and selection | `data_prod.meta`, membership DPAs, `data_prod_tag` | Group root `index.yaml`: declared product type, version inventory and current selections. |
| Realized product facts and relationships | `data_prod`, DPAs, owned sources | `<store_run.name>/index.yaml`: exact product, inputs and artifacts. |
| Invocation facts and outcome | `run` and `run.meta` | `.runs/by_uid/<run_uid>/run.yaml`; realized index carries a publication-time `run` snapshot. |
| Store-run record and output outcomes | `store_run` | `.runs/by_uid/<run_uid>/store_run.yaml`; destination-local name, counter and per-product materialization receipts. |
| Full resolved workflow and selected inputs | `run.context` | Namespace run `context.inputs` and `context.outputs`; realized index `provenance.inputs[interface].config` retains output-specific resolved values. |
| Edge-specific execution details | `data_prod_assoc_run.context` | `assocs[].runs[run_uid]`: execution context and attribution. |
| Creator and latest modifier | Record audit columns | Portable actor references; no database-local FKs in indexes. |

- Import/export maps these portable records to the relational owners above.
  A realized index is immutable and may record a still-running multi-output run;
  the independent run record retains its final invocation state. The member's audit
  creation time identifies its independent publication completion.

- A namespace owns one authoritative allocator and durable `counter_next` state.
  A saved store-run reservation is reused for its execution. Allocation is
  serialized; allocator state outlives the lock and process. Counter values need
  not be continuous.

- `WorkflowRun.create()` creates one destination-independent execution record.
  The KIDs batch driver shares it across acquisitions. `DataProdStore.materialize()`
  reserves its store-run name under the namespace lock and records each
  output request and completion. `write_run()` saves supplied execution snapshots.
  Store-run records retain failed reservations as well as successful outputs.

| Term | Meaning and scope |
| --- | --- |
| Counter key | Optional context in a generated name, such as `{value:04d}-{key}`; not execution identity on its own. |
