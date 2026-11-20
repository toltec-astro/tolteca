# Observation data

`tolteca_datamodels` makes acquisition files and their observing context available
for inspection and input selection. The tools form a stack of four levels. Start at the
level that matches what you know about your files, and move up when you need to
compare many files, search a data tree, or configure a workflow.

## Concepts

| Term | Meaning |
| --- | --- |
| Observation | Observing activity identified by an observation number within an acquisition master, for example `tcs-145609`. |
| Acquisition | The files recorded by the participating interfaces, identified by master, observation number, subobservation number and scan number, for example `tcs-145609-0-1`. |
| Interface | One producer's contribution to an acquisition: the telescope `tel`, a detector readout such as `toltec0`, or the half-wave plate `hwpr`. |
| Source | A data resource to read: a path, a file location or an already-open object. An astronomical source is the observed target, not this. |
| Model table | A pandas DataFrame whose columns hold information collected from model objects. A file table has one row per source; it is not an astronomical source catalog. |
| File accessor | Prepares a file table and operates on it: read metadata, query, open. |
| Data accessor | Exposes one opened source's metadata and arrays, for example `.lmt_tel`. |
| Data model | A named set of capabilities, such as `lmt` or `toltec`: filename conventions, file accessors and file stores. A row's `data_models` lists the models that can interpret its data. |

## Choose a level

| Level | You have | Python entry points | Configuration and command line | Guide |
| --- | --- | --- | --- | --- |
| 1 | One file whose format you know | `open_lmt_tel`, `open_toltec_kids`, `open_toltec_hwpr`; `FilepathInfo.from_source` | | [Read one acquisition](read-acquisition.md) |
| 2 | Several known files to compare or select | `LmtFileAccessor`, `ToltecFileAccessor`, `FileAccessor`: `.from_sources()`, `.query()`, `.query_obs_spec()`, `.open()` | | [File tables](file-tables.md) |
| 3 | A data tree to search | `LmtFileStore`, `ToltecFileStore`, `query_file_store()` | | [File stores](file-stores.md) |
| 4 | A recipe shared by programs | `InputDiscovery(runtime).discover()` | `input_discovery:` in YAML; `tolteca_kids --show_inputs`, `tolteca_deploy run` | [Discover inputs](input-discovery.md#describe-discovery-in-a-recipe) |

Each level builds on the one below. A store finds candidate paths for a file accessor;
`InputDiscovery` combines stores, file accessors and selectors, driven by
configuration; and the command-line programs take such a recipe and pass the resulting
table to their pipeline. A file table is the working object at levels 2 to 4.
Use its `open()` context to work with the selected files through their data accessors.

Levels 2 to 4 select files with [observation selectors](obs_spec.md) such as
`tcs-145609/{tel,0}`, which constrain observation and interface identifiers
independently of any instrument.

## More about observation data

| Task | Page |
| --- | --- |
| Write observation selectors such as `tcs-145609/{tel,0}` | [Observation selectors](obs_spec.md) |
| Look up classes, signatures and fields | [Observation data API](datamodels-api.rst) |
| Understand how the framework is built, for developers who add data models | [Framework architecture](datamodels-architecture.rst) |

## Continue into processing

Input selection leaves a model table ready for its consumer. Detector reduction
belongs to [KIDs workflows](kids-inputs.md); reusable results and their relationships
belong to [data products](data-products-api.rst).
