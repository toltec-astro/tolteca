# TolTECA software terminology

The [TolTEC Data Handbook glossary](https://toltec.lmtgtm.org/dr/science/docs/handbook/glossary.html) defines the scientific terms shared across packages. Use the guides below for the software that handles each task.

## Software concepts by domain

Follow the guide for the operation you want to perform. These pages introduce
software objects together with their inputs, outputs and examples.

| Concepts | Guide |
| --- | --- |
| Source, identification location, filename information and reader metadata | [Read one acquisition](read-acquisition.md#sources-locations-and-metadata) |
| Model table, file accessor and data accessor | [File tables](file-tables.md) |
| ObsSpec | [Observation selectors](obs_spec.md) |
| File store | [File stores](file-stores.md) |
| Input discovery | [Discover inputs](input-discovery.md) |
| Data-model membership, specification and capability selection | [Choose registered or explicit capabilities](file-tables.md#choose-registered-or-explicit-capabilities) |
| Blocks, probe axes and processing steps | [KIDs reduction](kids-reduction.md) |
| Product, artifact, item, index, reference and publication | [Data products](data-products.rst) |
| Named collection, version group, realization, tag and exact contributors | [Collections and results](named-group-products.rst) |
| Namespace, store and workflow run | [Product stores and runs](data-product-runs.rst) |
| Inventory, product associations and database registration | [DPDB product indexes][dpdb-guide] |
| Job and execution backend | [Execution guide][execution-guide] |
| Deployment location and service profile | [Deployment configuration][deployment-guide] |

A source catalog contains astronomical sources. A file **model table** contains
records used to select data for processing. Use the qualified term when both
occur in the same workflow.

Publication of a processing result, registration of its records in DPDB and an
archive release are distinct operations. Availability through a store or database
does not by itself establish scientific quality or approval for release.

[dpdb-guide]: https://toltec.lmtgtm.org/dr/science/docs/tolteca_db/product-indexes.html
[execution-guide]: https://toltec.lmtgtm.org/dr/science/docs/tolteca_deploy/execution.html
[deployment-guide]: https://toltec.lmtgtm.org/dr/science/docs/tolteca_deploy/config.html
