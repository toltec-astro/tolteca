Data products
=============

Products and their contents
---------------------------

A product identifies the observation, grouping or processing result you want
to describe. Its record connects that identity to metadata, data items and
relationships. Use these distinctions when reading or preparing an index:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Term
     - Meaning
   * - Product
     - An identified entity or result, such as a raw acquisition or a processed sweep.
   * - Artifact
     - Data retained for exchange or persistence: arrays, tables, plots or reports.
   * - Item
     - A constituent of a product with its own metadata and source locator.
   * - Product index
     - The document describing a product and the records and locations needed to interpret its references. The scientific payload is stored separately.
   * - Product reference
     - The product UID with its applicable namespace UUID. A path or database-local integer does not replace this identity.
   * - Association
     - A directed relationship between products, with its type and relationship metadata.

Use ``tolteca_datamodels.core.data_prod`` to define portable product records,
resolve artifact references and record executions in filesystem namespaces.
Specialize the product metadata for your application. These APIs do not load
instrument support or require a database.

Product UIDs, reference UIDs and association target UIDs must be nonempty strings.
For an empty tag, use ``current=None``. Tags remain editable:

.. doctest::

   >>> from tolteca_datamodels.core.data_prod import DataProdRef, DataProdTag
   >>> tag = DataProdTag(current=DataProdRef(uid="measurement:1"))
   >>> tag.current.uid
   'measurement:1'
   >>> tag.current = None
   >>> tag.current is None
   True

Define a product schema
-----------------------

Use ``tolteca_datamodels.core.data_prod`` when the product metadata and identity
rules differ from TolTECA observations. Specialize the record with your metadata
type, then specialize the index with that record:

.. doctest::

   >>> from datetime import date
   >>> from pathlib import Path
   >>> from tempfile import TemporaryDirectory
   >>> from tolteca_datamodels.core.types import FrozenDataModel
   >>> from tolteca_datamodels.core.data_prod import DataProdIndexBase, DataProdBase
   >>> class MeasurementMeta(FrozenDataModel):
   ...     quantity: str
   ...     unit: str
   ...     acquired: date
   ...     uncertainty: float | None
   >>> class MeasurementRecord(DataProdBase[MeasurementMeta]):
   ...     pass
   >>> class MeasurementIndex(DataProdIndexBase[MeasurementRecord]):
   ...     pass
   >>> index = MeasurementIndex(
   ...     product=MeasurementRecord(
   ...         uid="measurement:1",
   ...         meta=MeasurementMeta(
   ...             quantity="temperature", unit="K",
   ...             acquired=date(2026, 11, 6), uncertainty=None,
   ...         ),
   ...     ),
   ... )
   >>> with TemporaryDirectory() as directory:
   ...     path = index.write(Path(directory) / "index.yaml")
   ...     restored = MeasurementIndex.read(path)
   >>> restored.product.meta.unit
   'K'
   >>> restored.product.meta == index.product.meta
   True

The same ``read``, ``write`` and source-location methods are available.
Nested ``products`` also use ``MeasurementIndex``. Add validators to the record
for your product identity and metadata requirements.

Use application-defined relationship labels in ``DataProdAssoc.data_prod_assoc_type``.
Portable records require no database registration; a database consumer may require
its own registered relationship types.

Each item must name a declared location. List a given location/path pair only once
per product; other products may reference the same source. If you include a
namespace record, its UUID must match ``product.namespace_uid``. If you include a
store-run record, include its execution in ``run`` with the matching UID.

YAML persistence preserves explicit null values and restores metadata through its
declared field types. Strict validation remains in effect: an integer field does
not accept a quoted number. Computed fields are reconstructed from the loaded
metadata and are not saved as input fields. Custom field serializers must supply
a representation that their corresponding validators can read.

``DataProdStoreBase(path, namespace=...)`` provides namespace identity and run
records for a custom filesystem store. Use ``write_run`` to save resolved
configuration and the outcome supplied by the execution owner. The concrete
store supplies its product paths and publication operations.

Read product references
-----------------------

Read a saved document through your index subclass. Resolving an item does not
open its payload:

.. code-block:: python

   index = MeasurementIndex.read("index.yaml")
   for item in index.product.data_items:
       print(item.meta, index.get_source_uri(item.source))

Each item's ``source.path`` is a :class:`~pathlib.PurePosixPath`, validated by
Tollan's ``SubPath`` field type. It names an item under the location selected by
``source.location``. Supply literal filename characters, including spaces,
``%``, ``#`` and ``?``; do not URL-encode the path.

``read`` preserves these relative paths. ``get_source_uri`` joins the path to
its location and returns an encoded, absolute URI without opening the artifact.
Relative location roots resolve against the opened index directory. For an exact
saved realization, pass its ``index.yaml`` directly to ``MeasurementIndex.read``.

Declare a location with ``root``, for example ``{root: .}``, ``{root: ..}``,
or ``{root: 'archive:/data'}``. Use ``index.resolve_location(key)`` to obtain a
:class:`~tollan.utils.fileloc.FileLoc`; its ``url`` is the absolute URI and its
``path`` is the decoded path. Relative roots need the index directory and do not
use an application configuration root.

.. doctest::

   >>> from tolteca_datamodels.core.data_prod import DataProdSource, Location
   >>> source = DataProdSource(location="archive", path="scans/scan 50%#?.nc")
   >>> source.path.as_posix()
   'scans/scan 50%#?.nc'
   >>> location = Location(root="https://archive.example/data")
   >>> location.resolve().joinpath(source.path).url
   'https://archive.example/data/scans/scan%2050%25%23%3F.nc'

Portable YAML stores this reference as ``source: {location: archive, path: ...}``.

Moving an index and its relative artifacts together preserves their references.
Reopen the copied index to resolve its sources at the new location. Writing an
opened document preserves its references and leaves its existing resolution base
unchanged. YAML formatting and comments are not preserved. Reading uses safe
YAML tags; Python-specific object tags are rejected.

Preserve partial attribution
----------------------------

Use :class:`~tolteca_datamodels.core.data_prod.DataProdAudit` to retain creation and
latest modification times. Set ``created_by`` or ``updated_by`` to ``None`` when
that actor is unknown. Include each known actor in the index's ``actors`` mapping,
keyed by ``actor.uid``. A key that differs from the actor's UUID raises a validation
error in both product and namespace indexes.
For example, an imported historical record can have an unknown creator and a
known latest modifier:

.. doctest::

   >>> from datetime import datetime, timezone
   >>> from uuid import UUID
   >>> from tolteca_datamodels.core.data_prod import DataProdAudit
   >>> audit = DataProdAudit(
   ...     created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
   ...     updated_at=datetime(2026, 11, 9, tzinfo=timezone.utc),
   ...     updated_by=UUID("00000000-0000-0000-0000-000000000001"),
   ... )
   >>> audit.created_by is None, audit.updated_by is not None
   (True, True)

Saving and loading the index preserves the timestamps and the known modifier.
Both actors may be unknown; the audit still retains its timestamps. An omitted
``audit`` means no audit record was supplied.

Namespaces, stores and runs
---------------------------

A :class:`~tolteca_datamodels.core.data_prod.DataProdNamespace` identifies a persistent
output scope. A :class:`~tolteca_datamodels.core.data_prod.DataProdStoreBase` provides
filesystem access to one namespace. Each
:class:`~tolteca_datamodels.core.data_prod.WorkflowRun` records one execution independently of destination.
The store's ``namespace`` argument is a relative path; the stable namespace UUID
is stored in ``namespace.yaml``.

Use :meth:`~tolteca_datamodels.core.data_prod.WorkflowRun.create` to create an execution
record and track its outcome. Save it with
:meth:`~tolteca_datamodels.core.data_prod.DataProdStoreBase.write_run`.
Use :meth:`~tolteca_datamodels.core.data_prod.DataProdStoreBase.read_run` to inspect an
existing record. To reserve an automatic name for a run that may produce no
products, pass ``run_name=CounterRunNameGenerator(...)`` to ``write_run``.
Repeated writes of the same execution reuse its saved reservation.
A conflicting name or counter-generation policy raises ``ValueError``.
Omit ``run_name`` to save the execution without allocating a name.

Counter values can have gaps. If allocation is interrupted before the reservation
is saved, automatic naming advances on retry. A fixed name left claimed by the
interruption raises ``FileExistsError``. A saved reservation is reused, even when
the caller did not receive the successful return.

To release an abandoned fixed name, stop writers and confirm that no saved
reservation or published output uses that name. Then remove the empty
``<namespace>/.runs/by_name/<name>`` folder with ``rmdir``. Deleting a product
directory does not release its name. Leave completed reservations and counter
claims intact; do not reduce ``counter_next``.

Pass ``uid=None`` to generate an execution UUID, or supply a UUID object or string
to retain that identity. Invalid values raise a validation error. ``context`` and
``meta`` accept dictionaries with string keys; ``None`` uses an empty dictionary.
``WorkflowRun.create`` records the captured environment in ``meta["runtime_info"]``.
You can also save an imported ``WorkflowRun`` and reserve its name when it has
no captured runtime metadata.

Runs and new namespaces share a default actor within the process when the host,
user, platform, Python environment and timezone match. Commands and workflow
configuration belong to each run and do not change this actor. Namespace
creation reuses a matching actor already recorded in that namespace.

Pass ``actor=`` to ``WorkflowRun.create`` when reusing a particular recorded actor,
including one from another process. Run creation does not search stores. Saving
or importing a run retains its actor UUID and attribution; actors with matching
environment facts are not merged. For example, to reuse a namespace's creator:

.. doctest::

   >>> from tempfile import TemporaryDirectory
   >>> from tolteca_datamodels.core.data_prod import DataProdStoreBase, WorkflowRun
   >>> with TemporaryDirectory() as directory:
   ...     store = DataProdStoreBase(directory, namespace="science")
   ...     namespace = store.ensure_namespace()
   ...     creator = namespace.actors[namespace.namespace.audit.created_by]
   ...     run = WorkflowRun.create(actor=creator)
   ...     run.actor_uid == creator.uid
   True

The store manages a :class:`~tolteca_datamodels.core.data_prod.StoreRun` for each
execution's output in its namespace. ``store_run.name`` is the local run
name. Namespace exports carry these records in ``store_runs``, keyed by ``store_run.run_uid``. Include the
corresponding execution in ``runs``, keyed by ``run.uid``. Mismatched keys or a
missing execution raise a validation error. Reserved names and assigned counter
values must be unique within the namespace. An execution need not have a store-run
record. One store-run record can describe several product outputs sharing that name.

Namespace metadata and exports share one document type,
:class:`~tolteca_datamodels.core.data_prod.DataProdNamespaceIndex`.
``read_namespace()`` returns metadata and actors without reading run history.
``read_namespace_index(run_uids=[...])`` includes selected runs for transport or
registration. Run records remain in separate files.

Inspect run history
-------------------

.. code-block:: python

   store = DataProdStoreBase("dataprod_toltec", namespace="scratch")
   # Identity, actors and next counter
   descriptor = store.read_namespace()
   # One run's latest saved outcome
   run = store.read_run(run_uid)
   bundle = store.read_namespace_index(run_uids=[run_uid])

Omit ``run_uids`` only when you want to export the full run history. Run lifecycle
updates write the individual run record. Use ``read_run`` for its latest saved
outcome; an index's ``run`` is a snapshot.
Saving an earlier running snapshot keeps the completed record. Conflicting
execution or actor information raises an error without replacing the saved facts.
Publication outcomes are available in the exported ``store_runs`` records.

Path and run-name values
------------------------

Namespace paths use :data:`~tollan.config.types.path.SubPath` from Tollan.
They are relative to the product-store root.

Tag labels and counter keys use
:data:`~tollan.config.types.name.PathSegment` from Tollan. ``RunName`` adds
the product store's reserved-name checks.

.. autodata:: tolteca_datamodels.core.data_prod.run.CounterRunNameFormat

.. autodata:: tolteca_datamodels.core.data_prod.run.RunName

.. autodata:: tolteca_datamodels.core.data_prod.run.RunUid

Extension bases
---------------

.. autopydantic_model:: tolteca_datamodels.core.data_prod.DataProdBase
   :class-doc-from: class
   :model-show-json: False
   :inherited-members: BaseModel
   :members: reference

.. autopydantic_model:: tolteca_datamodels.core.data_prod.DataProdIndexBase
   :class-doc-from: class
   :model-show-json: False
   :inherited-members: BaseModel
   :members: read, write, resolve_location, get_source_uri

.. autoclass:: tolteca_datamodels.core.data_prod.DataProdStoreBase
   :members: ensure_namespace, read_namespace, read_run, read_namespace_index, write_namespace_index, namespace_path, write_run

Supporting records
------------------

.. automodule:: tolteca_datamodels.core.data_prod
   :members:
   :imported-members:
   :exclude-members: DataProdBase, DataProdIndexBase, DataProdStoreBase, CounterRunNameFormat, RunName, RunUid
