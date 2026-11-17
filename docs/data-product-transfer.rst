Batches, inventories and reviews
================================

A batch packages already prepared records for transport or database registration.
It can carry product indexes, namespace/run records and review history without
copying artifact payloads. Read :doc:`data-product-records` and
:doc:`data-product-runs` first for the objects being packaged.

:class:`~tolteca_datamodels.data_prod.DataProdBatch` is the transfer container.
:class:`~tolteca_datamodels.data_prod.DataProdInventoryIndex` identifies who
reports the records it wraps. A :class:`~tolteca_datamodels.data_prod.DataProdReview`
records an assessment of an exact product.

Build a batch
-------------

Use :class:`~tolteca_datamodels.data_prod.DataProdBatch` to submit prepared indexes
together. ``items`` accepts product and namespace indexes, or
``DataProdInventoryIndex`` records that identify the inventory reporting their
contents. Namespace indexes can carry selected runs. ``meta`` describes the
submission; it does not replace metadata on a product or configuration on a run.
For example, package a raw-acquisition description and restore it from JSON:

.. doctest::

   >>> from tolteca_datamodels.data_prod import (
   ...     DataProd, DataProdIndex, DataProdBatch,
   ... )
   >>> from tolteca_datamodels.data_prod.metadata import RawObsMeta
   >>> raw_index = DataProdIndex(product=DataProd(
   ...     uid="tcs-145609-0-1",
   ...     meta=RawObsMeta(
   ...         name="Tune acquisition", data_prod_type="dp_raw_obs",
   ...         master="tcs", obsnum=145609, subobsnum=0, scannum=1,
   ...     ),
   ... ))
   >>> batch = DataProdBatch(
   ...     meta={"purpose": "archive handoff"}, items=[raw_index],
   ... )
   >>> restored = DataProdBatch.model_validate_json(batch.model_dump_json())
   >>> restored.items == batch.items, restored.meta == batch.meta
   (True, True)

For raw files, :doc:`raw-obs-context` shows how ``build_raw_obs_batch`` prepares
these records from acquired data. To register a prepared batch, pass it to
``tolteca_db.product_index.ingest_data_prod_batch(session, batch)``.
Constructing a batch does not inspect files or register anything. The destination
checks dependencies; related records may be in the batch or already registered.

``reviews`` and ``actors`` also permit assessment-only submissions against
existing products, with empty ``items``. Actor mapping keys must match their
records' UUIDs. Include predecessor assessments when the destination does not
already have them. An entirely empty batch is valid.

Restore a serialized batch
~~~~~~~~~~~~~~~~~~~~~~~~~~

Use JSON validation when restoring saved JSON, including decoded JSON mappings:

.. doctest::

   >>> from pydantic_core import to_json
   >>> snapshot = batch.model_dump(mode="json")
   >>> DataProdBatch.model_validate_json(to_json(snapshot)) == restored
   True

Use ``model_validate`` for Python values with their declared types. JSON arrays,
for example, restore tuple fields through JSON validation without weakening the
record's strict Python validation.

Retain resolvable source locations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each index retains its own source locations. Wrapping an opened index preserves
its relative-location resolution in memory. For a batch transported independently
of the index files, prepare absolute location roots: JSON does not retain the
directory from which an index was opened.

Preserve reporting ownership
----------------------------

Wrap contributed product and namespace indexes in
:class:`~tolteca_datamodels.data_prod.DataProdInventoryIndex` when transferring
records with an explicit reporting owner. Create an inventory identity once and
reuse it for later contributions. Using ``raw_index`` from the example above:

.. doctest::

   >>> from uuid import uuid4
   >>> from tolteca_datamodels.data_prod import (
   ...     DataProdInventory, DataProdInventoryIndex,
   ... )
   >>> inventory = DataProdInventory(uid=uuid4(), name="archive")
   >>> contribution = DataProdInventoryIndex(
   ...     inventory=inventory, entries=[raw_index],
   ... )
   >>> transfer = DataProdBatch(items=[contribution])
   >>> restored_transfer = DataProdBatch.model_validate_json(
   ...     transfer.model_dump_json(),
   ... )
   >>> restored_transfer.items[0].inventory.uid == inventory.uid
   True

The inventory identifies whose records are reported. It has no storage root:
each contained index retains its own locations and product identities. An empty
``entries`` list is valid. Registration decides how to combine these contributions
with records already at the destination.

Inspect portable assessments
----------------------------

A :class:`~tolteca_datamodels.data_prod.DataProdReview` describes one actor's
complete findings for an exact product, role and QA namespace. Reuse the actor
identity when revising an assessment, and name the preceding review explicitly:

.. doctest::

   >>> from datetime import UTC, datetime
   >>> from tolteca_datamodels.data_prod import (
   ...     Actor, DataProdReview, current_reviews,
   ... )
   >>> reviewer = Actor(
   ...     uid=uuid4(), name="Example curator", created_at=datetime.now(UTC),
   ... )
   >>> first = DataProdReview(
   ...     product=raw_index.product.reference, actor_uid=reviewer.uid,
   ...     role="curator", namespace="quality",
   ...     flags={"needs_followup": True}, note="Inspection is pending.",
   ... )
   >>> revised = DataProdReview(
   ...     product=first.product, actor_uid=reviewer.uid,
   ...     role=first.role, namespace=first.namespace, previous_uid=first.uid,
   ...     flags={"needs_followup": False}, note="Inspection is complete.",
   ... )
   >>> history = [first, revised]
   >>> [current] = current_reviews(history)
   >>> current.flags
   {'needs_followup': False}
   >>> review_batch = DataProdBatch(
   ...     reviews=history, actors={reviewer.uid: reviewer},
   ... )
   >>> review_batch.items
   []

Each review supplies a complete flag mapping. Omitted flags are no longer findings
from that assessment; an empty mapping still records a review. The reviewing
workflow defines what flags mean. No overall acceptance decision is inferred.

``current_reviews`` expects consistent history and selects assessments with no
successor in the supplied records. Timestamps only order the display. Include
complete history when determining current state: the function cannot know about
an omitted revision. Other actors, roles, QA namespaces and exact products retain
their independent assessments.

The record validates its fields and rejects a self predecessor. DPDB registration
checks referenced records and history consistency, including conflicting
successors. See the `DB review guide
<https://toltec.lmtgtm.org/dr/science/docs/tolteca_db/quality-review.html>`_
for registration, conflict handling and CLI operations.

Preserve partial attribution
----------------------------

Use :class:`~tolteca_datamodels.data_prod.DataProdAudit` to retain creation and
latest modification times. Set ``created_by`` or ``updated_by`` to ``None`` when
that actor is unknown. Include each known actor in the index's ``actors`` mapping,
keyed by ``actor.uid``. A key that differs from the actor's UUID raises a validation
error in both product and namespace indexes.
For example, an imported historical record can have an unknown creator and a
known latest modifier:

.. doctest::

   >>> from datetime import datetime, timezone
   >>> from uuid import UUID
   >>> from tolteca_datamodels.data_prod import DataProdAudit
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

Continue to the `DPDB publication example
<https://toltec.lmtgtm.org/dr/science/docs/tolteca_db/publish-product-example.html>`_
for database transactions, reporting inventories and journals. Those operations
consume the portable records; constructing or serializing a batch does not
perform them.
