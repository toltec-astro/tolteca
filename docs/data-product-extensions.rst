Product schema extensions (for developers)
==========================================

Use these extension bases when adding a domain whose product metadata or
identity rules differ from TolTECA's. Application callers using the existing
TolTECA product types can stay with :doc:`data-product-records`.

Specialize the record and its document
--------------------------------------

Import the extension bases from ``tolteca_datamodels.core.data_prod``. Specialize
the record with your metadata type, then specialize the index with that record:

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

Preserve typed metadata
-----------------------

YAML persistence preserves explicit null values and restores metadata through its
declared field types. Strict validation remains in effect: an integer field does
not accept a quoted number. Computed fields are reconstructed from the loaded
metadata and are not saved as input fields. Custom field serializers must supply
a representation that their corresponding validators can read.

Add a filesystem store
----------------------

``DataProdStoreBase(path, namespace=...)`` provides namespace identity and run
records for a custom filesystem store. Use ``write_run`` to save resolved
configuration and the outcome supplied by the execution owner. The concrete
store supplies its product paths and publication operations.

See :doc:`data-products-api` for the extension-base signatures and fields.
