Products, items and indexes
===========================

A :class:`~tolteca_datamodels.data_prod.DataProd` describes what the data
represents. Its :class:`~tolteca_datamodels.data_prod.DataProdItem` records describe
the constituent data resources. A :class:`~tolteca_datamodels.data_prod.DataProdIndex`
packages the product with the locations and supporting records needed to
interpret it. This page constructs a raw-acquisition description; it does not
open acquisition files.

Product, item and index
-----------------------

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

Construct typed metadata
------------------------

Choose the metadata class for the product type you are describing. These classes
also supply metadata to the database representation. Import them from ``tolteca_datamodels.data_prod.metadata``.

Product metadata validates typed construction, dictionary input and field
assignment. Unknown fields and incompatible values raise
``pydantic.ValidationError``. A raw product requires all four acquisition identity
components; missing subobservation and scan numbers do not default to zero.
Partial interface records, such as ``RoachInterfaceMeta``, can still describe
the available evidence without a complete product identity.

For producers constructing a raw product from known acquisition facts:

.. doctest::

   >>> from tolteca_datamodels.data_prod import DataProd
   >>> from tolteca_datamodels.data_prod.metadata import RawObsMeta
   >>> meta = RawObsMeta(
   ...     name="tcs-145609-0-1", data_prod_type="dp_raw_obs",
   ...     master="tcs", obsnum=145609, subobsnum=0, scannum=1,
   ... )
   >>> product = DataProd(uid="tcs-145609-0-1", meta=meta)
   >>> product.meta.source_name = "M31"
   >>> restored = DataProd.model_validate_json(product.model_dump_json())
   >>> restored.meta == product.meta
   True

Use ``model_validate`` for dictionaries containing Python values and
``model_validate_json`` for serialized JSON, including encoded datetimes.
To prepare products directly from acquisition files, use the workflow in
:doc:`raw-obs-context`.

Describe a data item
--------------------

Add an item for each constituent you want a consumer to find. Its metadata
describes the constituent; its source names a path under a declared location.
Using ``product`` from above:

.. doctest::

   >>> from tolteca_datamodels.data_prod import (
   ...     DataProdIndex, DataProdItem, DataProdSource, Location,
   ... )
   >>> product.data_items.append(DataProdItem(
   ...     meta={"interface": "toltec7", "format": "netcdf"},
   ...     source=DataProdSource.model_validate({
   ...         "location": "archive",
   ...         "path": "toltec7_145609_000_0001.nc",
   ...     }),
   ... ))
   >>> index = DataProdIndex(
   ...     product=product,
   ...     locations={"archive": Location(root="https://archive.example/data")},
   ... )
   >>> index.get_source_uri(index.product.data_items[0].source)
   'https://archive.example/data/toltec7_145609_000_0001.nc'

The illustrative filename need not exist to construct the record. Each item must
name a declared location, and a location/path pair can occur only once within a
product. Other products may reference the same resource. Resolving or opening it
is a separate task, covered in :doc:`data-product-sources`.

Inspect the document
--------------------

.. list-table:: Where to find each value
   :header-rows: 1

   * - Object
     - Contents
   * - ``index.product``
     - Identity, typed metadata, items, associations and tags.
   * - ``index.locations``
     - Roots used to resolve the items' source paths.
   * - ``index.products``
     - Supporting or member product documents included for exchange.
   * - Supporting records on ``index``
     - Namespace, actors and producing run, when supplied.

.. doctest::

   >>> index.product.reference == product.reference
   True
   >>> index.product.meta.source_name
   'M31'

Save this document with ``index.write(path)``. The source-location guide shows
how to reopen it and resolve its artifacts; the runs guide introduces the
supporting execution records.

Choose a product identity
-------------------------

Observation and raw-acquisition UIDs include an explicit master. Use
``make_obs_product_uid`` or ``make_raw_obs_product_uid`` to construct one from
known components, or ``normalize_primary_uid`` for an existing identifier.
Master labels start with an ASCII letter and contain only ASCII letters and
digits. Canonical UIDs use lowercase masters and unpadded numbers; normalization
does not rewrite the master recorded in metadata.

.. doctest::

   >>> from tolteca_datamodels.data_prod import (
   ...     make_raw_obs_product_uid, normalize_primary_uid,
   ... )
   >>> make_raw_obs_product_uid("TCS", 145609, 0, 1)
   'tcs-145609-0-1'
   >>> normalize_primary_uid("TCS-00145609-000-0001")
   'tcs-145609-0-1'

Observations, raw acquisitions, named collections, intrinsic calibration groups
and version groups have global identity: their ``product.namespace_uid`` must
be ``None``. You can place their indexes under several namespace directories
without changing that identity. A derived-product draft can omit its namespace;
materialization assigns it, and database registration requires it. A generic
``DataProdRef`` retains the supplied UID and namespace without inferring a
product type from the UID spelling.

Product UIDs, reference UIDs and association target UIDs must be nonempty strings.

.. list-table:: Identity and display values
   :header-rows: 1

   * - Value
     - Meaning
   * - ``product.uid``
     - Stable product identifier, such as ``tcs-145609-0-1``.
   * - ``product.namespace_uid``
     - Output identity scope for a realized result; absent for global logical products.
   * - ``product.reference``
     - The complete UID and namespace UUID used by relationships and lookups.
   * - ``product.meta.name``
     - A display name; changing it does not change identity.

Continue with :doc:`data-product-relationships` to connect this record to other
products and distinguish collection membership, primary placement and versions.
