Data-product API
================

This reference lists the portable record models, filesystem operations and
extension bases. For an introduction and the order in which to learn them, start
with :doc:`data-products`. The guides provide runnable examples; this page
provides signatures, field constraints and method details.

.. list-table:: API by task
   :header-rows: 1

   * - Task
     - Entry points
     - Guide
   * - Describe a product
     - ``DataProd``, ``DataProdItem``, ``DataProdIndex``
     - :doc:`data-product-records`
   * - Link products and versions
     - ``DataProdRef``, ``DataProdAssoc``, ``DataProdTag``
     - :doc:`data-product-relationships`
   * - Locate artifacts
     - ``DataProdSource``, ``Location``, ``DataProdIndex.read``
     - :doc:`data-product-sources`
   * - Save an execution and its output
     - ``WorkflowRun``, ``StoreRun``, ``DataProdStore``
     - :doc:`data-product-runs`, :doc:`data-product-publication`
   * - Transport records or assessments
     - ``DataProdBatch``, ``DataProdInventoryIndex``, ``DataProdReview``
     - :doc:`data-product-transfer`
   * - Add another schema
     - ``DataProdBase``, ``DataProdIndexBase``, ``DataProdStoreBase``
     - :doc:`data-product-extensions`

TolTECA records and store
-------------------------

.. autopydantic_model:: tolteca_datamodels.data_prod.DataProd
   :class-doc-from: class
   :inherited-members: BaseModel
   :members: reference, get_primary
   :model-show-json: False

.. autopydantic_model:: tolteca_datamodels.data_prod.DataProdIndex
   :class-doc-from: class
   :inherited-members: BaseModel
   :members: read, write, resolve_location, get_source_uri
   :model-show-json: False

.. autoclass:: tolteca_datamodels.data_prod.DataProdStore
   :members:
   :inherited-members:

.. automodule:: tolteca_datamodels.data_prod
   :members:
   :imported-members:
   :exclude-members: DataProdIndex, DataProd, DataProdStore, artifact_table_from_indexes,
      product_table_from_indexes

Path and run-name values
------------------------

Namespace paths use :data:`~tollan.config.types.path.SubPath` from Tollan.
They are relative to the product-store root.

Tag labels, counter keys and primary-directory labels use
:data:`~tollan.config.types.name.PathSegment` from Tollan. ``RunName`` adds
the product store's reserved-name checks.

.. autodata:: tolteca_datamodels.data_prod.CounterRunNameFormat

.. autodata:: tolteca_datamodels.data_prod.RunName

.. autodata:: tolteca_datamodels.data_prod.RunUid

Product metadata
----------------

.. automodule:: tolteca_datamodels.data_prod.metadata
   :members:
   :exclude-members: _retort

Product and artifact model tables
---------------------------------

A product table selects whole products, including groups and products without
artifacts; DPDB quality review selects its subjects from it. An artifact table
selects the files within products. Both take the same model-table expressions,
built from portable indexes or from DPDB. See :doc:`reduction-workdir` for
selecting calibration artifacts.

.. autofunction:: tolteca_datamodels.data_prod.artifact_table_from_indexes

.. autofunction:: tolteca_db.model_table.artifact_table_from_dpdb

.. autofunction:: tolteca_datamodels.data_prod.product_table_from_indexes

.. autofunction:: tolteca_db.model_table.product_table_from_dpdb

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
