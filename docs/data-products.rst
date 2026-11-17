Data products
=============

Use data products to describe acquired data, save processing results and reuse
an exact result later. ``tolteca_datamodels.data_prod`` supplies the records and
filesystem store. A product index can travel with its artifacts and can be read
without a database. ``tolteca_db`` registers those records for shared queries.

The shared concepts
-------------------

A **product** is an identified acquisition, collection or result. An **item**
describes one of its data resources, such as an array, table, plot or report.
An **index** is the document that describes the product and gives the locations
needed to find its items. The artifact payloads remain in their own files.

For example, one processed acquisition can have an array store, a diagnostic
plot and a text report. Those are three items of one product. Repeating the
processing creates another product realization; a version group keeps the
realizations together and a tag such as ``main`` selects one.

.. list-table:: Concepts and their public classes
   :header-rows: 1
   :widths: 20 35 45

   * - Concept
     - API
     - Use
   * - Product, item and index
     - :class:`~tolteca_datamodels.data_prod.DataProd`,
       :class:`~tolteca_datamodels.data_prod.DataProdItem`,
       :class:`~tolteca_datamodels.data_prod.DataProdIndex`
     - Describe an acquisition, collection or result, its data resources and the document that locates them.
   * - Reference, association and tag
     - :class:`~tolteca_datamodels.data_prod.DataProdRef`,
       :class:`~tolteca_datamodels.data_prod.DataProdAssoc`,
       :class:`~tolteca_datamodels.data_prod.DataProdTag`
     - Identify an exact product, relate products and select a realization from a version group.
   * - Source and location
     - :class:`~tolteca_datamodels.data_prod.DataProdSource`,
       :class:`~tolteca_datamodels.data_prod.Location`
     - Resolve an item's relative path against a declared root to find its artifact.
   * - Namespace and store
     - :class:`~tolteca_datamodels.data_prod.DataProdNamespace`,
       :class:`~tolteca_datamodels.data_prod.DataProdStore`
     - Identify an output scope and read or write its filesystem representation.
   * - Execution and local run
     - :class:`~tolteca_datamodels.data_prod.WorkflowRun`,
       :class:`~tolteca_datamodels.data_prod.StoreRun`
     - Record an execution independently of destination, and its local name and output outcomes in one namespace.
   * - Batch and inventory
     - :class:`~tolteca_datamodels.data_prod.DataProdBatch`,
       :class:`~tolteca_datamodels.data_prod.DataProdInventoryIndex`
     - Package records for transfer and identify their reporting owner, without copying artifacts.
   * - Review
     - :class:`~tolteca_datamodels.data_prod.DataProdReview`
     - Record an assessment of an exact product by an actor in a given role.

A **version group** and a **named group** are products too: they use ``DataProd``
with :class:`~tolteca_datamodels.data_prod.metadata.VersionGroupMeta` or
:class:`~tolteca_datamodels.data_prod.metadata.NamedGroupMeta` metadata.
An **artifact** is the payload itself, such as a saved array or plot;
``DataProdItem`` describes how to find it.

These are related objects, not successive processing stages. A product's
identity, its producing execution and its current file locations answer different
questions. Moving a saved tree preserves its recorded identities; repeating the
processing creates a new realization.

How the document reaches the data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. graphviz::

   digraph product_overview {
     rankdir=TB;
     node [shape=box, fontname="Helvetica"];
     index [label="DataProdIndex\nIndex document"];
     product [label="DataProd\nIdentity + metadata"];
     item [label="DataProdItem\nMetadata + source"];
     payload [label="Artifacts\nArrays, tables, plots, reports"];
     {rank=same; index; product;}
     {rank=same; item; payload;}
     index -> product [label="describes"];
     product -> item [label="contains"];
     item -> payload [label="locates"];
   }

Follow a result through the system
----------------------------------

For an acquisition such as ``tcs-145609-0-1``:

1. Its raw-product record identifies the acquisition. Items describe the available
   files; an index records their locations.
2. A processing run produces a result whose primary relationship points back to
   that acquisition.
3. A store saves the result in a namespace. The saved index describes its exact
   realization and artifact paths.
4. A later run produces another realization in the same version group. The group's
   ``main`` tag can select the new one while the earlier result remains readable.
5. A batch can carry these records to DPDB for shared queries. Filesystem use does
   not require registration.

The :doc:`publication example <data-product-publication>` performs this complete
sequence through saved files, stopping at batch preparation. DPDB's guide covers
the subsequent registration.

Explore the layers
------------------

Start with the record model, then execution and storage. The worked examples
combine them into complete output trees. Each area groups related concepts;
the individual guides explain their classes and usage.

.. list-table:: From records to a reusable output tree
   :header-rows: 1
   :widths: 30 70

   * - Area
     - What you will learn
   * - :doc:`data-product-record-model`
     - Describe products, relationships and source locations; package their records and reviews for exchange.
   * - :doc:`data-product-execution-storage`
     - Choose a namespace, record an execution and save or reopen its outputs.
   * - :doc:`data-product-examples`
     - Publish a complete product tree, repeat a result, and organize outputs under a named group.

Choose an entry point
---------------------

.. list-table::
   :header-rows: 1

   * - You have
     - Start with
   * - A saved index or product directory
     - ``DataProdIndex.read``; :doc:`data-product-sources`.
   * - Metadata and artifact paths to describe
     - ``DataProd``, ``DataProdItem`` and ``DataProdIndex``; :doc:`data-product-records`.
   * - A new processing result to save
     - ``WorkflowRun`` and ``DataProdStore.materialize``; :doc:`data-product-publication`.
   * - Acquisition files to prepare for processing
     - ``build_raw_obs_batch`` and :doc:`raw-obs-context`.
   * - A configured KIDs workflow
     - Its output step writes products; :doc:`kids-reduction` and :doc:`workflow-artifacts`.
   * - Records to register or query in DPDB
     - `DPDB publication and registration <https://toltec.lmtgtm.org/dr/science/docs/tolteca_db/publish-product-example.html>`_.

Continue with :doc:`named-group-products` when a project or selected collection
owns the result. Use :doc:`data-products-api` to look up signatures and fields.
For developers adding another schema, see :doc:`data-product-extensions`.

.. toctree::
   :hidden:
   :maxdepth: 2

   data-product-record-model
   data-product-execution-storage
   data-product-examples
   data-products-api
   data-product-extensions
