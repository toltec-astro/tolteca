Record model
============

Use the record model to describe data and its relationships without opening the
scientific payloads or registering them in a database. These records answer
different questions about the same product: what it represents, how it relates
to other products, where its data resides and how its description is exchanged.

Start with products, items and indexes to construct a description. Then choose
the other guides according to what you need to add or inspect:

.. list-table::
   :header-rows: 1
   :widths: 43 57

   * - Guide
     - What you can describe
   * - :doc:`data-product-records`
     - Product identity and metadata, constituent data items, and the index document that packages them.
   * - :doc:`data-product-relationships`
     - Connections between products, primary placement, version groups and an exact version selected by a tag.
   * - :doc:`data-product-sources`
     - Item paths, location roots and how an index resolves them after it is opened or moved.
   * - :doc:`data-product-transfer`
     - Collections of records for transfer, their reporting owner and assessments of exact products.

To save processing results and record the execution that produced them, continue
with :doc:`data-product-execution-storage`. For these records used together in
a complete output tree, see :doc:`data-product-examples`.

.. toctree::
   :hidden:
   :maxdepth: 1

   data-product-records
   data-product-relationships
   data-product-sources
   data-product-transfer
