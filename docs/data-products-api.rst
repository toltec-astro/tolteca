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
