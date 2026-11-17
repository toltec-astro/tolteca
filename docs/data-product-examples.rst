Worked examples
===============

These examples combine the :doc:`data-product-record-model` with
:doc:`data-product-execution-storage`. Start with the product-tree example to
follow a result from its initial description to saved artifacts and later reuse.
Then use the named-group example when a project or selected collection owns
the result instead of a single observation.

.. list-table::
   :header-rows: 1
   :widths: 43 57

   * - Example
     - What you will produce and inspect
   * - :doc:`data-product-publication`
     - A complete local tree with two acquisition descriptions, results from two executions, version selections and a batch ready for registration. The runnable script needs no acquisition files or database.
   * - :doc:`named-group-products`
     - A named collection with explicit observation membership, published science results and independent version selections. This example extends the first workflow to a collection primary.

The guides link to the relevant record and storage concepts as they use them.
For publication by a configured reduction workflow, continue to
:doc:`workflow-artifacts`.

.. toctree::
   :hidden:
   :maxdepth: 1

   data-product-publication
   named-group-products
