Publish and reopen a product tree
=================================

This example brings records, relationships, source locations and runs together.
It writes two acquisition descriptors, publishes a result for each in one run,
then repeats the first acquisition in a second run. The output is a complete
``dataprod_toltec/scratch`` tree that you can inspect without DPDB.

The acquisition identifiers follow the reference-data convention. The artifacts
are demonstration text reports, not numerical reductions. No acquisition files,
database or remote service are needed.

Run the complete example
------------------------

Use an environment with this TolTECA checkout and its dependencies installed.
From the repository root, choose a new output directory:

.. code-block:: bash

   python examples/data_product_store.py /tmp/tolteca-product-example

The script refuses an existing output directory and checks the saved identities,
version selections, report content and final run outcomes. Download the
:download:`complete script <../examples/data_product_store.py>` to run it elsewhere.
The sections below show the same operations in order.

Describe the acquisitions
-------------------------

A raw acquisition is a logical product. Describe its required quartet and give
it a canonical UID; saving the descriptor does not copy any acquired files.

.. literalinclude:: ../examples/data_product_store.py
   :language: python
   :pyobject: describe_acquisition

Create one namespace store and write both descriptions:

.. literalinclude:: ../examples/data_product_store.py
   :language: python
   :start-after: # setup-start
   :end-before: # setup-end
   :dedent: 4

``root`` is the new directory passed to the script. The descriptors appear at
``scratch/145609-0-1/dp_raw_obs.yaml`` and ``scratch/145609-0-2/dp_raw_obs.yaml``.
Use :doc:`raw-obs-context` when these descriptions should come from actual files.

Prepare a result and its artifact writer
----------------------------------------

A result draft names the logical result, its acquisition primary and its
producing execution. The store will assign the realized product's identity and
destination. Keep the logical result UID the same when repeating that result.
The primary association must use a registered association type, and its target
must have the product type that type references: ``dpa_raw_obs_proc_raw_obs``
references a ``dp_raw_obs``. Otherwise ``materialize`` raises ``ValueError``.

.. literalinclude:: ../examples/data_product_store.py
   :language: python
   :pyobject: describe_result

An artifact specification writes its retained object beneath a folder and returns
the files it wrote. This example writes a one-line report:

.. literalinclude:: ../examples/data_product_store.py
   :language: python
   :pyobject: TextSummary

Inside the publication scope, ``write_artifacts(output_index, artifacts,
producer=...)`` has each specification write beneath ``output_index.base_path``
and adds every written file to the realization as a data item. ``producer`` names
the writer, such as a step's or specification's qualified key
(``TextSummary.get_context_key()``).
Use ``meta`` for selection keys or producer-specific ``context``. The writer
supplies the item kind, file format and name, producer, and run identity; these
fields cannot be overridden through ``meta``.

Publish several products in one run
-----------------------------------

.. literalinclude:: ../examples/data_product_store.py
   :language: python
   :pyobject: publish_results

The execution context closes before the ``saved_runs`` callback saves its final
outcome. Both products share that execution UUID and the namespace's local run
name. Each materialization completes independently.

``materialize`` accepts one derived-product index containing its producing
``run`` and primary relation. It yields a destination-bound copy. Write artifacts
into that copy with ``write_artifacts``.

Normal exit saves the index and installs the output; an exception discards
temporary files and preserves the previous tag. The original index remains
unchanged.

Logical observations, acquisitions, calibration groups and named collections are
saved with ``write_index``, and a version group forms from its materialized
members; materialization rejects all of them. See
:doc:`named-group-products` for collection and result-version examples.
No ``write_batch`` operation is currently provided.

A normal exit can publish a metadata-only result. The producing workflow must
raise or otherwise stop publication when required scientific results are absent.
Keep lazy input resources open until artifact writing finishes. Previously
completed products survive a later failure; a run does not roll back the tree.

Repeat one result
-----------------

.. literalinclude:: ../examples/data_product_store.py
   :language: python
   :start-after: # publish-start
   :end-before: # publish-end
   :dedent: 4

.. list-table:: Outputs after both runs
   :header-rows: 1

   * - Acquisition
     - First run
     - Second run
     - Selected by ``main``
   * - ``tcs-145609-0-1``
     - ``0000-example``
     - ``0001-recheck``
     - ``0001-recheck``
   * - ``tcs-145609-0-2``
     - ``0000-example``
     - No new result
     - ``0000-example``

The names belong to the namespace's run allocation. Each product's version group
owns its own ``main`` tag, so repeating the first result does not change the second.

.. graphviz::

   digraph publication_flow {
     rankdir=LR;
     node [shape=box, fontname="Helvetica"];
     run [label="One execution"];
     a [label="Acquisition 1 result"];
     b [label="Acquisition 2 result"];
     ga [label="Group 1: versions + main"];
     gb [label="Group 2: versions + main"];
     run -> a;
     run -> b;
     a -> ga [label="materialize"];
     b -> gb [label="materialize"];
   }

Inspect the generated tree
--------------------------

This tree was generated by the script. Execution UUIDs are replaced by labels;
locks, counters, reservations and membership bookkeeping are omitted.

.. literalinclude:: _static/data-products/tree.txt
   :language: text

.. list-table:: What each saved document describes
   :header-rows: 1

   * - File
     - Contents
   * - ``namespace.yaml``
     - Namespace identity, actors and allocation state.
   * - ``.runs/by_uid/<uuid>/run.yaml``
     - The execution's saved configuration and outcome.
   * - ``.runs/by_uid/<uuid>/store_run.yaml``
     - Its local name and output outcomes in this namespace.
   * - ``dp_raw_obs.yaml``
     - A logical acquisition, with no producing-run suffix.
   * - ``dp_raw_obs_proc/index.yaml``
     - The version group and its current selections.
   * - ``<run-name>/index.yaml``
     - One exact realization, its metadata, sources and run snapshot.

Reopen the selected result and an earlier version
-------------------------------------------------

.. literalinclude:: ../examples/data_product_store.py
   :language: python
   :start-after: # read-start
   :end-before: # read-end
   :dedent: 4

Only ``read_text`` opens the report payload. Index reads inspect saved records.
Reading the group returns both committed realizations; ``run_uids`` selects the
earlier one without changing the group's tag.

Package the saved records
-------------------------

The returned batch includes both acquisitions, both version groups with their
members, and the namespace with the two final run records:

.. literalinclude:: ../examples/data_product_store.py
   :language: python
   :start-after: # batch-start
   :end-before: # batch-end
   :dedent: 4

This batch is usable in memory with the opened indexes' location context.
Follow :doc:`data-product-transfer` before serializing it independently of the
index files. Product indexes retain the run snapshot at their publication;
``store.read_run`` supplies the final saved execution outcome.

Continue into a larger workflow
-------------------------------

.. list-table::
   :header-rows: 1

   * - Task
     - Continue with
   * - Register these records and query them in a database
     - `DPDB publication and registration <https://toltec.lmtgtm.org/dr/science/docs/tolteca_db/publish-product-example.html>`_
   * - Save results whose primary is a project or selected collection
     - :doc:`named-group-products`
   * - Produce real KIDs results and declared artifacts
     - :doc:`kids-reduction` and :doc:`workflow-artifacts`
   * - Configure a reduction that saves products
     - :doc:`reduction-workdir`
