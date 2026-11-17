Namespaces, stores and runs
===========================

Use a store when you want a predictable output tree rather than an arbitrary
index filename. Use a run to record an execution and its outcome. A store and a
run are independent: the same execution can write several products or use
several destination stores.

Create a namespace store
------------------------

A :class:`~tolteca_datamodels.data_prod.DataProdNamespace` identifies a persistent
output scope. A :class:`~tolteca_datamodels.data_prod.DataProdStore` provides
filesystem access to one namespace. Each
:class:`~tolteca_datamodels.data_prod.WorkflowRun` records one execution independently of destination.
The store's ``namespace`` argument is a relative path; the stable namespace UUID
is stored in ``namespace.yaml``.

.. code-block:: python

   from tolteca_datamodels.data_prod import DataProdStore

   store = DataProdStore("dataprod_toltec", namespace="scratch")
   namespace_index = store.ensure_namespace()
   namespace_uid = namespace_index.namespace.uid

The directory is ``dataprod_toltec/scratch``. ``scratch`` is its relative path;
the saved UUID is its identity. Reopening that store reuses the namespace.

Record an execution
-------------------

Use :meth:`~tolteca_datamodels.data_prod.WorkflowRun.create` to create an execution
record and track its outcome. Save it with
:meth:`~tolteca_datamodels.data_prod.DataProdStore.write_run`.
Use :meth:`~tolteca_datamodels.data_prod.DataProdStore.read_run` to inspect an
existing record. To reserve an automatic name for a run that may produce no
products, pass ``run_name=CounterRunNameGenerator(...)`` to ``write_run``.
Subsequent materialization of the same execution reuses that name. See
:doc:`reduction-workdir` for a complete no-op example.

Save the execution again after its context exits, once its outputs are complete.
Saving a finished execution also settles an output that an interrupted
publication left pending: it is recorded as completed when its version was
committed, otherwise as failed.

Pass ``uid=None`` to generate an execution UUID, or supply a UUID object or string
to retain that identity. Invalid values raise a validation error. ``context`` and
``meta`` accept dictionaries with string keys; ``None`` uses an empty dictionary.
``WorkflowRun.create`` records the captured environment in ``meta["runtime_info"]``.
You can also save an imported ``WorkflowRun`` and reserve its name when it has
no captured runtime metadata.

Separate execution identity from output names
---------------------------------------------

The store manages a :class:`~tolteca_datamodels.data_prod.StoreRun` for each
execution's output in its namespace. ``index.store_run.name`` is the local run
name; ``index.base_path`` is the actual product directory. Namespace exports
carry these records in ``store_runs``, keyed by ``store_run.run_uid``. Include the
corresponding execution in ``runs``, keyed by ``run.uid``. Mismatched keys or a
missing execution raise a validation error. Reserved names and assigned counter
values must be unique within the namespace. An execution need not have a store-run
record. One store-run record can describe several product outputs sharing that name.

.. list-table:: Identity, naming and placement
   :header-rows: 1

   * - Object
     - What it identifies
     - Where you inspect it
   * - ``DataProdNamespace``
     - One output scope across moves or copies.
     - ``store.read_namespace().namespace``
   * - ``WorkflowRun``
     - One execution, independent of destination.
     - ``store.read_run(run_uid)``
   * - ``StoreRun``
     - That execution's local name and output outcomes in one namespace.
     - ``index.store_run`` or exported ``store_runs``.
   * - Product directory
     - The location of one saved realization.
     - ``index.base_path``

Choose the operation
--------------------

.. list-table:: Operations and their effects
   :header-rows: 1

   * - Task
     - API
     - Result
   * - Describe raw files
     - ``build_raw_obs_batch(files, locations=...)``
     - A ``DataProdBatch`` in memory; no persistence or run.
   * - Save or refresh an observation, acquisition, calibration group or named collection
     - ``store.write_index(index)``
     - A saved descriptor retaining its product identity, placed at the primary the index records. Loop over the raw batch's items to save several.
   * - Record an invocation
     - ``WorkflowRun.create(context=...)``
     - Run identity, inputs and outcome. A run can produce no products.
   * - Produce a new processing result
     - ``with store.materialize(index) as output_index:``, then ``write_artifacts(output_index, artifacts)``
     - A new immutable realization attributed to that run, with its artifacts and tag selection.
   * - Select a completed result
     - ``store.set_tag(realization, tag="main")``
     - The version group's tag selects that realization; the realization is unchanged.
   * - Register records in DPDB
     - ``tolteca_db.product_index.ingest_data_prod_batch(session, batch)``
     - Database records in the supplied transaction.

You may write raw indexes inside a run. The run records the invocation's outcome,
but those writes do not become publications or acquire that run as their producer.
A later exception marks the run failed without removing indexes already written.
Likewise, earlier completed materializations survive a later failure in the same run.
A run does not provide filesystem transaction rollback.

For the complete materialization lifecycle and artifact-writing example, follow
:doc:`data-product-publication`. Use ``DataProdIndex.write(path)`` to save an
index at an arbitrary path, without applying the store's layout.

Read saved executions and export selected runs
----------------------------------------------

Namespace metadata and exports share one document type,
:class:`~tolteca_datamodels.data_prod.DataProdNamespaceIndex`.
``read_namespace()`` returns metadata and actors without reading run history.
``read_namespace_index(run_uids=[...])`` includes selected runs for transport or
registration. Run records remain in separate files.

.. code-block:: python

   store = DataProdStore("dataprod_toltec", namespace="scratch")
   # Identity, actors and next counter
   descriptor = store.read_namespace()
   # One run's latest saved outcome
   run = store.read_run(run_uid)
   bundle = store.read_namespace_index(run_uids=[run_uid])

Omit ``run_uids`` only when you want to export the full run history. Run lifecycle
updates write the individual run record. A product index retains the run snapshot
at that product's publication; use ``read_run`` for the final invocation outcome.
Saving an earlier running snapshot keeps the completed record. Conflicting
execution or actor information raises an error without replacing the saved facts.
Publication outcomes are available in the exported ``store_runs`` records.

Reuse actor identity
--------------------

An actor records who or what created a record, with its stable runtime facts.
Runs reference that actor while retaining their own inputs, configuration and
outcome.

Runs and new namespaces share a default actor within the process when the host,
user, platform, Python environment and timezone match. Commands and workflow
configuration belong to each run and do not change this actor. Namespace creation
and tag promotion reuse a matching actor already recorded in that namespace.

Pass ``actor=`` to ``WorkflowRun.create`` when reusing a particular recorded actor,
including one from another process. Run creation does not search stores. Saving
or importing a run retains its actor UUID and attribution; actors with matching
environment facts are not merged. For example, to reuse a namespace's creator:

.. doctest::

   >>> from tempfile import TemporaryDirectory
   >>> from tolteca_datamodels.data_prod import DataProdStore, WorkflowRun
   >>> with TemporaryDirectory() as directory:
   ...     store = DataProdStore(directory, namespace="science")
   ...     namespace = store.ensure_namespace()
   ...     creator = namespace.actors[namespace.namespace.audit.created_by]
   ...     run = WorkflowRun.create(actor=creator)
   ...     run.actor_uid == creator.uid
   True

Handle interrupted name allocation
----------------------------------

Counter values can have gaps. If allocation is interrupted before the reservation
is saved, automatic naming advances on retry. A fixed name left claimed by the
interruption raises ``FileExistsError``. A saved reservation is reused, even when
the caller did not receive the successful return.

To release an abandoned fixed name, stop writers and confirm that no saved
reservation or published output uses that name. Then remove the empty
``<namespace>/.runs/by_name/<name>`` folder with ``rmdir``. Deleting a product
directory does not release its name. Leave completed reservations and counter
claims intact; do not reduce ``counter_next``.
