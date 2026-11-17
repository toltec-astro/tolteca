Publish results under a named group
===================================

Use a named collection when no single observation owns the result: a project,
a source observed over several nights, or a selected subset for a coadd.
Keep the collection's identity when its membership changes. Each saved result
records the observations actually used, so an earlier result remains interpretable
after the collection is refreshed.

Represent the collection as a :class:`~tolteca_datamodels.data_prod.DataProd`
with :class:`~tolteca_datamodels.data_prod.metadata.NamedGroupMeta` metadata
(``data_prod_type="dp_named_group"``). Its associations record membership;
other products can name this group as their primary.

For the basic product/store lifecycle, first follow :doc:`data-product-publication`.
This example adds collection membership and independently selected results.

Choose the product and its primary
----------------------------------

.. list-table:: Products and their contents
   :header-rows: 1

   * - Product
     - Contents
     - Primary
   * - ``dp_science``
     - Science maps, optional catalogs, and diagnostics
     - One ``dp_obs`` or a ``dp_named_group``
   * - ``dp_pointing``
     - Pointing solution, supporting maps and diagnostics
     - One observation or a named collection
   * - ``dp_beammap``
     - Detector beam maps, fitted properties and diagnostics
     - One observation or a named collection
   * - ``dp_named_group``
     - Collection metadata and explicitly related products
     - Itself
   * - ``dp_version_group``
     - Realizations of one logical result and selections such as ``main``
     - The same primary as its realizations

Create a collection UUID once with ``str(uuid4())`` and retain it on refresh and
import. Its display name and readable directory are separate, following the same
identity/placement convention as a namespace. Equal names do not merge collections.

.. list-table:: Identity and placement
   :header-rows: 1

   * - Value
     - Example
     - Use
   * - ``group.product.uid``
     - ``b7f65f98-4f32-4e0a-a66a-cd278670f7cc``
     - Stable identity in associations and result UIDs
   * - ``group.product.meta.name``
     - ``Orion``
     - Display label; can change without moving files
   * - ``primary_path`` on the first ``store.write_index``
     - ``orion``
     - Directory in this namespace; reused by subsequent output steps

Save the descriptor before materializing results:

.. code-block:: python

   store.write_index(group, primary_path="orion")

The store remembers the path in ``namespace.yaml``. Another namespace or reporting
inventory can place the same UUID under a different directory. Paths must be unique
within that namespace projection and cannot use reserved observation/acquisition
labels. Changing a display name leaves the current directory intact. Changing the
recorded path is a projection operation; ``write_index`` does not move an existing
tree. Optional ``NamedGroupMeta.context`` can record a project ID, source name,
or the rule/import source used to assemble the collection.

An observation can belong to several collections and keeps its own directory.
Membership may also include calibration products or another named collection.
Membership alone never selects the inputs of a result or assigns directories to
other collections.

.. graphviz::

   digraph named_results {
     rankdir=LR;
     node [shape=box, fontname="Helvetica"];
     group [label="Named collection\nStable UUID; display name: Orion"];
     obs [label="Observation products\n149101, 149102, 149103"];
     versions [label="Science version group\n<group UUID>_science"];
     first [label="0000-coadd\nExact contributors: 149101, 149102"];
     second [label="0001-coadd\nExact contributors: all three"];
     group -> obs [label="current membership"];
     versions -> group [label="primary"];
     versions -> first [label="main"];
     versions -> second [label="another version"];
     first -> group [label="primary"];
     second -> group [label="primary"];
     first -> obs [label="selected contributors"];
     second -> obs [label="selected contributors"];
   }

Run the persistence example
---------------------------

From the TolTECA repository, run:

.. code-block:: bash

   python examples/named_group_products.py /path/to/new-example

This runnable example writes observation descriptors, a collection and three
result realizations. Its artifacts are JSON selection receipts; it does not
perform a numerical coadd. For a real multi-observation reduction, follow
:doc:`reduction-coadd`. To create or refresh a collection from a TolProj project,
follow :doc:`project-workspace`.

The example uses the same :meth:`~tolteca_datamodels.data_prod.DataProdStore.materialize`
operation as workflow output. Its producer supplies the collection index as a
supporting record, chooses a logical result UID and lists exact observation
associations. The store determines the destination and writes the realization.

.. literalinclude:: ../examples/named_group_products.py
   :language: python
   :pyobject: publish

``<group UUID>_science`` and ``<group UUID>_subset`` are independent logical
results. They have separate version groups and ``main`` selections, even though
one execution writes both under the shared store-run name ``0000-coadd``.
The next execution produces ``0001-coadd`` for the full result. The example
then explicitly selects its earlier version; the subset's selection stays intact.

Inspect the generated tree
--------------------------

The following tree was generated by the example. Hidden run records are included;
lock files, reservation directories and empty staging directories are omitted.

.. literalinclude:: _static/named-group-products/tree.txt
   :language: text

The collection descriptor is ``orion/dp_named_group.yaml``. The scientific
result's ``index.yaml`` above the numbered directories describes its version
group. Each numbered directory contains an immutable realization index and its
own ``selection.json``. ``science/`` and ``subset/`` are readable result directories;
their logical UIDs still contain the group UUID. The observation descriptors
remain at ``149101/``, ``149102/`` and ``149103/``.

The generated namespace descriptor records the placement separately:

.. literalinclude:: _static/named-group-products/primary-paths.yaml
   :language: yaml

Include the namespace descriptor along with product indexes when registering a
batch in DPDB. The same named-group UUID must be in that batch or already registered.
DPDB projection uses the selected inventory's recorded namespace and primary paths.
The direct reduction output step includes ``namespace.yaml`` in its registration.

Refresh a collection or select a result
---------------------------------------

.. list-table:: Operations
   :header-rows: 1

   * - Task
     - Operation
   * - Save a new collection descriptor
     - ``store.write_index(group, primary_path="orion")``
   * - Replace its filesystem descriptor
     - ``store.write_index(group, overwrite=True)``; reuses its path
   * - Register portable records in DPDB
     - ``ingest_data_prod_batch(session, batch)``; additive ingestion
   * - Explicitly replace DB collection metadata and membership
     - ``tolteca_db.named_groups.update_named_group(session, group)``;
       supply the complete desired membership and commit the transaction
   * - Select a saved full-result version
     - ``store.set_tag(realization)``, with the realization read from ``<group path>/science/0000-coadd``

A collection update changes its current membership. It does not rewrite saved
results, their exact observation associations, saved calibration contexts, or
version tags. DPDB's explicit update affects the shared collection even when
several inventories report it; ordinary ingestion never deletes omitted members.

For the reduction export step, choose ``product_type`` in ``reduce.export``;
it defaults to ``dp_science``. Set ``product_uid`` when you need independently
selectable results of the same type. A single-observation export otherwise uses
``<observation UID>_<product kind>``, such as ``tcs-149101_science``.

All three result types use ``dpa_reduction_obs`` for observation relations. Its
``reduction_obs`` context records contributor roles; the current output marks
measurement contributors as ``science``. Exact raw acquisitions and selected
calibration products remain linked by ``dpa_reduction_raw_obs`` and
``dpa_reduction_data_prod`` and described in saved raw observation contexts.
