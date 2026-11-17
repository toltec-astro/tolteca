References, associations and tags
=================================

A product can depend on another product, belong to a collection or represent one
version of a result. Use a :class:`~tolteca_datamodels.data_prod.DataProdAssoc`
for the relationship and a :class:`~tolteca_datamodels.data_prod.DataProdRef`
for its target. A :class:`~tolteca_datamodels.data_prod.DataProdTag` selects an
exact product realization. The association type explains what the link means;
``is_primary`` separately marks the link used for storage and browsing.

Distinguish the relationships
-----------------------------

.. list-table::
   :header-rows: 1

   * - Relationship
     - Question it answers
     - Example
   * - Primary
     - Under which logical entity is this result grouped?
     - A processed acquisition points to its raw acquisition.
   * - Scientific input or calibration
     - Which products contributed to this result?
     - A science result records its observations and selected calibrations.
   * - Collection membership
     - What belongs to this named collection?
     - A project contains observations of several targets.
   * - Version membership
     - What are the alternative realizations of this logical result?
     - A processing result has versions from two executions.

Associations relate products. Data items belong directly to a product; they are
not association targets. Acquisition calibration evidence is also available per
interface through ``DataProd.cal_refs``. Preparing calibration bindings from that
evidence belongs to :doc:`raw-obs-context`.

Resolve a product's primary
---------------------------

Use :meth:`~tolteca_datamodels.data_prod.DataProd.get_primary` to find the
observation, acquisition or named collection that groups a result for storage
and browsing. Supply a lookup keyed by complete product references, including
namespace UUIDs. For example, resolve a processed acquisition to its raw product:

.. doctest::

   >>> from tolteca_datamodels.data_prod import DataProd, DataProdAssoc
   >>> from tolteca_datamodels.data_prod.metadata import (
   ...     RawObsMeta, RawObsProcMeta,
   ... )
   >>> raw = DataProd(
   ...     uid="tcs-145609-0-1",
   ...     meta=RawObsMeta(
   ...         name="Tune", data_prod_type="dp_raw_obs",
   ...         master="tcs", obsnum=145609, subobsnum=0, scannum=1,
   ...     ),
   ... )
   >>> processed = DataProd(
   ...     uid="tcs-145609-0-1_proc",
   ...     meta=RawObsProcMeta(
   ...         name="Processed tune", data_prod_type="dp_raw_obs_proc",
   ...         master="tcs", obsnum=145609, subobsnum=0, scannum=1,
   ...     ),
   ...     assocs=[DataProdAssoc(
   ...         uid=raw.uid, is_primary=True,
   ...         data_prod_assoc_type="dpa_raw_obs_proc_raw_obs",
   ...     )],
   ... )
   >>> products = {raw.reference: raw, processed.reference: processed}
   >>> processed.get_primary(products.__getitem__).reference == raw.reference
   True

Resolution follows the marked primary associations. It does not traverse other
inputs, calibration links or group membership. Observations, acquisitions and
named collections return themselves, without a lookup. An empty version group
can resolve its primary before it contains any versions.

A missing target raises ``KeyError``. Missing or ambiguous primary links, a
lookup result with a different UID or namespace, and cycles raise ``ValueError``.
Resolution does not read files or create directories; the supplied lookup owns
any retrieval it needs. See :doc:`named-group-products` for publishing results
under a named collection.

Distinguish a version from its group
------------------------------------

A version group has a stable logical result UID and declares its members' product
type. Each realization is a separate product with its own metadata, items,
namespace and producing run. A tag belongs to the group and selects one exact
realization, or is empty. Materialization records the group in the realization's
``meta.version_group``; a draft has none. ``write_index`` places a copied
realization by this reference.

.. graphviz::

   digraph product_versions {
     rankdir=LR;
     node [shape=box, fontname="Helvetica"];
     raw [label="Raw acquisition\ntcs-145609-0-1"];
     group [label="Version group\ntcs-145609-0-1_proc"];
     first [label="Realization from run A"];
     second [label="Realization from run B"];
     group -> raw [label="primary"];
     group -> first [label="version member"];
     group -> second [label="version member; main"];
     first -> raw [label="primary"];
     second -> raw [label="primary"];
   }

Moving ``main`` changes which result is selected; it does not alter either
realization. Directory labels such as ``0000-example`` are local run names,
not product references. :doc:`data-product-runs` explains their ownership.

For an empty tag, use ``current=None``. Tags remain editable:

.. doctest::

   >>> from tolteca_datamodels.data_prod import DataProdRef, DataProdTag
   >>> tag = DataProdTag(current=DataProdRef(uid="measurement:1"))
   >>> tag.current.uid
   'measurement:1'
   >>> tag.current = None
   >>> tag.current is None
   True

Read selected versions
----------------------

After creating a store tree with :doc:`data-product-publication`, use its version
group directory as ``product_directory`` and an execution UUID as ``run_uid``.

Pass a version-group directory to read its completed versions. Use execution
UUIDs to limit the returned members:

.. code-block:: python

   group = DataProdIndex.read(product_directory, run_uids=[run_uid])
   for member in group.products:
       print(member.product.reference)

The result retains membership edges and tags only for included products, matched
by both UID and namespace UUID. A tag for the same UID in another namespace is
not selected. Primary and other non-membership relationships remain unchanged.
An empty ``run_uids`` list returns no members or tags but retains those other
relationships. Omit ``run_uids`` to read all committed versions. Reading an index
does not open its artifact payloads.

Requested runs that are absent or not yet committed are omitted. A committed
record that points to a different run or disagrees with its realized product
raises ``ValueError``. Both live and expanded portable groups validate included
members against their declared product type. They do not require every referenced
member to be embedded or restrict the types of other supporting products.

For mixed-type collections and exact contributors, continue with
:doc:`named-group-products`. A named collection and a version group serve
different purposes: collection membership need not consist of alternative
results of one type.
