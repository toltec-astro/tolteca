Sources and locations
=====================

Read an index to inspect its product metadata, relationships and artifact
locations. Reading the document does not load the scientific payloads. This
works for a standalone index as well as an index in a product store.

A :class:`~tolteca_datamodels.data_prod.DataProdSource` names an item's relative
path and location key. The corresponding
:class:`~tolteca_datamodels.data_prod.Location` supplies its root.
Use :class:`~tolteca_datamodels.data_prod.DataProdIndex` to resolve those paths
and read the document that describes them.

Choose what to open
-------------------

.. list-table::
   :header-rows: 1

   * - You want
     - Operation
   * - One exact saved product
     - ``DataProdIndex.read(path_to_index_yaml)``
   * - The realization selected by a tag
     - ``resolve_index(group_directory, tag="main")``, then ``read``.
   * - All committed versions
     - ``DataProdIndex.read(group_directory)``
   * - Versions from selected executions
     - ``DataProdIndex.read(group_directory, run_uids=[...])``; see :doc:`data-product-relationships`.

In this example, ``product_directory`` is a version-group directory from
:doc:`data-product-publication`:

.. code-block:: python

   from tolteca_datamodels.data_prod import DataProdIndex, resolve_index

   path = resolve_index(product_directory, tag="main")
   index = DataProdIndex.read(path)
   for item in index.product.data_items:
       print(item.meta, index.get_source_uri(item.source))

Tell items apart
----------------

Each item of a TolTEC product is an acquisition file, an artifact written by a
producing run, or the product's own index file. The ``type`` key in its
metadata tells them apart:

.. list-table::
   :header-rows: 1

   * - Item
     - ``type``
     - Metadata
   * - Acquisition file
     - none, or ``roach``, ``hwpr``, ``tel``
     - The file's header metadata, such as ``interface``, ``obsnum``,
       ``subobsnum`` and ``scannum``; with a ``type``, an imported acquisition
       record.
   * - Artifact
     - ``artifact``
     - ``format``, ``name``, ``producer`` and ``run_uid``, and optional selection
       keys such as ``interface``, ``scope`` and ``kind``. Producer-specific
       values are under ``context``. See
       :class:`~tolteca_datamodels.data_prod.metadata.ArtifactMeta`.
   * - Index file
     - ``data_prod_index``
     - ``namespace_uid``, and ``run_uid`` for a realization. See
       :class:`~tolteca_datamodels.data_prod.metadata.IndexItemMeta`.

For example, list the artifacts with their producers:

.. code-block:: python

   for item in index.product.data_items:
       if item.meta.get("type") == "artifact":
           print(item.meta["name"], item.meta["producer"])

Resolve item paths
------------------

Each item's ``source.path`` is a :class:`~pathlib.PurePosixPath`, validated by
Tollan's ``SubPath`` field type. It names an item under the location selected by
``source.location``. Supply literal filename characters, including spaces,
``%``, ``#`` and ``?``; do not URL-encode the path.

``read`` preserves these relative paths. ``get_source_uri`` joins the path to
its location and returns an encoded, absolute URI without opening the artifact.
Relative location roots resolve against the opened index directory. For an exact
saved realization, pass its ``index.yaml`` directly to ``DataProdIndex.read``.

Choose a location root
----------------------

Declare a location with ``root``, for example ``{root: .}``, ``{root: ..}``,
or ``{root: 'archive:/data'}``. Use ``index.resolve_location(key)`` to obtain a
:class:`~tollan.utils.fileloc.FileLoc`; its ``url`` is the absolute URI and its
``path`` is the decoded path. Relative roots need the index directory and do not
use an application configuration root.

.. doctest::

   >>> from tolteca_datamodels.core.data_prod import DataProdSource, Location
   >>> source = DataProdSource.model_validate({
   ...     "location": "archive", "path": "scans/scan 50%#?.nc",
   ... })
   >>> source.path.as_posix()
   'scans/scan 50%#?.nc'
   >>> location = Location(root="https://archive.example/data")
   >>> location.resolve().joinpath(source.path).url
   'https://archive.example/data/scans/scan%2050%25%23%3F.nc'

Portable YAML stores this reference as ``source: {location: archive, path: ...}``.
Artifact tables expose the relative spelling in ``source_path`` and the resolved
URI in ``source``.

Move an index with its artifacts
--------------------------------

Moving an index and its relative artifacts together preserves their references.
Reopen the copied index to resolve its sources at the new location. Writing an
opened document preserves its references and leaves its existing resolution base
unchanged. YAML formatting and comments are not preserved. Reading uses safe
YAML tags; Python-specific object tags are rejected.

A location may also be an absolute local path or a remote FileLoc. Such a
reference continues to point to that location after moving the index. For a
self-contained copied tree, retain relative roots and copy the referenced
artifacts with their indexes.

See :doc:`data-product-transfer` when transporting records independently of
the original index files. For selecting artifact rows across indexes, see
:doc:`reduction-workdir` and the artifact model-table API in
:doc:`data-products-api`.
