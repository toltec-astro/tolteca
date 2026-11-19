Observation data API
====================

Signatures, fields and defaults for the observation-data packages. Start with
:doc:`datamodels` to choose an entry point, or follow :doc:`read-acquisition`,
:doc:`file-tables`, :doc:`file-stores` and :doc:`input-discovery` for reading
acquisitions, collecting file information, searching data trees and selecting
inputs. :doc:`obs_spec` describes observation-selector syntax. Developers who add
filename conventions, file workflows, stores or instruments can read
:doc:`datamodels-architecture`.

File tables and source access
-----------------------------

.. automodapi:: tolteca_datamodels.file

.. automethod:: tolteca_datamodels.file.FileAccessor.from_sources

.. automodapi:: tolteca_datamodels.core.file
   :skip: FileAccessorBase, FileAccessorView, SourceSourceLoc, SourceTrace, SourceTraceEntry

.. Shared workflow methods are inherited; document them without duplicating
   Tollan's ModelTableAccessor API or exposing the private delegation mixin.

.. autoclass:: tolteca_datamodels.core.file.FileAccessorBase
   :members:
   :inherited-members: ModelTableAccessor

.. autoclass:: tolteca_datamodels.core.file.FileAccessorView
   :members:
   :inherited-members:

.. autoclass:: tolteca_datamodels.core.file.SourceTraceEntry
   :members:

.. autoclass:: tolteca_datamodels.core.file.SourceTrace
   :members:

.. autopydantic_model:: tolteca_datamodels.core.file.SourceSourceLoc
   :class-doc-from: class

.. The Pydantic documenter renders this model's own methods and properties but
   omits the computed properties inherited from the location base.

.. autoattribute:: tolteca_datamodels.core.file.SourceSourceLoc.source_loc_is_local

.. autoattribute:: tolteca_datamodels.core.file.SourceSourceLoc.source_loc_host

.. autoattribute:: tolteca_datamodels.core.file.SourceSourceLoc.source_loc_scheme

.. automodapi:: tolteca_datamodels.core.source
   :skip: SourceOpener, SourceReaderRules

.. The policy contains live callables; its repr includes process-specific addresses.
.. autodata:: tolteca_datamodels.core.source.source_opener
   :no-value:

.. Legacy autodoc loses overload signatures when rendering inherited methods.
   Keep inherited fields here and link matching methods to their owning API below.

.. autopydantic_model:: tolteca_datamodels.core.source.SourceOpener
   :class-doc-from: class
   :model-show-json: False
   :inherited-members: BaseModel
   :exclude-members: __call__, map_table

.. autopydantic_model:: tolteca_datamodels.core.source.SourceReaderRules
   :class-doc-from: class
   :model-show-json: False
   :inherited-members: BaseModel
   :exclude-members: __call__, map_table

Both opener and reader policies support matching a record with
:meth:`~tollan.accessor.pandas.model_table.RecordRuleMapper.__call__` or a table with
:meth:`~tollan.accessor.pandas.model_table.RecordRuleMapper.map_table`.
Matching returns the selected callable or accessor class without opening
sources or reading metadata.

.. Source policies contain live callables, not
   JSON configuration. Their callable fields remain documented above.

Use the built-in ``source_opener`` or a custom
:class:`~tolteca_datamodels.core.source.SourceOpener` for direct format
opening. Use a named accessor directly for an already-open object with a known
reader. File accessors resolve the selected data models' opener and reader
policies when the reader must be selected.

.. automodapi:: tolteca_datamodels.lmt.file

.. automodapi:: tolteca_datamodels.toltec.file

Filename information
--------------------

Parse a filename with ``FilepathInfo`` from ``tolteca_datamodels.filepath``;
:ref:`observation-source-access` describes how sources are prepared, and
:ref:`observation-data-imports` states what each import enables.

.. automodapi:: tolteca_datamodels.core.filepath
   :skip: FilepathInfo

.. autopydantic_model:: tolteca_datamodels.core.filepath.FilepathInfo
   :class-doc-from: class

.. autopydantic_model:: tolteca_datamodels.lmt.filepath.LmtFilepathInfo
   :class-doc-from: class

.. autopydantic_model:: tolteca_datamodels.toltec.filepath.ToltecFilepathInfo
   :class-doc-from: class

.. autopydantic_model:: tolteca_datamodels.toltec.filepath.ToltecLmtFilepathInfo
   :class-doc-from: class

.. _observation-input-discovery:

Discovery and stores
--------------------

Import ``InputDiscovery`` from ``tolteca_datamodels.input_discovery`` for configured
discovery with built-in input forms, or ``query_file_store`` for a direct store
query without runtime configuration. Use ``data_models`` from
``tolteca_datamodels.data_model`` to look up registered definitions and select
workflow capabilities.

.. automodapi:: tolteca_datamodels.core.data_model
   :skip: DataModelSpec, DataModelSet

.. autodata:: tolteca_datamodels.data_model.data_models
   :no-value:

.. automethod:: tolteca_datamodels.data_model.data_models.register

.. autoclass:: tolteca_datamodels.data_model.DataModelSet
   :members: from_select

.. autopydantic_model:: tolteca_datamodels.core.data_model.DataModelSpec
   :class-doc-from: class
   :model-show-json: False

.. autoclass:: tolteca_datamodels.core.data_model.DataModelSet
   :members: from_specs, select

.. These definitions contain Python accessor and store classes. They are live
   registrations; InputDiscoveryConfig below is the serializable configuration.

.. automodapi:: tolteca_datamodels.core.file_store

.. automodapi:: tolteca_datamodels.lmt.file_store

.. automodapi:: tolteca_datamodels.toltec.file_store

.. automodapi:: tolteca_datamodels.lmt_base.file
   :skip: metadata_column_mapper

.. autodata:: tolteca_datamodels.lmt_base.file.metadata_column_mapper
   :no-value:

.. automodapi:: tolteca_datamodels.lmt_base.file_store

.. autofunction:: tolteca_datamodels.core.input_discovery.query_file_store

.. autopydantic_model:: tolteca_datamodels.core.input_discovery.InputDiscoveryConfig
   :class-doc-from: class

.. autoclass:: tolteca_datamodels.core.input_discovery.InputDiscovery
   :members: discover

.. autopydantic_model:: tolteca_datamodels.core.input_discovery.InputDiscoveryEntry
   :class-doc-from: class

.. autopydantic_model:: tolteca_datamodels.core.input_discovery.FileStoreInputDiscovery
   :class-doc-from: class

.. autopydantic_model:: tolteca_datamodels.core.input_discovery.GlobSource
   :class-doc-from: class

.. autopydantic_model:: tolteca_datamodels.core.input_discovery.SourcesInputDiscovery
   :class-doc-from: class

.. autopydantic_model:: tolteca_datamodels.lmt_base.input_discovery.LmtBaseFileStoreInputDiscovery
   :class-doc-from: class

.. autopydantic_model:: tolteca_datamodels.lmt.input_discovery.LmtInputDiscovery
   :class-doc-from: class

.. autopydantic_model:: tolteca_datamodels.toltec.input_discovery.ToltecInputDiscovery
   :class-doc-from: class

Observation selectors
---------------------

``query_obs_spec()`` accepts text, an observation number or a parsed selector
using the selected workflows' common selector class. A partial view with one
workflow accepts the same selectors as that workflow's concrete accessor.
If the selected workflows use different selector classes, pass a parsed selector.
Positional constraints apply once across the combined table.

.. automodapi:: tolteca_datamodels.core.obs_spec
   :skip: ColumnSelector, FieldConstraint, FieldPattern, FieldSelection, ValueBounds, ObsSpecContext

.. The automodapi attribute-summary template tries to import required dataclass
   fields as class attributes. Use autodoc directly for the selector dataclasses.

.. autoclass:: tolteca_datamodels.core.obs_spec.ColumnSelector
   :members:

.. autoclass:: tolteca_datamodels.core.obs_spec.FieldConstraint
   :members:

.. autoclass:: tolteca_datamodels.core.obs_spec.FieldPattern(before, selection, after)
   :members:

.. Pydantic's generated FieldPattern signature contains an Annotated expression
   with an f-string which Sphinx 9 cannot parse. Parameter documentation preserves
   the accepted values; only the displayed signature is specified here.

.. autoclass:: tolteca_datamodels.core.obs_spec.FieldSelection
   :members:

.. autoclass:: tolteca_datamodels.core.obs_spec.ValueBounds
   :members:

.. autoclass:: tolteca_datamodels.core.obs_spec.ObsSpecContext
   :members:

.. automodapi:: tolteca_datamodels.lmt_base.obs_spec

Acquisition readers
-------------------

.. automodapi:: tolteca_datamodels.lmt.tel

.. automodapi:: tolteca_datamodels.toltec.hwpr
   :skip: ToltecHwprMetadata

.. autopydantic_model:: tolteca_datamodels.toltec.hwpr.ToltecHwprMetadata
   :class-doc-from: class

.. automodapi:: tolteca_datamodels.toltec.kids
   :skip: ToltecBlockIndex

.. automodapi:: tolteca_datamodels.toltec.axes

Shared metadata and field types
-------------------------------

.. automodapi:: tolteca_datamodels.core.types
   :skip: FrozenDataModel

.. autopydantic_model:: tolteca_datamodels.core.types.FrozenDataModel
   :class-doc-from: class

.. automodapi:: tolteca_datamodels.lmt_base.funcs

.. autopydantic_model:: tolteca_datamodels.lmt_base.metadata.LmtObsQuartetMixin
   :class-doc-from: class

.. autopydantic_model:: tolteca_datamodels.lmt_base.metadata.InterfaceInfoMixin
   :class-doc-from: class

.. autopydantic_model:: tolteca_datamodels.lmt_base.metadata.LmtRawObsMetadata
   :class-doc-from: class

.. autopydantic_model:: tolteca_datamodels.lmt_base.filepath.FilepathInfoMixin
   :class-doc-from: class

.. autopydantic_model:: tolteca_datamodels.lmt_base.filepath.FilepathOrigMetadataMixin
   :class-doc-from: class

.. autopydantic_model:: tolteca_datamodels.lmt.metadata.LmtTelMetadata
   :class-doc-from: class

.. automodapi:: tolteca_datamodels.lmt.types
   :skip: LmtInterfaceInfo

.. autopydantic_model:: tolteca_datamodels.lmt.types.LmtInterfaceInfo
   :class-doc-from: class

.. automodapi:: tolteca_datamodels.toltec.types
   :skip: ToltecDataKind

.. automodapi:: tolteca_datamodels.toltec.toltec_db
   :skip: IntEnum

.. _native-kids-metadata:

Native KIDs files
-----------------

.. currentmodule:: None

.. py:class:: tolteca_datamodels.toltec.metadata.ToltecRawObsMetadata

   Recorded acquisition headers available through ``ds.toltec_kids.meta``.

.. py:class:: tolteca_datamodels.toltec.metadata.ToltecProcessedKidsMetadata

   The processed file's ``filepath_orig`` identifies its source acquisition.
   ``filepath_info_orig`` interprets that name without reading the parent.
   If the writer supplied no source reference, ``ds.toltec_kids.meta`` returns None.

.. py:class:: tolteca_datamodels.toltec.metadata.ToltecRawObsBlocksMetadata

   Acquisition metadata, discovered block count, ``block_index``, ``block_index_table``,
   ``chan_axis_table`` and ``probe_axis_table``. Available through
   ``ds.toltec_kids.block_index.meta`` before block creation, and through
   the prepared block collection's ``meta`` afterwards.

.. py:class:: tolteca_datamodels.toltec.metadata.ToltecRawObsBlockMetadata

   Acquisition fields and scalar geometry for one recorded block.

.. autoclass:: tolteca_datamodels.toltec.kids.blocks.ToltecBlockIndex

.. autoclass:: tolteca_datamodels.toltec.types.ToltecDataKind
   :members:

.. toctree::

   datamodels-types

See :doc:`data-products` for portable product indexes and publication.
