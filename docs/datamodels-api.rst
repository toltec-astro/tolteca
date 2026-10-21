Observation data API
====================

See :doc:`guide` for reading acquisitions, collecting file information and selecting
inputs. :doc:`obs_spec` describes observation-selector syntax. The APIs below also
support extensions that add filename conventions, readers and discovery inputs.

File tables and source access
-----------------------------

Import ``FileAccessor`` from ``tolteca_datamodels.file`` to enable built-in LMT and
TolTEC support. Import the core class when defining an extension without loading
those instruments. Both use the registrations available when a table is built.

.. automodapi:: tolteca_datamodels.core.file
   :skip: SourceTraceEvent, FileAccessorView

.. autopydantic_model:: tolteca_datamodels.core.file.FileAccessorView
   :class-doc-from: class
   :model-show-json: False
   :special-members: __call__

.. autopydantic_model:: tolteca_datamodels.core.file.SourceTraceEvent
   :class-doc-from: class

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

Import ``FilepathInfo`` from ``tolteca_datamodels.filepath`` to parse built-in
LMT and TolTEC filenames. Importing ``tolteca_datamodels.core.filepath`` does not
load instrument parsers, but it exposes the same class and uses all available
registrations, including parsers loaded later. Neither import restricts parsing
to a selected data-model scope. Concrete result models support annotations and
instance checks.

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

Discovery and stores
--------------------

Import ``InputDiscovery`` from ``tolteca_datamodels.input_discovery`` for configured
discovery with built-in input forms, or ``query_file_store`` for a direct store
query without runtime configuration. Use ``data_models`` from
``tolteca_datamodels.data_model`` to look up registered definitions and select
workflow capabilities.

.. automodapi:: tolteca_datamodels.core.data_model
   :skip: DataModelSpec, DataModelSet

.. autodata:: tolteca_datamodels.core.data_model.data_models
   :no-value:

.. autopydantic_model:: tolteca_datamodels.core.data_model.DataModelSpec
   :class-doc-from: class
   :model-show-json: False

.. autopydantic_model:: tolteca_datamodels.core.data_model.DataModelSet
   :class-doc-from: class
   :model-show-json: False

.. These definitions contain Python accessor and store classes. They are live
   registrations; InputDiscoveryConfig below is the serializable configuration.

.. automodapi:: tolteca_datamodels.core.file_store

.. automodapi:: tolteca_datamodels.lmt.file_store

.. automodapi:: tolteca_datamodels.toltec.file_store

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

.. autopydantic_model:: tolteca_datamodels.core.obs_spec.ObsSpecContext
   :class-doc-from: class

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

.. toctree::

   datamodels-types

TolTEC acquisition and block metadata and data-product models are documented in
:doc:`kids-pipeline-api` alongside their processing consumers.
