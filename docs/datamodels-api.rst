Observation data API
====================

See :doc:`guide` for reading acquisitions, collecting file information and selecting
inputs. :doc:`obs_spec` describes observation-selector syntax. The APIs below also
support extensions that add filename conventions, readers and discovery inputs.

File tables and source access
-----------------------------

Use ``tolteca_datamodels.file.FileAccessor.from_sources`` to collect filename
information without opening files by default. Pass ``mode="guess_or_read"`` to
read unidentified sources or ``mode="read"`` to also read matched sources.
It returns an accessor; its ``table`` property
provides the underlying DataFrame without copying. Use
``tolteca_datamodels.file.file_table_from_sources`` when you want the DataFrame
directly. Both enable built-in LMT and
TolTEC filename support and add observation columns such as ``obsnum`` and
``interface`` on parsed rows, alongside the universal ``file_*`` snapshots.
Matching rows also receive ``quartet``, ``uid_obs``, ``uid_raw_obs`` and
``uid_raw_obs_file`` from their observation identifiers. An identifier is null
when its required components are missing.
Optional instruments load their own filepath modules to provide filename
metadata and register their data-model spec to enable classification and other
processing through the root accessor.
Unfamiliar filenames provide ``file_ext`` and a base
``filepath_info`` with ``data_models=frozenset()``. By default, invalid recognized filenames also remain,
with a parsing failure in ``source_trace``. Use ``on_error="raise"`` to stop on
those failures. Invalid source/location pairs also retain a row by default,
with missing information columns and a validation failure in ``source_trace``.
A filename parsing failure retains its validated source and location.

Use ``file_accessor.source``, ``file_accessor.source_loc``,
``file_accessor.filepath_info``, ``file_accessor.data_models`` and
``file_accessor.source_trace`` to access the corresponding columns as pandas
Series. These properties preserve the table's index and
stored objects. They raise ``KeyError`` when the column is absent; accessing a
property does not create columns or process sources.

Pass a finite iterable of entries, such as a list, tuple or glob iterator.
The table follows iteration order. Wrap a single resource or source/location
dict in a list. An invalid outer collection always raises, independently of
``on_error``; invalid entries within a valid collection follow that policy.

.. code-block:: python

   from tolteca_datamodels.file import FileAccessor, file_table_from_sources

   file_accessor = FileAccessor.from_sources(["example.nc"])
   assert file_accessor.table.file_ext.tolist() == ["nc"]
   assert file_accessor.source_trace_summary().kind.tolist() == [
       "file_accessor_rule_unmatched"
   ]

   table = file_table_from_sources(["example.nc"])
   assert table.file_ext.tolist() == ["nc"]

Each row's ``source_trace`` is a :class:`~tolteca_datamodels.core.file.SourceTrace`.
It retains the latest issue of each kind; repeated attempts replace that kind's
details. Successful retries do not clear recorded failures.
Use ``trace.iter_entries()`` to inspect all recorded issues or
``trace.iter_error_entries()`` to inspect failures only. ``trace.has_error``
checks whether any failures were recorded. Filename parsing retains the parsed
``filepath_info``. A location without a matching filename pattern, such as
``/``, records ``filepath_rule_unmatched``. The root and concrete accessors also
match each valid row against their selected accessor rules. If none match, the
row records ``file_accessor_rule_unmatched``, including when ``data_models`` is
empty or missing. Both rule misses are informational, even with
``on_error="raise"``. Matching rows add no positive trace. A row with failed
source/location validation retains its failure without an accessor-rule miss.
These diagnostics are recorded before deriving observation columns, so they
remain available if derivation fails.
Use ``table.filepath_info.notna().sum()`` to count parsed filenames; no recorded
issues does not mean that every later processing step has run.
``file_accessor.source_trace_summary()`` counts the rows containing each
issue kind. Selecting or reordering rows retains their trace objects. Recording an
entry with ``trace.record(entry)`` also updates ``has_error``; the change is
visible in any table sharing that object.

Binding ``FileAccessor(table)`` requires no columns and does not modify the table.
``from_sources(sources)`` validates and installs source/location pairs in a new
table, then guesses filename information.
Marked filename metadata initializes missing unprefixed
columns before matching. Matching uses current columns, including null values,
without falling back to a filename snapshot.
For example:

.. code-block:: python

   file_accessor = FileAccessor.from_sources(["example.nc"])
   table = file_accessor.table
   table.index = [7]
   assert FileAccessor(table).table is table
   assert table.index.tolist() == [7]
   assert table.file_ext.tolist() == ["nc"]

Factories create the ``source_trace`` column. Retain this column when selecting
rows if you want to keep their recorded issues. ``file_accessor.source_trace``
returns the column once present;
``file_accessor.source_trace_summary()`` returns an empty summary when it is absent.

For a single resource, import ``SourceSourceLoc`` from
``tolteca_datamodels.source``. This enables built-in filename support:

.. code-block:: python

   from tolteca_datamodels.source import SourceSourceLoc

   item = SourceSourceLoc.model_validate("archive:/renamed.nc")
   assert item.filepath_info.ext == "nc"
   assert item.filepath_info.data_models == frozenset()
   assert item.source_loc_host == "archive"

Use ``with item.open() as resource:`` for local NetCDF or ECSV input. An explicit
``source_loc`` can supply the original filename for a renamed local source or an
already-open object. Opening uses the original ``source``, not ``source_loc``;
the default opener requires remote resources to be staged locally first.
Resources opened by this context are closed on exit when closable; borrowed
objects remain caller-owned. Pass a ``SourceOpener`` policy with
``item.open(opener=my_opener)`` to customize opening. Rules read the prepared
``filepath_info`` object; see ``SourceOpener`` for composing opening policies.

Standalone source validation and opening raise on failure and do not read metadata.
The batch factory records per-input preparation failures by default; its ``mode``
controls metadata reading. File accessors also provide the explicit opening and
metadata operations described below.

Importing ``SourceSourceLoc`` from ``core.file`` does not enable instruments, but
uses any filename registrations already loaded.

The core ``FileAccessorBase`` uses available parsers without loading instruments.
It projects supported fields as ``file_*`` columns and initializes unprefixed
columns for fields annotated with ``MetadataField``. Use the root entry point
for UIDs and instrument classification. Importing that entry point does not
change the core class's projection rules.

To initialize a metadata column from a filename field, use the singleton
``MetadataField`` from ``tolteca_datamodels.core.filepath``. For example:

.. code-block:: python

   from typing import Annotated
   from tolteca_datamodels.core.filepath import FilepathInfo, MetadataField

   class ObservationFilename(FilepathInfo):
       obsnum: Annotated[int, MetadataField]
       prefix: str

The marked field provides both ``file_obsnum`` and ``obsnum``; the unmarked field
provides only ``file_prefix``. Inherited fields retain their markers. If a subclass
redeclares a field with a narrower type, include the marker in that annotation too.
For LMT acquisition filenames, ``lmt_base.filepath.FilepathInfoMixin`` marks the
shared observation and interface fields automatically, including redeclarations.
The marker does not change validation, conversion units or dtypes, and does not
make filename information authoritative.

The root accessor uses the capabilities selected by all registered data models.
Concrete accessors from ``lmt.file`` and ``toltec.file`` apply their own matching
rules and UIDs without requiring a registered data-model spec.
``DummyInstruFileAccessor`` composes separate counter and companion-telescope
accessors. Either child also works directly without registering the dummy spec.
They still retain every row and use all available
filename parsers. For example, a TolTEC telescope file has both ``lmt`` and
``toltec`` memberships, so it matches either concrete accessor. A dummy
telescope file matches LMT and dummy support, not TolTEC.
Each row is handled by one selected accessor. With both LMT and TolTEC enabled,
TolTEC handles their shared telescope rows, and LMT handles other telescope
rows. Dummy support has the same precedence over LMT for dummy telescope files.
Selecting only LMT still handles either instrument's telescope files.
Matching uses current ``data_models``, not ``file_data_models`` or previous
traces. Marked metadata is available before matching; UIDs are derived once per
matched row.
Matching is repeated against current values for each operation; no selected
accessor class is retained in the table.

On a composite, use ``file_accessor.partial(accessor_types=(ToltecFileAccessor,))``
to narrow capabilities over the same table. It does not filter rows, run checks
or open resources.
The returned view retains its selected accessor classes, even if another data model is
registered later. Further ``partial()`` calls only narrow that selection.
``file_accessor.view`` creates an unrestricted view of the accessor's available
capabilities. To use a prepared ``DataModelSet``, pass its
``file_accessor_types`` to ``partial(accessor_types=...)`` or construct
``FileAccessorView(file_accessor, accessor_types=models.file_accessor_types)``.
The view's ``file_accessor`` property returns the originating accessor.
Concrete accessors do not provide ``partial()`` or ``view``; they already select
one policy.

Use ``with file_accessor.open() as table:`` on an accessor or partial view to open matching
and unidentified sources and identify their readers. The shared table has live ``io_obj`` and
``accessor_obj`` columns within that scope. On exit, columns from an enclosing
open scope are restored; otherwise both columns are removed.
Resources opened by the scope are closed then. Open objects supplied as sources
remain caller-owned.
Concrete and composite accessors expose these columns as ``file_accessor.io_obj``
and ``file_accessor.accessor_obj``. Both properties raise ``KeyError`` outside the open
scope.

``file_accessor.read_metadata()`` reads matching and unidentified
sources and replaces current observation columns with their metadata. It reuses live readers inside
``open()``; outside that scope it opens and closes the required resources.
Each explicit call reads and projects metadata, including rows populated by the
factory or an earlier call. Filename snapshots remain unchanged. Source
failures are recorded per row by default; ``on_error="raise"`` propagates them.
If no opener matches, the row records ``opener_rule_unmatched`` without raising.
Failure to obtain a live reader records ``reader_creation_failed`` and follows
``on_error``. This includes no reader being found after inspecting the opened
contents, an inspection error, or an error constructing the reader. Successful
content-based identification records no issue.
A metadata projection error always raises and leaves destination columns
unchanged. A reader returning no metadata remains openable, records
``metadata_unavailable`` and retains the row's existing metadata values.

To implement a file-handling policy, subclass
:class:`~tolteca_datamodels.core.file.FileAccessorWorkflow` and declare
``_file_match_rules``. Bind it with ``FileAccessorBase[YourWorkflow]`` when defining
your :class:`~tolteca_datamodels.core.file.FileAccessorBase` subclass.
Implement ``_prepare_derived_columns(positions)``
to return column Series in the selected row order without changing the table.
Use ``_prepare_derived_column_defaults()`` for default Series covering all rows,
including empty and unmatched inputs. For example:

.. code-block:: python

   from tollan.accessor.pandas.model_table import Contains
   from tolteca_datamodels.core.file import FileAccessorWorkflow, FileAccessorBase

   class FilenameFormatWorkflow(FileAccessorWorkflow):
       _file_match_rules = ({"data_models": Contains(value="toltec")},)

       def _prepare_derived_columns(self, positions):
           return {"format_name": self.table.file_ext.take(positions).str.upper()}

   class FilenameFormatAccessor(FileAccessorBase[FilenameFormatWorkflow]):
       pass

   file_accessor = FilenameFormatAccessor.from_sources([
       "toltec8_145605_000_0001_2025_10_31_09_20_06_tune.nc",
       "unidentified.nc",
   ])
   assert file_accessor.table.format_name.iloc[0] == "NC"
   assert file_accessor.table.format_name.isna().tolist() == [False, True]

Use ``_query_symbols`` to make named objects available
in both direct and composite queries, such as ``{"ToltecDataKind": ToltecDataKind}``.
Keep policy-local shorthand in the inherited ``_query_namespace`` setting.
A composite exposes shorthand only when all selected policies agree on its value.

To group capabilities, subclass
:class:`~tolteca_datamodels.core.file.CompositeFileWorkflow` and declare
``_component_accessor_types``. Bind it with ``CompositeFileAccessor[YourWorkflow]``
on a :class:`~tolteca_datamodels.core.file.CompositeFileAccessor` subclass to expose
``partial()``. The root entry point assembles its groups from
registered data-model specs. Selection can name a complete group or one of its children.
For dynamic availability, override ``_get_component_accessor_types()``.
When two policies overlap, declare ``_supersedes_workflow_types`` on the more
specialized concrete workflow to give it priority. Otherwise, the first matching
rule wins;
ambiguous matches are not detected. Cyclic composition or precedence declarations
raise when the selection or matching operation evaluates them.

.. code-block:: python

   from tolteca_datamodels.toltec.file import ToltecFileAccessor

   file_accessor = FileAccessor.from_sources([
       "tel_toltec_2025-10-31_145609_00_0002.nc",
       "example.nc",
   ])
   selected = file_accessor.partial(accessor_types=(ToltecFileAccessor,))
   assert selected.table is file_accessor.table
   assert len(selected.table) == 2
   assert selected.accessor_types == (ToltecFileAccessor,)
   assert selected.table.uid_raw_obs_file.iloc[0] == "tcs-145609-0-2-tel"

TolTEC support adds ``data_kind_toltec``; dummy-instrument support adds
``data_kind_dummy_instru``. These are separate nullable-integer flag columns,
not a combined ``data_kind``. Each enabled classifier uses its own ``unknown``
value for unsupported filenames, foreign files and invalid sources. Its column
is present even when there are no matching rows or the input is empty.
If derivation fails, existing derived columns retain their values and no new
derived columns are added. Filename information and matching diagnostics remain
available for inspection.

Use qualified enum names when querying a mixed table. A concrete accessor also
accepts its shorthand member names. A composite or view accepts a shorthand when
all selected concrete classes define the same object under that name; the dummy
counter and telescope policies therefore share ``counts`` and ``telescope``.
Querying a view evaluates the expression over all rows; it does not
implicitly discard rows outside the selected capability.

.. code-block:: python

   classified = FileAccessor.from_sources([
       "toltec8_145605_000_0001_2025_10_31_09_20_06_tune.nc",
       "example.nc",
   ])
   sweeps = classified.query(
       "bits_any_set(data_kind_toltec, ToltecDataKind.raw_sweep)"
   )
   assert sweeps.interface.tolist() == ["toltec8"]
   toltec = classified.partial(accessor_types=(ToltecFileAccessor,))
   assert toltec.query("bits_any_set(data_kind_toltec, unknown)").source.tolist() == [
       "example.nc"
   ]

Use ``iter_accessors()`` on a composite accessor or its partial view to process
disjoint groups of matching rows. Each pair contains row positions and a concrete
accessor over the **original full table**, not a sliced copy. Apply updates to
those positions to leave other rows unchanged, including when index labels repeat:

.. code-block:: python

   file_accessor.table["inspected"] = False
   for positions, accessor in selected.iter_accessors():
       assert isinstance(accessor, ToltecFileAccessor)
       assert accessor.table is file_accessor.table
       column = accessor.table.columns.get_loc("inspected")
       accessor.table.iloc[positions, column] = True
   assert file_accessor.table.inspected.tolist() == [True, False]

Matching runs once when iteration starts. All groups are determined before the
first is yielded, along with traces for unmatched or unidentified rows. Those
rows are not yielded but remain in the table. Groups follow their first matching
row. Matching again uses any changed values; an active iterator keeps its original
groups. You may add or update columns, but do not change row order or count during
iteration. Iterating alone does not derive columns or open resources.

To build a table using an explicit selection of capabilities, use
``FileAccessorView.from_sources``:

.. code-block:: python

   from tolteca_datamodels.core.file import FileAccessorView

   selected = FileAccessorView.from_sources([
       "toltec8_145605_000_0001_2025_10_31_09_20_06_tune.nc",
       "example.nc",
   ], accessor_types=(ToltecFileAccessor,))
   assert selected.table.roach.dropna().tolist() == [8]
   assert selected.table.file_ext.tolist() == ["nc", "nc"]

This produces the same columns and compatibility traces as
``ToltecFileAccessor.from_sources(...)`` with the same inputs. Both retain foreign
and unidentified rows; the selected capabilities determine which rows receive
UIDs and classifications. Guessing includes compatibility matching and
traces automatically. Composite views provide the same guessing, opening, metadata update and query
workflow with a restricted selection of concrete capabilities.

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

Import ``FilepathInfo`` from ``tolteca_datamodels.filepath`` to parse built-in
LMT and TolTEC filenames. Importing ``tolteca_datamodels.core.filepath`` does not
load instrument parsers, but it exposes the same class and uses all available
registrations, including parsers loaded later. Neither import restricts parsing
to a selected data-model scope. Calling a concrete model's ``from_source`` instead
limits results to that class and its registered descendants. For example,
``LmtFilepathInfo.from_source`` can return ``ToltecLmtFilepathInfo``, but cannot
return ``ToltecFilepathInfo`` or the generic base. No eligible match raises unless
``raise_on_error=False`` is supplied. Calling ``FilepathInfo.from_source`` on
unfamiliar names returns a base ``FilepathInfo`` with the final
suffix as ``ext`` (empty for extensionless names). Every parsed object provides
``data_models`` as a ``frozenset``. An empty set means the filename identifies no
data-model membership; it does not establish that the contents are unreadable.

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

.. toctree::

   datamodels-types

TolTEC acquisition and block metadata are documented in :doc:`kids-pipeline-api`.
See :doc:`data-products-api` for portable product indexes and publication.
