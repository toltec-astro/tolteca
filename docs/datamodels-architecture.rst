Framework architecture
======================

This page is for developers who need to know how ``tolteca_datamodels`` is built.
It is not needed to read, select or discover observation data; those workflows are
covered by :doc:`datamodels`, and :doc:`datamodels-api` lists the signatures.

``tolteca_datamodels`` is a framework for any data that arrives as files: its
``core`` packages know nothing about an instrument, and LMT and TolTEC are two data
models built on them. ``tolteca_dummy_instru`` is a minimal third data model
that exercises every extension point, so it is the shortest worked reference for
adding another one. The sections below follow the pieces a data model supplies.

.. list-table::
   :header-rows: 1
   :widths: 22 48 30

   * - Piece
     - Built from
     - Consumed by
   * - Filename layout
     - A ``FilepathInfo`` subclass registered with a filename parser; fields marked
       ``MetadataField`` initialize metadata columns
     - File tables, stores
   * - File workflow
     - ``FileAccessorWorkflow`` with row match rules, derived columns, and opener and
       reader policies; bound with ``FileAccessorBase[...]``
     - File accessors
   * - Composite
     - ``CompositeFileWorkflow`` listing component accessor types
     - The root accessor and ``partial()``
   * - Selector labels
     - ``ObsSpecContext`` mapping interface labels to numbers
     - File accessors, stores
   * - File store
     - ``FileStoreBase`` subclass that yields glob patterns, with optional substores
     - Search, input discovery
   * - Registration
     - ``DataModelSpec`` in ``data_models``; an input discovery entry
     - Root entry points, recipes

.. _observation-data-imports:

Core imports
------------

The core ``FileAccessorBase`` uses available parsers without loading instruments.
It projects supported fields as ``file_*`` columns and initializes unprefixed
columns for fields annotated with ``MetadataField``. The root entry point adds UIDs
and instrument classification; importing it does not change the core class's
projection rules. Imports differ in which filename parsers and data models they
enable:

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Import
     - Enables
   * - ``tolteca_datamodels.core.filepath``
     - ``FilepathInfo`` with every parser registered so far, including parsers
       loaded later; no instrument parsers by itself
   * - ``tolteca_datamodels.filepath``
     - The same class with the built-in LMT and TolTEC parsers
   * - ``tolteca_datamodels.core.file``
     - ``SourceSourceLoc`` and the core accessors, using the parsers already
       loaded; no instruments
   * - ``tolteca_datamodels.source``
     - ``SourceSourceLoc`` with the built-in filename support
   * - ``tolteca_datamodels.file``
     - The root ``FileAccessor`` and ``file_table_from_sources`` with the built-in
       filename support, assembled from the registered data models
   * - ``tolteca_dummy_instru.filepath``
     - The dummy instrument's filename parsers only
   * - ``tolteca_dummy_instru.register()``
     - Its ``DataModelSpec``, file store and input discovery entry

Neither filepath import restricts parsing to a selected data-model scope. Calling a
concrete model's ``from_source`` instead limits results to that class and its
registered descendants. An optional instrument loads its own filepath module to
provide filename metadata, and registers its ``DataModelSpec`` to enable
classification and the other processing of the root accessor.

A ``DataModelSpec`` associates a data-model identity with its file accessors and
stores, and a ``DataModelSet`` selects among the registered specs. File accessors
depend on neither: a concrete accessor works without a spec or a registry, and the
root entry point assembles the registrations.

Initialize metadata columns from filenames
------------------------------------------

To initialize a metadata column from a filename field, use the singleton
``MetadataField`` from ``tolteca_datamodels.core.filepath``. For example:

.. testcode:: extension

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

Define a file workflow
----------------------

A concrete file accessor provides row match rules, guess derivation, opener and
reader policies, metadata column mappings, and observation-selector interpretation.
None of them requires a data-model tag on the accessor class.

To implement a file-handling policy, subclass
:class:`~tolteca_datamodels.core.file.FileAccessorWorkflow` and declare
``_file_match_rules``. Bind it with ``FileAccessorBase[YourWorkflow]`` when defining
your :class:`~tolteca_datamodels.core.file.FileAccessorBase` subclass.
Implement ``_prepare_derived_columns(positions)``
to return column Series in the selected row order without changing the table.
Use ``_prepare_derived_column_defaults()`` for default Series covering all rows,
including empty and unmatched inputs. For example:

.. testcode:: extension

   import tolteca_datamodels.toltec.filepath  # enables TolTEC filename parsing
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

Group capabilities
------------------

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

Declare selector labels
-----------------------

A data model whose interfaces carry labels such as ``foo00`` declares their numbers
in an ``ObsSpecContext``, and reuses it in its file accessor and store. The same
parsed selector can be applied with different contexts. Label maps are shared Python
declarations: keep them unchanged while in use, and create another context for a
different interpretation.

Without a context, literal labels, numeric columns and explicit numeric-slot
patterns work, but bare numeric constraints do not match string labels. Contexts do
not restrict literal matching to the declared labels.

.. testcode:: extension

   import pandas as pd
   from tolteca_datamodels.core.obs_spec import ObsSpecContext
   from tolteca_datamodels.lmt_base.obs_spec import ObsSpec

   numbers = {"foo00": 0, "foo01": 1}
   context = ObsSpecContext(field_label_numbers={"interface": numbers})
   spec = ObsSpec.parse("1/(0,2)[-1]")
   foo_table = pd.DataFrame({
       "obsnum": [1, 1, 1],
       "master": ["tcs"] * 3,
       "subobsnum": [0] * 3,
       "scannum": [1] * 3,
       "interface": ["foo00", "foo01", "foo01"],
   })
   assert spec.select(foo_table, context=context).index.tolist() == [1, 2]
   assert spec.resolve_values("interface", list(numbers), context=context) == [
       "foo00",
       "foo01",
   ]
   assert ObsSpec.parse("1/(0,2)").select(foo_table).empty

Include unnumbered labels with ``None`` when declaring a context. For example,
``{"tel": None, "foo00": 0, "foo01": 1}`` keeps wildcard interface positions
unordered even when the table happens to contain only numbered interfaces.
Combining contexts with ``|`` returns a new context. It accepts identical
declarations and raises ``ValueError`` for conflicting numeric values for the same
field and label. Creating or importing another context does not change existing
selections.

Add a file store
----------------

A file store subclasses ``FileStoreBase`` (``LmtFileStoreBase`` for LMT-style
selectors) and implements ``_iter_local_glob_patterns(spec, subpath)``. It yields the
patterns of its own locations, relative to each subpath. ``_subpaths`` lists the
locations the store manages, where ``PurePosixPath(".")`` is the store root; it is empty by
default. ``spec.resolve_values()`` resolves the interface constraints against the
store's own labels before globbing, so the instrument name need not equal its
interface prefix.

A store contains the stores registered beneath it. ``Substore(Child, subpath=...)`` as
a class attribute registers the child at ``parent.path / subpath`` and names the
selection descriptor. The subpath follows Tollan's ``SubPath`` contract, which
normalizes it. Each class has one registered subpath per parent, and registering it
again raises ``ValueError``, even at the same subpath. A subclass inherits its parent's
registrations, and ``_substore_paths`` holds them per class.

.. testcode:: extension

   from pathlib import PurePosixPath
   from tolteca_datamodels.core.file_store import FileStoreBase, StoreView, Substore
   from tolteca_datamodels.lmt_base.file_store import LmtFileStoreBase

   interfaces = {f"bar{i}": i for i in range(4)}

   class BarFileStore(LmtFileStoreBase):
       _obs_spec_context = ObsSpecContext(field_label_numbers={"interface": interfaces})
       _subpaths = (PurePosixPath("."),)

       def _iter_local_glob_patterns(self, spec, subpath):
           obsnum = str(spec.obsnum) if isinstance(spec.obsnum, int) else "*"
           interface_labels = spec.resolve_values(
               "interface", list(interfaces), context=self._obs_spec_context
           )
           for interface in interface_labels:
               yield f"{interface}/{interface}_{obsnum}_*.nc"

   class SiteFileStore(LmtFileStoreBase):
       bar = Substore(BarFileStore, subpath="./instruments//bar/")

       def _iter_local_glob_patterns(self, spec, subpath):
           return iter(())

   site_store = SiteFileStore("/data")
   assert site_store.substores[BarFileStore].path == site_store.path / "instruments/bar"
   assert isinstance(site_store.bar, StoreView)
   assert list(site_store.iter_glob_patterns("1/(1,5)")) == [
       "instruments/bar/bar1/bar1_1_*.nc",
       "instruments/bar/bar2/bar2_1_*.nc",
       "instruments/bar/bar3/bar3_1_*.nc",
   ]
   assert list(site_store.iter_glob_patterns("1/{bar0,bar2}")) == [
       "instruments/bar/bar0/bar0_1_*.nc",
       "instruments/bar/bar2/bar2_1_*.nc",
   ]

The bare bounds ``(1,5)`` resolve to ``bar1`` through ``bar3`` because ``bar0`` is
number 0, and the prefixed form ``toltec(1,5)`` constrains another instrument's
interfaces, so this store yields nothing for it. A store that does not support the
selector type skips its own files but still visits its substores.

.. testcode:: extension

   assert list(site_store.iter_glob_patterns("1/toltec(1,5)")) == []
   try:
       SiteFileStore.register_substore(BarFileStore, "other")
   except ValueError:
       pass
   else:
       raise AssertionError("expected ValueError")

Registering a child after the class is defined, as an optional instrument does,
uses ``ParentStore.register_substore(ChildStore, subpath, name="child")``. It raises
``ValueError`` on re-registration, a cycle, or a name that is invalid or already
used, and ``pydantic.ValidationError`` for an invalid subpath. The optional name makes
``store.child`` and ``ParentStore.child`` selection descriptors available. Register
before constructing stores: construction creates the children without searching
directories or opening files, and later registrations affect only stores constructed
afterwards.

The dummy instrument
--------------------

``DummyInstruFileStore`` and ``DummyInstruFileAccessor`` are the dummy instrument's
store and composite accessor, and ``tolteca_dummy_instru.register()`` enables them in
assembled workflows by doing three things:

* registering its ``DataModelSpec`` with ``data_models``;
* registering ``DummyInstruFileStore`` under ``LmtFileStore`` as ``dummy_instru``,
  at the subpath of the instrument name;
* adding the ``data_dummy_instru_path`` entry to the input discovery configuration.

Importing the package does not register them, and repeated calls are harmless.
``DummyInstruFileAccessor`` composes separate counter and companion-telescope accessors. Either child also works directly without
registering the spec, and they still retain every row and use all available
filename parsers. A dummy telescope file has both ``lmt`` and ``dummy_instru``
memberships, so it matches the LMT and dummy accessors, but not TolTEC:

.. testcode:: extension

   import tolteca_dummy_instru.filepath  # filename parsers only
   from tolteca_datamodels.lmt.file import LmtFileAccessor
   from tolteca_datamodels.toltec.file import ToltecFileAccessor
   from tolteca_dummy_instru.file import DummyInstruFileAccessor

   dummy_names = [
       "chan01-tcs-1-0-2-20251031T120000.nc",
       "tel_dummy_instru_2025-10-31_000001_00_0002.nc",
       "tel_toltec_2025-10-31_145609_00_0002.nc",
   ]
   tables = {
       cls: cls.from_sources(dummy_names).table
       for cls in (LmtFileAccessor, ToltecFileAccessor, DummyInstruFileAccessor)
   }
   uid_missing = {cls: t.uid_raw_obs_file.isna().tolist() for cls, t in tables.items()}
   assert uid_missing[LmtFileAccessor] == [True, False, False]
   assert uid_missing[ToltecFileAccessor] == [True, True, False]
   assert uid_missing[DummyInstruFileAccessor] == [False, False, True]
   assert tables[DummyInstruFileAccessor].uid_raw_obs_file.iloc[1] == "tcs-1-0-2-tel"

Each row is handled by one selected accessor. With the root accessor and both LMT
and TolTEC enabled, TolTEC handles their shared telescope rows and LMT handles the other
telescope rows; dummy support has the same precedence over LMT for dummy telescope
files. Selecting only LMT still handles either instrument's telescope files. The dummy
counter and telescope policies share the query shorthand ``counts`` and ``telescope``,
which a composite exposes because they agree.
