"""Sphinx configuration using Astropy's extension defaults."""

import os
from dataclasses import MISSING, fields, is_dataclass
from inspect import signature
from pathlib import Path
from typing import get_type_hints

from docutils import nodes
from sphinx.ext import intersphinx
from sphinx.ext.autodoc import SUPPRESS
from sphinx.util import typing as sphinx_typing
from sphinx.util.inspect import stringify_signature
from sphinx.util.typing import stringify_annotation
from sphinx_astropy.conf.v2 import *  # pyright: ignore[reportWildcardImportFromLibrary]  # noqa: F403
from sphinx_astropy.conf.v2 import exclude_patterns, extensions
from sphinxcontrib.autodoc_pydantic.directives.autodocumenters import (
    PydanticFieldDocumenter,
)

import tolteca

# Astropy supplies numpydoc, automodapi, viewcode, and inheritance diagrams.
extensions += [
    "myst_parser",
    "sphinx.ext.doctest",
    "sphinxcontrib.autodoc_pydantic",
    "sphinxcontrib.typer",
]
# autodoc-pydantic 2.2 needs the Documenter API to render model methods under
# Sphinx 9. Remove when it supports the new autodoc engine.
autodoc_use_legacy_class_based = True
# Keep named opening policies readable instead of expanding their callable rules.
autodoc_preserve_defaults = True
# Ordinary backticks can contain parameter names, rather than Python objects.
default_role = None
# PSD imports its generic input view only for typing; distinguish it from the
# instrument view documented on the reducer API page.
autodoc_type_aliases = {
    "KidsTimestreamView": "tolteca_kidsproc.accessors.views.TimestreamView",
}
myst_heading_anchors = 3
nitpicky = True
# Pytest executes source docstrings; Sphinx owns explicit guide testcode blocks.
doctest_test_doctest_blocks = ""
# Guide examples that read reference data or use native tools locate them from
# the development workspace root (the folder holding the Tolteca repositories).
doctest_global_setup = (
    "from pathlib import Path\n"
    f"workspace = Path({str(Path(__file__).resolve().parents[2])!r})\n"
)

_TOLLAN_DOCS_URL = os.environ.get(
    "TOLLAN_DOCS_URL", "https://toltec-astro.github.io/tollan/"
)
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "sqlalchemy": ("https://docs.sqlalchemy.org/en/20/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "dask": ("https://docs.dask.org/en/stable/", None),
    # Use a matching local inventory when documenting unpublished Tollan APIs.
    "tollan": (
        _TOLLAN_DOCS_URL,
        os.environ.get("TOLLAN_DOCS_INVENTORY"),
    ),
}

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
master_doc = "index"
project = "tolteca"
copyright = "2026, Zhiyuan Ma"
author = "Zhiyuan Ma"
version = release = tolteca.__version__
language = "en"
exclude_patterns += ["Thumbs.db", ".DS_Store"]
html_theme = "sphinx_book_theme"
html_title = "TolTECA documentation"
html_static_path = []

# Legacy autodoc treats generic parameters as classes; the Pydantic renderer
# links its synthetic "all fields" validator label as an object.
nitpick_ignore = [
    ("py:class", "DataSourceT"),
    ("py:class", "MapperT"),
    ("py:class", "MetaT"),
    ("py:class", "ProductT"),
    # Function/class type parameters, not importable objects.
    ("py:class", "HandlerT"),
    ("py:class", "RuntimeContextT"),
    ("py:class", "ObsSpecT"),
    # Plotly publishes no Sphinx inventory for Figure.
    ("py:class", "go.Figure"),
    ("py:class", "plotly.graph_objs._figure.Figure"),
    ("py:obj", "tolteca_kidsproc.analysis.d21.D21Analysis.all fields"),
    ("py:obj", "tolteca_kidsproc.analysis.psd.PSDAnalysis.all fields"),
    # Same autodoc-pydantic synthetic label for a cross-field input validator.
    ("py:obj", "tolteca_quicklook.config.SavedInputConfig.all fields"),
    ("py:obj", "tolteca_kids.tlaloc_output.TlalocOutputConfig.all fields"),
    # These model validators have the same autodoc-pydantic synthetic target.
    ("py:obj", "tolteca_datamodels.raw_obs_context.context.RawObsContext.all fields"),
    ("py:obj", "tolteca_datamodels.raw_obs_context.context.RawObsDataItem.all fields"),
    (
        "py:obj",
        "tolteca_datamodels.raw_obs_context.calibration.SweepCalItem.all fields",
    ),
]

# Resolve imported annotation aliases to dependency inventory entries. The two
# Kids container types are typing-only; the returned objects are xarray types.
_REFERENCE_ALIASES = {
    "Figure": "matplotlib.figure.Figure",
    "Session": "sqlalchemy.orm.Session",
    "ColumnElement": "sqlalchemy.sql.expression.ColumnElement",
    "UUID": "uuid.UUID",
    "AbstractContextManager": "contextlib.AbstractContextManager",
    "MappingABC": "collections.abc.Mapping",
    "DriveFitStepContext": "tolteca_kids.drive_fit.step.DriveFitStepContext",
    "Artifact": "tollan.pipeline.Artifact",
    "ArtifactContext": "tollan.pipeline.ArtifactContext",
    "ArtifactFile": "tollan.pipeline.ArtifactFile",
    "MappedField": "tollan.accessor.MappedField",
    "NameMapping": "tollan.accessor.NameMapping",
    "SubpathMapping": "tollan.accessor.SubpathMapping",
    "XarraySourceT": "tollan.accessor.xarray.utils.XarraySourceT",
    "tollan.accessor.xarray.utils.XarraySourceT": (
        "tollan.accessor.xarray.utils.XarraySourceT"
    ),
    "Quantity": "astropy.units.Quantity",
    "QTable": "astropy.table.QTable",
    "pd.DataFrame": "pandas.DataFrame",
    "ConfigDict": "pydantic.ConfigDict",
    "JsonValue": "pydantic.types.JsonValue",
    "StringSelector": "tollan.config.types.StringSelector",
    # The parameterized Pydantic base has no separate inventory entry.
    "tollan.accessor.pandas.model_table.RecordRuleMapper[type]": (
        "tollan.accessor.pandas.model_table.RecordRuleMapper"
    ),
    "NonNegativeInt": "pydantic.NonNegativeInt",
    "npt.NDArray": "numpy.typing.NDArray",
    "numpy._typing._array_like.NDArray": "numpy.ndarray",
    "np.floating": "numpy.floating",
    "xr.DataArray": "xarray.DataArray",
    "xr.Dataset": "xarray.Dataset",
    "Path": "pathlib.Path",
    "FileLoc": "tollan.utils.fileloc.FileLoc",
    "RuntimeContext": "tollan.config.RuntimeContext",
    "RuntimeInfo": "tollan.config.RuntimeInfo",
    "xr.DataTree": "xarray.DataTree",
    "ToltecSweepMetadata": "tolteca_datamodels.toltec.metadata.ToltecSweepMetadata",
    "KidsDataset": "xarray.Dataset",
    "KidsDataTree": "xarray.DataTree",
    "LmtTelDataset": "xarray.Dataset",
    "ToltecKidsDataset": "xarray.Dataset",
    "ToltecHwprDataset": "xarray.Dataset",
}

# Legacy autodoc drops the owning class when qualifying PEP 695 aliases nested
# in LmtInfo/ToltecInfo. Link those annotations to the actual public type.
_LOCAL_REFERENCE_ALIASES = {
    f"tolteca_datamodels.{module}.types.{name}": (
        f"tolteca_datamodels.{module}.types.{owner}.{name}"
    )
    for module, owner, names in (
        ("lmt", "LmtInfo", ("InstruT", "MasterT")),
        (
            "toltec",
            "ToltecInfo",
            (
                "InstruT",
                "MasterT",
                "ObsTypeT",
                "ObsTypeCodeT",
                "ArrayT",
                "ArrayNameT",
                "RoachT",
            ),
        ),
    )
    for name in names
} | {
    # autodoc-pydantic's model-validator label refers to its owning model.
    "tolteca_datamodels.data_prod.review.DataProdReview.all fields": (
        "tolteca_datamodels.data_prod.DataProdReview"
    ),
    "tolteca_quicklook.config.QuicklookConfig.all fields": (
        "tolteca_quicklook.config.QuicklookConfig"
    ),
    "tolteca_reduce.config.InputConfig.all fields": "tolteca_reduce.config.InputConfig",
    "tolteca_reduce.config.ReduceConfig.all fields": (
        "tolteca_reduce.config.ReduceConfig"
    ),
    "tolteca_datamodels.raw_obs_context.calibration.PointingCalItem.all fields": (
        "tolteca_datamodels.raw_obs_context.calibration.PointingCalItem"
    ),
    "tolteca_datamodels.raw_obs_context.calibration.PhotometryCalItem.all fields": (
        "tolteca_datamodels.raw_obs_context.calibration.PhotometryCalItem"
    ),
    # The Viz methods import this view only for typing; autodoc-pydantic leaves
    # their method annotations unqualified.
    "SweepCheckView": "tolteca_kids.sweep_check_view.SweepCheckView",
    # autodoc-pydantic links its synthetic all-fields validator label.
    "tolteca_datamodels.core.data_prod.index.DataProdIndexBase.all fields": (
        "tolteca_datamodels.core.data_prod.DataProdIndexBase"
    ),
    "tolteca_datamodels.core.data_prod.namespace.DataProdNamespaceIndex.all fields": (
        "tolteca_datamodels.data_prod.DataProdNamespaceIndex"
    ),
    "tolteca_datamodels.data_prod.index.DataProd.all fields": (
        "tolteca_datamodels.data_prod.DataProd"
    ),
    "tolteca_datamodels.data_prod.index.DataProdIndex.all fields": (
        "tolteca_datamodels.data_prod.DataProdIndex"
    ),
    "tolteca_datamodels.data_prod.calibration.DataProdCalRef.all fields": (
        "tolteca_datamodels.data_prod.DataProdCalRef"
    ),
    "tolteca_datamodels.core.file.source.SourceSourceLoc.all fields": (
        "tolteca_datamodels.core.file.SourceSourceLoc"
    ),
    # Pydantic reports the defining module for re-exported PEP 695 aliases.
    **{
        f"tolteca_datamodels.core.data_prod.run.{name}": (
            f"tolteca_datamodels.data_prod.{name}"
        )
        for name in ("CounterRunNameFormat", "RunName", "RunUid")
    },
    "tolteca_kids.config.KidsConfig.all fields": "tolteca_kids.config.KidsConfig",
    "FileTableModeT": "tolteca_datamodels.core.file.workflow.FileTableModeT",
    "SourceErrorPolicyT": "tolteca_datamodels.core.file.trace.SourceErrorPolicyT",
    **{
        name: f"tolteca_datamodels.core.obs_spec.selectors.{name}"
        for name in ("Selector", "FieldValue", "FieldValues")
    },
    # The parameterized Pydantic base is not a separately documented class.
    (
        "tolteca_datamodels.core.input_discovery.entries."
        "FileStoreInputDiscovery[Annotated[ObsSpec, PlainValidator, PlainSerializer]]"
    ): ("tolteca_datamodels.core.input_discovery.FileStoreInputDiscovery"),
}


class _ModelFieldDocumenter(PydanticFieldDocumenter):
    def add_constraints(self):
        # autodoc-pydantic 2.2 also reports validator/serializer dataclass
        # attributes as constraints. Keep actual value constraints, without
        # function addresses or serialization bookkeeping in the reference.
        constraints = self.pydantic.inspect.fields.get_constraints(
            self.pydantic_field_name
        )
        for key in ("func", "json_schema_input_type", "return_type", "when_used"):
            constraints.pop(key, None)
        if constraints:
            source = self.get_sourcename()
            self.add_line(":Constraints:", source)
            for key, value in constraints.items():
                self.add_line(f"   - **{key}** = {value}", source)
            self.add_line("", source)

    def update_annotations(self, parent):
        # Sphinx 9's AttributeDocumenter copies inherited source annotations
        # onto the child class. That corrupts later Pydantic generic rebuilds:
        # inherited ClassVars can become fields in the child's module namespace.
        # Pydantic already owns the resolved field annotations used below.
        pass

    # autodoc-pydantic 2.2 retries raw annotations, including TYPE_CHECKING-only
    # names. Use Pydantic's resolved annotations and omit Annotated metadata.
    # Preserve the extension's default fields, validators, and JSON schema.
    def add_directive_header(self, sig):
        self.options.annotation = SUPPRESS
        super().add_directive_header(sig)
        field = self.pydantic.inspect.fields.get(self.pydantic_field_name)

        def field_type():
            pass

        field_type.__annotations__ = {"return": field.annotation}
        mode = (
            "smart"
            if self.config.autodoc_typehints_format == "short"
            else "fully-qualified"
        )
        annotation = stringify_annotation(
            get_type_hints(field_type)["return"], mode=mode
        )
        self.add_line(f"   :type: {annotation}", self.get_sourcename())


def _resolve_reference_alias(app, env, node, contnode):
    target = node.get("reftarget")
    # These PEP 695 aliases have no object entries in Tollan's inventory.
    # Link to the corresponding validation contracts instead.
    guides = {
        "tollan.config.types.path.AbsFilePath": "paths-and-file-locations",
        "tollan.config.types.quantity.FrequencyQuantityField": "quantities",
        "tollan.config.types.path.AbsAnyPath": "paths-and-file-locations",
        "tollan.config.types.path.AnyPath": "paths-and-file-locations",
        "tollan.config.types.path.SubPath": "paths-and-file-locations",
        "SubPath": "paths-and-file-locations",
        "tollan.config.types.fileloc.FileLocField": "paths-and-file-locations",
        "tollan.config.types.quantity.TimeQuantityField": "quantities",
        "tollan.config.types.time.UnixTimeField": "times",
        "tollan.config.types.skycoord.SkyCoordField": "coordinates",
    }
    if target in guides:
        root = _TOLLAN_DOCS_URL.rstrip("/")
        return nodes.reference(
            "",
            "",
            contnode,
            refuri=f"{root}/validation-types.html#{guides[target]}",
            reftitle=target,
        )
    # autodoc-pydantic renders type aliases as class references; dependencies
    # publish these under non-class inventory roles.
    if target in {
        "tollan.config.types.name.PathSegment",
        "tollan.plot.plotly.PlotlyFigExportFormat",
        "pydantic.types.JsonValue",
    }:
        node["reftype"] = "obj"
        return intersphinx.resolve_reference_detect_inventory(env, node, contnode)
    if target in _REFERENCE_ALIASES:
        node = node.copy()
        node["reftarget"] = _REFERENCE_ALIASES[target]
        node["reftype"] = "obj"
        return intersphinx.resolve_reference_detect_inventory(env, node, contnode)
    if target in _LOCAL_REFERENCE_ALIASES:
        return env.domains.python_domain.resolve_xref(
            env,
            node["refdoc"],
            app.builder,
            "obj",
            _LOCAL_REFERENCE_ALIASES[target],
            node,
            contnode,
        )
    return None


# Dataclass signatures otherwise expose Pydantic's Annotated validation objects
# as user-facing parameter types. Keep the ordinary types without mutating them.
def _dataclass_signature(app, what, name, obj, options, sig, return_annotation):  # noqa: PLR0913, PLR0917
    if what != "class" or not is_dataclass(obj):
        return None
    # Keep annotations that name TYPE_CHECKING-only imports, as autodoc does.
    hints = sphinx_typing.get_type_hints(obj)
    resolved = signature(obj)
    # <factory> is not valid Python signature syntax; it makes Sphinx split
    # generic annotations at their commas. Keep the type and mark that default.
    factories = {f.name for f in fields(obj) if f.default_factory is not MISSING}
    resolved = resolved.replace(
        parameters=[
            param.replace(
                annotation=hints.get(key, param.annotation),
                default=... if key in factories else param.default,
            )
            for key, param in resolved.parameters.items()
        ]
    )
    return stringify_signature(
        resolved,
        show_annotation=app.config.autodoc_typehints in {"signature", "both"},
        show_return_annotation=False,
        unqualified_typehints=app.config.autodoc_typehints_format == "short",
        short_literals=app.config.python_display_short_literal_types,
    ), return_annotation


# Legacy autodoc leaves these TYPE_CHECKING annotations unqualified, making
# the core capability container ambiguous with its assembled root subclass.
def _qualify_discovery_signature(app, what, name, obj, options, sig, result):  # noqa: PLR0913, PLR0917
    if sig and name.startswith("tolteca_datamodels.core.input_discovery."):
        return sig.replace(
            "DataModelSet", "tolteca_datamodels.core.data_model.DataModelSet"
        ), result
    return None


def setup(app):
    app.connect("autodoc-process-signature", _qualify_discovery_signature)
    app.connect("autodoc-process-signature", _dataclass_signature, priority=100)
    app.add_autodocumenter(_ModelFieldDocumenter, override=True)
    app.connect("missing-reference", _resolve_reference_alias)
