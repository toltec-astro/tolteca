"""Sphinx configuration using Astropy's extension defaults."""

import os
from typing import get_type_hints

from docutils import nodes
from sphinx.ext import intersphinx
from sphinx.ext.autodoc import SUPPRESS
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

_TOLLAN_DOCS_URL = os.environ.get(
    "TOLLAN_DOCS_URL", "https://toltec-astro.github.io/tollan/"
)
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
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
    # Function/class type parameters, not importable objects.
    ("py:class", "HandlerT"),
    ("py:class", "RuntimeContextT"),
    ("py:class", "ObsSpecT"),
    # Plotly publishes no Sphinx inventory for Figure.
    ("py:class", "go.Figure"),
    ("py:class", "plotly.graph_objs._figure.Figure"),
    ("py:obj", "tolteca_kidsproc.analysis.d21.D21Analysis.all fields"),
    ("py:obj", "tolteca_datamodels.data_prod.DataProdIndex.all fields"),
    ("py:obj", "tolteca_kidsproc.analysis.psd.PSDAnalysis.all fields"),
    ("py:obj", "tolteca_kids.tlaloc_output.TlalocOutputConfig.all fields"),
]

# Resolve imported annotation aliases to dependency inventory entries. The two
# Kids container types are typing-only; the returned objects are xarray types.
_REFERENCE_ALIASES = {
    "MappingABC": "collections.abc.Mapping",
    "DriveFitStepContext": "tolteca_kids.drive_fit.step.DriveFitStepContext",
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
    # autodoc-pydantic links its synthetic all-fields validator label.
    "tolteca_datamodels.data_prod.DataProdIndexCalRef.all fields": (
        "tolteca_datamodels.data_prod.DataProdIndexCalRef"
    ),
    "tolteca_kids.config.KidsConfig.all fields": "tolteca_kids.config.KidsConfig",
    "FileTableModeT": "tolteca_datamodels.core.file.accessor.FileTableModeT",
    "SourceErrorPolicyT": (
        "tolteca_datamodels.core.file.trace.SourceErrorPolicyT"
    ),
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
    # autodoc-pydantic renders type aliases as class references; Tollan publishes
    # this format vocabulary as a Python type alias.
    if target == "tollan.plot.plotly.PlotlyFigExportFormat":
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


def setup(app):
    app.add_autodocumenter(_ModelFieldDocumenter, override=True)
    app.connect("missing-reference", _resolve_reference_alias)
