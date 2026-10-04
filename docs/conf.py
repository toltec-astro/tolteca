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
    ("py:obj", "tolteca_kidsproc.analysis.d21.D21Analysis.all fields"),
    ("py:obj", "tolteca_kidsproc.analysis.psd.PSDAnalysis.all fields"),
]

# Resolve imported annotation aliases to dependency inventory entries. The two
# Kids container types are typing-only; the returned objects are xarray types.
_REFERENCE_ALIASES = {
    "MappingABC": "collections.abc.Mapping",
    "MappedField": "tollan.accessor.MappedField",
    "NameMapping": "tollan.accessor.NameMapping",
    "SubpathMapping": "tollan.accessor.SubpathMapping",
    "XarraySourceT": "tollan.accessor.xarray.utils.XarraySourceT",
    "tollan.accessor.xarray.utils.XarraySourceT": (
        "tollan.accessor.xarray.utils.XarraySourceT"
    ),
    "Quantity": "astropy.units.Quantity",
    "npt.NDArray": "numpy.typing.NDArray",
    "np.floating": "numpy.floating",
    "xr.DataArray": "xarray.DataArray",
    "xr.DataTree": "xarray.DataTree",
    "KidsDataset": "xarray.Dataset",
    "KidsDataTree": "xarray.DataTree",
}


class _ModelFieldDocumenter(PydanticFieldDocumenter):
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
    # This PEP 695 alias has no object entry in Tollan's inventory; its public
    # quantity guide specifies accepted units and serialization.
    if target == "tollan.config.types.quantity.FrequencyQuantityField":
        root = _TOLLAN_DOCS_URL.rstrip("/")
        return nodes.reference(
            "",
            "",
            contnode,
            refuri=f"{root}/validation-types.html#quantities",
            reftitle=target,
        )
    if target in _REFERENCE_ALIASES:
        node = node.copy()
        node["reftarget"] = _REFERENCE_ALIASES[target]
        node["reftype"] = "obj"
        return intersphinx.resolve_reference_detect_inventory(env, node, contnode)
    return None


def setup(app):
    app.add_autodocumenter(_ModelFieldDocumenter, override=True)
    app.connect("missing-reference", _resolve_reference_alias)
