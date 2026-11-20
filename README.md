# TolTECA

TolTECA provides Python tools for inspecting LMT and TolTEC observations,
selecting inputs, processing detector data, and working with scientific results.
You can inspect a single acquisition in Python or use configured workflows to
prepare and process a collection.

Start with the [observation-data guide](docs/datamodels.md) to read metadata,
build a file table, or discover inputs. The
[documentation index](docs/index.rst) links the processing and data-product domains.

## Installation

This development version requires Python 3.13 or newer and the matching Tollan
v3 environment. In an existing development environment, install this checkout
with:

```console
python -m pip install -e .
```

## Choose a task

| Task | Start here |
| --- | --- |
| Inspect telescope, detector, or HWPR data; select inputs | [Observation data](docs/datamodels.md) |
| Run KIDs processing | [Configured KIDs reduction](docs/kids-reduction.md) |
| Reopen and inspect scientific products | [Data-product API](docs/data-products-api.rst) |
| Work directly with scientific signal arrays | [KIDs signal processing](docs/kidsproc/index.md) |

Developers who add data models can read the
[framework architecture](docs/datamodels-architecture.rst).

## Development

Use the repository's `just` recipes in the configured development environment:

```console
just qa
just doc-build
just doc
```

`just qa` runs the configured checks and tests. `just doc-build` performs a strict
HTML build, and `just doc` serves the documentation locally.

Observation-data tests use the deployment's read-only reference acquisitions.
To include comparisons with the v2 implementation, set `TOLTECA_V2_PYTHON` to
the Python executable in a separate environment with TolTECA v2 installed.
From the repository root, run the telescope and filename comparisons with the
v3 environment's Python:

```console
TOLTECA_V2_PYTHON=/path/to/v2/bin/python python -m pytest \
    tests/tolteca_datamodels/lmt/test_tel_parity.py \
    tests/tolteca_datamodels/test_recorded_filenames.py
```

Without this variable, cross-environment comparisons skip; the other tests in
these modules still run.

The project uses the BSD 3-Clause license.
