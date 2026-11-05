"""Read-only data shared by tests and executable source examples."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_PACKAGE_ROOT = Path(__file__).parent
_LOCAL_DEV_LOCATION = _PACKAGE_ROOT.parent / "toltec_astro_dev"
_IO_EXAMPLE_MODULES = {
    _PACKAGE_ROOT / "src/tolteca_datamodels" / module
    for module in (
        "core/file/accessor.py",
        "lmt/file.py",
        "lmt/tel.py",
        "toltec/kids/core.py",
    )
}


@pytest.fixture(scope="session")
def data_lmt_path() -> Path:
    """Read-only reference tree from the active or local development deployment."""
    deploy_root = os.environ.get("TOLTECA_DEPLOY_ROOT")
    location_root = (
        Path(deploy_root).expanduser()
        if deploy_root is not None
        else _LOCAL_DEV_LOCATION
    )
    path = location_root / "run" / "data_lmt"
    if not path.is_dir():
        pytest.skip(f"deployed data_lmt test fixture not found at {path}")
    return path


@pytest.fixture(autouse=True)
def _datamodel_io_examples(request: pytest.FixtureRequest) -> None:
    """Provide the documented data_lmt directory to acquisition I/O examples."""
    if request.path not in _IO_EXAMPLE_MODULES:
        return
    data_root = request.getfixturevalue("data_lmt_path")
    example_root = request.getfixturevalue("tmp_path")
    (example_root / "data_lmt").symlink_to(data_root, target_is_directory=True)
    request.getfixturevalue("monkeypatch").chdir(example_root)
