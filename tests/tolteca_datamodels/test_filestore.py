"""Shared file settings are independent of application processing config."""

import pytest
from pydantic import ValidationError
from tollan.config import RuntimeContext

from tolteca_datamodels.filestore import FileAccess, FileAccessConfig
from tolteca_kids.core import KidsConfig


def test_shared_file_config_does_not_create_products(tmp_path):
    products = tmp_path / "products"
    runtime = RuntimeContext(
        {
            "files": {
                "data_lmt_path": str(tmp_path),
                "dataprod_toltec_path": str(products),
                "select": "roach < 7",
            },
            "kids": {"sweep_reduce": {"block": 1}},
        }
    )
    files = FileAccess(runtime)
    assert files.config.data_lmt_path == tmp_path
    assert files.config.dataprod_toltec_path == products
    assert files.config.select == "roach < 7"
    assert not products.exists()


def test_shared_file_config_snapshot_and_typo_validation():
    runtime = RuntimeContext({"files": {"select": "roach < 7"}})
    files = FileAccess(runtime)
    runtime.update_override_config({"files": {"select": "roach >= 7"}})
    assert files.config.select == "roach < 7"
    assert files.recreate().config.select == "roach >= 7"
    with pytest.raises(ValidationError, match="extra_forbidden"):
        FileAccessConfig.model_validate({"select": "roach == 0"})


@pytest.mark.parametrize("name", ["data_lmt_path", "dataprod_toltec_path", "select"])
def test_kids_config_does_not_accept_file_access_settings(name):
    with pytest.raises(ValidationError, match="extra_forbidden"):
        KidsConfig.model_validate({name: "unused"})
