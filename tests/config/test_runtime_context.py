import tempfile
from pathlib import Path

from tollan.utils.log import logger

from tolteca_config import __version__
from tolteca_config.core import (
    RuntimeContext,
    ConfigModel,
    ConfigHandler,
    SubConfigKeyTransformer,
)


def _create_sample_config_file(filepath):
    with filepath.open("w") as fo:
        fo.write(
            """
---
workflow:
  a: 1
  b: null
  c: 'some_value'
""",
        )


def test_runtime_context():
    with tempfile.TemporaryDirectory() as _tmp:
        tmp = Path(_tmp)
        f = tmp / "some_config.yaml"
        _create_sample_config_file(f)
        rc = RuntimeContext(f)
        assert rc.config.model_dump() == {
            "workflow": {
                "a": 1,
                "b": None,
                "c": "some_value",
            },
            "runtime_info": rc.runtime_info.model_dump(),
        }
        assert rc.runtime_info.version == __version__
    logger.debug(f"config:\n{rc.config.model_dump_yaml()}")


class SimpleWorkflowConfig(ConfigModel):
    """A simple workflow config for testing."""

    a: int
    b: None | int = None
    c: None | str = None


class SimpleWorkflow(
    SubConfigKeyTransformer["workflow"],
    ConfigHandler[SimpleWorkflowConfig],
):
    """A simple workflow for testing."""


def test_simple_workflow():
    with tempfile.TemporaryDirectory() as _tmp:
        tmp = Path(_tmp)
        f = tmp / "some_config.yaml"
        _create_sample_config_file(f)
        rc = RuntimeContext(f)
        # check construction
        wf0 = SimpleWorkflow(rc)
        wf1 = SimpleWorkflow(f)
        assert wf0.rc.config_backend.dict(
            exclude_runtime_info=True,
        ) == wf1.rc.config_backend.dict(exclude_runtime_info=True)
        # from __getitem__
        wf2, wf3 = rc[SimpleWorkflow, SimpleWorkflow]

        assert wf0.config == wf1.config == wf2.config == wf3.config
        # check config
        wf = wf0
        assert wf.config.model_dump() == {
            "a": 1,
            "b": None,
            "c": "some_value",
        }
        logger.debug(f"runtime config:\n{wf.runtime_config.model_dump_yaml()}")
        logger.debug(f"workflow config:\n{wf.config.model_dump_yaml()}")

        # update config
        wf.update_config({"c": "other_value"})
        assert wf.config.model_dump() == {
            "a": 1,
            "b": None,
            "c": "other_value",
        }
        logger.debug(f"workflow config:\n{wf.config.model_dump_yaml()}")
