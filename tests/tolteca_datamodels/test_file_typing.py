"""Accessor factories preserve subclasses and their workflow bindings."""

from typing import assert_type

import pandas as pd
import pytest

from tolteca_datamodels.core.file import (
    CompositeFileAccessor,
    CompositeFileWorkflow,
    FileAccessorBase,
    FileAccessorWorkflow,
)
from tolteca_datamodels.lmt.file import LmtFileAccessor, LmtFileWorkflow
from tolteca_datamodels.toltec.file import ToltecFileAccessor, ToltecFileWorkflow
from tolteca_dummy_instru.file import DummyInstruFileAccessor, DummyInstruFileWorkflow


def test_workflow_binding_survives_generic_and_concrete_inheritance():
    class Intermediate[WorkflowT: FileAccessorWorkflow](FileAccessorBase[WorkflowT]):
        pass

    class Specialized(Intermediate[ToltecFileWorkflow]):
        pass

    class Inherited(Specialized):
        pass

    table = pd.DataFrame()
    file_accessor = Inherited(table)
    assert_type(Inherited.from_sources([]), Inherited)
    assert Inherited._workflow_cls is ToltecFileWorkflow
    assert isinstance(file_accessor._workflow, ToltecFileWorkflow)
    assert file_accessor._workflow.table is table
    assert file_accessor._workflow is file_accessor._workflow


def test_concrete_and_composite_accessors_expose_their_workflow_types():
    class Group[WorkflowT: CompositeFileWorkflow](CompositeFileAccessor[WorkflowT]):
        pass

    class DummyGroup(Group[DummyInstruFileWorkflow]):
        pass

    table = pd.DataFrame()
    assert_type(LmtFileAccessor.from_sources([]), LmtFileAccessor)
    assert_type(ToltecFileAccessor.from_sources([]), ToltecFileAccessor)
    assert_type(DummyInstruFileAccessor.from_sources([]), DummyInstruFileAccessor)
    assert isinstance(LmtFileAccessor(table)._workflow, LmtFileWorkflow)
    assert isinstance(ToltecFileAccessor(table)._workflow, ToltecFileWorkflow)
    assert isinstance(DummyInstruFileAccessor(table)._workflow, DummyInstruFileWorkflow)
    file_accessor = DummyGroup(table)
    assert_type(DummyGroup.from_sources([]), DummyGroup)
    assert isinstance(file_accessor._workflow, DummyInstruFileWorkflow)
    assert DummyGroup._workflow_cls is DummyInstruFileWorkflow
    assert file_accessor.view.accessor_types == file_accessor._workflow.accessor_types
    assert file_accessor.view.table is table


def test_workflow_binding_rejects_conflicting_or_invalid_declarations():
    with pytest.raises(TypeError, match="explicitly sets '_workflow_cls'"):

        class Conflicting(FileAccessorBase[ToltecFileWorkflow]):
            _workflow_cls = LmtFileWorkflow

    with pytest.raises(TypeError, match="FileAccessorWorkflow subclass"):

        class Invalid(FileAccessorBase[int]):  # ty: ignore[invalid-type-arguments]
            pass
