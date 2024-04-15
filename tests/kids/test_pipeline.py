from dataclasses import dataclass, field

import pytest
from pydantic import Field

from tolteca_kids.pipeline import SequentialPipeline, Step, StepConfig, StepContext


class MyConfig(StepConfig):
    a: str
    b: str = "not_set"


@dataclass(kw_only=True)
class MyContextData:
    result: str = ...
    b: str = ...


class MyStepContext(StepContext["MyStep", MyConfig]):
    data: MyContextData = Field(default_factory=MyContextData)


@dataclass(kw_only=True)
class MyData:
    meta: dict = field(default_factory=dict)
    value: str


class MyStep(Step[MyConfig, MyStepContext]):

    @classmethod
    def run(cls, data: MyData, context):
        cfg = context.config
        ctd = context.data
        ctd.result = data.value + cfg.a
        ctd.b = len(ctd.result)
        return True


def test_mystep():

    with pytest.raises(ValueError, match="validation error"):
        MyStep()

    with pytest.raises(ValueError, match="not a valid MyConfig instance"):
        MyStep("a")

    s0 = MyStep(MyConfig(a="a"))
    d0 = s0(MyData(value="d"))
    c0 = MyStep.get_context(d0)
    assert c0.data.result == "da"
    assert c0.completed

    # rebuild step
    assert c0.make_step().config == s0.config

    # sequential pipeline steps of same class overrite
    s1 = MyStep(MyConfig(a="b"))
    p = SequentialPipeline(steps=[s0, s1])
    d1 = p(MyData(value="d"))
    c1 = MyStep.get_context(d1)
    assert c1.data.result == "db"
    assert c1.completed

    # sequential pipeline steps of same class overrite
    MyStep2 = MyStep.alias("Step2")
    s2 = MyStep2(MyConfig(a="c"))
    p2 = SequentialPipeline(steps=[s0, s1, s2])
    d2 = p2(MyData(value="d"))
    c1 = MyStep.get_context(d2)
    c2 = MyStep2.get_context(d2)
    assert c1.data.result == "db"
    assert c1.completed
    assert c2.data.result == "dc"
    assert c2.completed
    ctxs = p2.get_contexts(d2)
    assert MyStep2.context_key in ctxs
    assert MyStep.context_key in ctxs
