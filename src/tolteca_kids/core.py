from functools import cached_property
from typing import Literal

from pydantic import Field

from tolteca_config.core import ConfigHandler, ConfigModel, SubConfigKeyTransformer

from .dataprod_output import DataProdOutput, DataProdOutputConfig
from .kids_find import KidsFind, KidsFindConfig, KidsFindPlot, KidsFindPlotConfig
from .pipeline import SequentialPipeline
from .sweep_check import (
    SweepCheck,
    SweepCheckConfig,
    SweepCheckPlot,
    SweepCheckPlotConfig,
)
from .tlaloc_output import TlalocOutput, TlalocOutputConfig

__all__ = ["Kids"]


class KidsConfig(ConfigModel):
    """The config for KIDs processing."""

    sweep_check: SweepCheckConfig = Field(default_factory=SweepCheckConfig)
    sweep_check_plot: SweepCheckPlotConfig = Field(default_factory=SweepCheckPlotConfig)

    kids_find: KidsFindConfig = Field(default_factory=KidsFindConfig)
    kids_find_plot: KidsFindPlotConfig = Field(default_factory=KidsFindPlotConfig)

    tlaloc_output: TlalocOutputConfig = Field(default_factory=TlalocOutputConfig)
    output: DataProdOutputConfig = Field(default_factory=DataProdOutputConfig)


class Kids(SubConfigKeyTransformer[Literal["kids"]], ConfigHandler[KidsConfig]):
    """The class to work with KIDs data."""

    @ConfigHandler.auto_cache_reset
    @cached_property
    def sweep_check(self):
        """The sweep check step."""
        return SweepCheck(self.config.sweep_check)

    @ConfigHandler.auto_cache_reset
    @cached_property
    def sweep_check_plot(self):
        """The sweep check plot step."""
        return SweepCheckPlot(self.config.sweep_check_plot)

    @ConfigHandler.auto_cache_reset
    @cached_property
    def kids_find(self):
        """The kids find step."""
        return KidsFind(self.config.kids_find)

    @ConfigHandler.auto_cache_reset
    @cached_property
    def kids_find_plot(self):
        """The kids find plot step."""
        return KidsFindPlot(self.config.kids_find_plot)

    @ConfigHandler.auto_cache_reset
    @cached_property
    def tlaloc_output(self):
        """The tlaloc output step."""
        return TlalocOutput(self.config.tlaloc_output)

    @ConfigHandler.auto_cache_reset
    @cached_property
    def output(self):
        """The tlaloc output step."""
        return DataProdOutput(self.config.output)

    @property
    def pipeline(self):
        """The pipeline."""
        return SequentialPipeline(
            steps=[
                self.sweep_check,
                self.sweep_check_plot,
                self.kids_find,
                self.kids_find_plot,
                self.tlaloc_output,
                self.output,
            ],
        )
