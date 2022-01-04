#!/usr/bin/env python


from schema import Schema, Optional, Use, Literal, Or
from tollan.utils.dataclass_schema import DataclassNamespace
from tollan.utils.log import get_logger
from astropy.table import Table
from .. import instrument_registry, sources_registry
from ...utils.common_schema import RelPathSchema
from .lmt import lmt_info as site_info
from ...cal import ToltecCalib
from ...utils import get_pkg_data_path
from .simulator import ToltecObsSimulator, ToltecHwpConfig
from .models import ToltecPowerLoadingModel


__all__ = [
    'site_info', 'ToltecObsSimulatorConfig', 'ToltecHwpConfig',
    'ToltecPowerLoadingModelConfig']


def _load_calobj(index_filepath, allow_fallback=True):
    """Load calibration object from index file path `p`."""
    logger = get_logger()
    try:
        return ToltecCalib.from_indexfile(index_filepath)
    except Exception:
        if allow_fallback:
            logger.debug(
                'invalid calibration object index file path,'
                ' fallback to builtin default')
            default_cal_indexfile = get_pkg_data_path().joinpath(
                    'cal/toltec_default/index.yaml')
            return ToltecCalib.from_indexfile(default_cal_indexfile)
        else:
            raise


def _load_array_prop_table(filepath):
    try:
        return Table.read(filepath, format='ascii.ecsv')
    except Exception:
        return None


@instrument_registry.register('toltec')
class ToltecObsSimulatorConfig(DataclassNamespace):
    """The config class for TolTEC observation simulator."""

    _namespace_from_dict_schema = Schema({
        Optional(
            'polarized',
            default=False,
            description='The toggle of whether to simulate polarized signal.'):
        bool,
        Optional(
            'hwp',
            default=ToltecHwpConfig,
            description='The dict contains the HWP config.',
            ): ToltecHwpConfig.schema,
        # TODO make this explicit, not fallback
        Optional(
            'calobj_index',
            default='cal/default/index.yaml',
            description='The calibration object index file path.'):
        RelPathSchema(),
        Optional(
            'array_prop_table',
            default=None,
            description='The array prop table to use instead of the one '
                        'provided in the calobj_index.'):
        RelPathSchema(),
        })

    _namespace_to_dict_schema = Schema({
        'hwp': Use(lambda c: c.to_dict()),
        str: object
        })

    logger = get_logger()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.polarized:
            simu_cls = ToltecObsSimulator
        # get array prop table
        calobj = self.calobj = _load_calobj(
            self.calobj_index, allow_fallback=True)
        if self.array_prop_table is not None:
            apt = _load_array_prop_table(self.array_prop_table)
            self.logger.info(f"use user input array prop table:\n{apt}")
        else:
            apt = calobj.get_array_prop_table()
            self.logger.info(f"use array prop table from calobj {calobj}")
        self.observer = simu_cls.observer
        self.simulator = simu_cls(
            array_prop_table=apt,
            polarized=self.polarized, hwp_config=self.hwp)

    def __call__(self, cfg):
        return self.simulator


@sources_registry.register('toltec_power_loading')
class ToltecPowerLoadingModelConfig(DataclassNamespace):
    """The config class to compute the TolTEC detector power loading.
    """

    _namespace_from_dict_schema = Schema({
        Literal('atm_model_name', description='The atmosphere model to use.'):
        Or(None, 'am_q25', 'am_q50', 'am_q75', 'toast')
        })

    def __call__(self, cfg):
        return ToltecPowerLoadingModel(
            atm_model_name=self.atm_model_name)
