#! /usr/bin/env python

from tollan.utils.log import get_logger
import functools
import copy
from tolteca.io.toltec import KidsModelParams


class KidsViewData(object):

    from .. import tolteca_toltec_datastore as _datastore

    logger = get_logger()

    def __init__(self, info, save):
        self.meta = copy.deepcopy(info)
        self._update_meta()
        self._update_fileinfo()
        self._load_data()
        self._save = functools.partial(save, self=self)

    def _update_meta(self):
        if self.meta['interface'].startswith('toltec'):
            self.meta['interface_type'] = 'toltec'
        else:
            self.meta['interface_type'] = None

        if self.meta['interface_type'] == 'toltec':
            self.meta['roach_id'] = int(
                    self.meta['interface'].lstrip('toltec'))

    def _update_fileinfo(self):
        if self.meta['interface_type'] is None:
            return
        if self.meta['interface_type'] == 'toltec':
            master = self.meta['Master'].lower()
            interface = self.meta['interface']
            obsid = self.meta['Obsnum']
            subobsid = self.meta['SubObsNum']
            scanid = self.meta['ScanNum']

            def _get_filepath(key, pattern):
                files = list(self._datastore.rootpath.glob(pattern))
                if not files:
                    self.logger.debug(
                            f"no {key} file found with pattern={pattern}")
                    return None
                if len(files) > 1:
                    self.logger.debug(
                        f"duplicated {key} file found with pattern={pattern}")
                    return None
                return files[0].as_posix()

            self.meta['filepaths'] = dict()
            for key, pattern in [
                    (
                        'raw_data',
                        f'{master}/{interface}/{interface}_'
                        f'{obsid:06d}_{subobsid:02d}_{scanid:04d}_*.nc'),
                    (
                        'model_params',
                        f'reduced/{interface}_'
                        f'{obsid:06d}_{subobsid:02d}_{scanid:04d}_*.txt'),
                    (
                        'reduced_data',
                        f'reduced/{interface}_'
                        f'{obsid:06d}_{subobsid:02d}_{scanid:04d}_*.nc'),
                    ]:
                path = _get_filepath(key, pattern)
                if path is not None:
                    self.meta['filepaths'][key] = path

    def _load_data(self):
        for key in ('raw_data', 'model_params', 'reduced_data'):
            if key in self.meta['filepaths']:
                setattr(self, key, getattr(self, f'_load_{key}')())

    def _load_raw_data(self):
        return None

    def _load_model_params(self):

        self._model_params = model_params = KidsModelParams(
                self.meta['filepaths']['model_params'])

        result = {
                'model_cls': model_params.model_cls.__name__,
                'param_names': list(model_params.model_cls.param_names),
                'n_models': len(model_params.model)
                }
        for name in result['param_names']:
            result[name] = getattr(model_params.model, name).value.tolist()
        result['x0'] = (
                model_params.model.f0 /
                model_params.model.fr - 1.).value.tolist()
        return result

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith('_')}

    def save(self):
        self._save()
