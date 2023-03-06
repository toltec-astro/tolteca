#!/usr/bin/env python

import functools
import subprocess
import re
import shutil
import os
import git
import pathlib

from copy import deepcopy
from cached_property import cached_property
from packaging.version import Version, InvalidVersion
from packaging.specifiers import SpecifierSet
from astropy.table import Table, vstack
from scalpl import Cut
import numpy as np
import astropy.units as u
from pathlib import Path
from schema import Or
from dataclasses import dataclass, field, replace
from typing import Union
from tollan.utils.dataclass_schema import add_schema
from tollan.utils import rupdate
from tollan.utils import call_subprocess_with_live_output
from tollan.utils.log import get_logger, timeit
from tollan.utils import ensure_abspath
from tollan.utils.fmt import pformat_yaml

from .base import PipelineEngine, PipelineEngineError
from ...utils import get_user_data_dir
from ...utils.misc import get_nested_keys
from ...utils.common_schema import RelPathSchema, PhysicalTypeSchema
from ...utils.runtime_context import yaml_load
from ...datamodels.toltec.data_prod import ToltecDataProd
from ...datamodels.toltec.basic_obs_data import BasicObsData
from ...common.toltec import toltec_info


REMOTE_CITLALI_REPO_URL = 'https://github.com/toltec-astro/citlali.git'
LOCAL_CITLALI_REPO_PATH = get_user_data_dir().joinpath("engines/citlali")
GIT_REMOTE_TIMEOUT = 10.


@functools.lru_cache(maxsize=1)
def _get_local_citlali_repo():
    """Return the Citlali git repo made available in the user data directory.

    This repo is maintained by this package for check commit histories and
    comparing versions.
    """
    logger = get_logger()

    if LOCAL_CITLALI_REPO_PATH.exists():
        repo = git.Repo(LOCAL_CITLALI_REPO_PATH)
        try:
            repo.remote(name='origin').fetch(
                kill_after_timeout=GIT_REMOTE_TIMEOUT)
            return repo
        except Exception:
            logger.debug(
                f"error fetching local citlali repo in "
                f"{LOCAL_CITLALI_REPO_PATH}")
        return repo

    logger.debug(f"setup local citlali repo in ${LOCAL_CITLALI_REPO_PATH}")

    # class _Progress(git.remote.RemoteProgress):
    #     def update(self, op_code, cur_count, max_count=None, message=''):
    #         logger.debug(self._cur_line)

    repo = git.Repo.clone_from(
        REMOTE_CITLALI_REPO_URL,
        LOCAL_CITLALI_REPO_PATH,
        # progress=_Progress()
        )
    return repo


UNKNOWN_VERSION = 'unknown'
"""Value used when the version is unknown."""


class CitlaliExec(object):
    """A low level wrapper class to run Citlali executable."""

    logger = get_logger()

    def __init__(self, path):
        path = self._path = ensure_abspath(path)
        version = self._version = self.get_version(path)
        if version is UNKNOWN_VERSION:
            raise ValueError(f"invalid Citlali executable path {path}")

    def __repr__(self):
        return f'{self.__class__.__name__}(version={self.version})'

    @property
    def path(self):
        """The path of the Citlali executable."""
        return self._path

    @property
    def version(self):
        """The version of the Citlali executable."""
        return self._version

    @cached_property
    def semver(self):
        """The semantic version of the Citlali executable."""
        return self._ver_to_semver(self._version)

    @classmethod
    def get_version(cls, path):
        """Get the version of the Citlali."""
        output = subprocess.check_output(
                (path, '--version'),
                stderr=subprocess.STDOUT,
                ).decode()
        cls.logger.debug(f'check version of {path}:\n{output}')
        r = re.compile(
            r'^(?P<name>citlali\s)?(?P<version>.+)\s\((?P<timestamp>.+)\)$',
            re.MULTILINE)
        m0 = re.search(r, output)
        # import pdb
        # pdb.set_trace()
        if m0 is None:
            cls.logger.warning(
                f"unable to parse citlali version: \n{output}")
            version = UNKNOWN_VERSION
        else:
            m = m0.groupdict()
            version = m['version']
        return cls._norm_ver(version, with_rev=False)

    def get_default_config(self):
        """Get the default config of the Citlali."""
        path = self.path
        output = subprocess.check_output(
                (path, '--dump_config'),
                stderr=subprocess.STDOUT,
                ).decode()
        self.logger.debug(f'dump config of {path}:\n{output}')
        return yaml_load(output)

    @staticmethod
    def _norm_ver(ver, with_rev=True):
        # removes any non-standard suffix
        if with_rev:
            return re.sub(r'(-dirty)$', '', ver)
        return re.sub(r'(-dirty|-[0-9a-z]+|-[0-9a-z]+-.+)$', '', ver)

    @classmethod
    def _ver_to_semver(cls, ver):
        """Convert version string (tag, rev hash, etc.) to SemVer.

        This is done by querying the local Citlali repo history.
        """
        try:
            return Version(ver)
        except InvalidVersion:
            pass
        # try find the latest version tag in the history
        repo = _get_local_citlali_repo()
        try:
            _ver = cls._norm_ver(
                repo.git.describe(ver, contains=True), with_rev=False)
        except Exception:
            _ver = cls._norm_ver(
                repo.git.describe(ver, contains=False), with_rev=False)
        cls.logger.debug(f"version {ver} -> semver {_ver}")
        return Version(_ver)

    def check_version(self, version):
        verspec = SpecifierSet(version)
        self.logger.debug(f"check {self.semver} against {verspec}")
        return self.semver in verspec

    def check_for_update(self):
        """Check the current Citlali version agains the Github remote head.

        """
        logger = get_logger()
        repo = _get_local_citlali_repo()

        def get_git_changelog(v1, v2):

            changelog = repo.git.log(f'{v1}..{v2}', oneline=True)
            if changelog == '':
                # same revision
                changelog = None
            return changelog

        # get remote head version
        remote_rev = repo.git.rev_parse('origin/HEAD', short=True)
        changelog = get_git_changelog(self.version, remote_rev)

        if changelog is not None:
            changelog_section = (
                f"\n"
                f"###################### Attention #######################\n"
                f"You are using an outdated version of Citlali. The latest\n"
                f"version has the following changes:\n\n"
                f"{changelog}\n"
                f"########################################################"
                f"\n"
                )
        else:
            changelog_section = 'Citlali is update-to-date!'
        logger.info(
            f"\n\n"
            f"* Executable path: {self.path}"
            f"* Remote Citlali rev: {remote_rev}\n"
            f"* Local Citlali version: {self.version}\n"
            f'{changelog_section}'
            )

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def _get_line_buf_cmd():
        stdbuf = shutil.which('stdbuf')
        if stdbuf is not None:
            return [stdbuf, '-oL']
        return list()

    def run(self, config_file, log_level="INFO", **kwargs):
        exec_path = self.path
        citlali_cmd = [
                exec_path.as_posix(),
                '-l', log_level.lower(),
                config_file.as_posix(),
                ]
        cmd = self._get_line_buf_cmd() + citlali_cmd
        self.logger.info(
            "run {} cmd: {}".format(self, ' '.join(citlali_cmd)))
        return call_subprocess_with_live_output(cmd, **kwargs)


@functools.lru_cache(maxsize=None)
def _get_citlali_exec(path):
    # This caches all the found exec.
    return CitlaliExec(path=path)


class Citlali(PipelineEngine):
    """A wrapper class of Citlali, the TolTEC data reduction pipeline
    engine.

    It searches for instances of ``citlali`` executables and check their
    versions against the required version. The latest one is adopted if
    multiple are found.

    Parameters
    ----------
    path : str, `pathlib.Path`, list
        The path to search the executable for. A list of paths is accepted
        which searches the executable in the list in order.

    version : str
        A version specifier/predicate that specifies the required version of
        pipeline.

    use_env_path : bool
        If True, the system PATH env var is consulted.
    """

    logger = get_logger()

    def __init__(
            self, path=None, version=None, use_env_path=True):
        citlali_executables = self.find_citlali_executables(
            path=path, version=version, use_env_path=use_env_path)
        if len(citlali_executables) == 0:
            raise PipelineEngineError(
                f"Cannot find Citlali executables for "
                f"version {version}")
        elif len(citlali_executables) > 1:
            self.logger.warning(
                f"Found multiple Citlali executables for "
                f"version={version}\n"
                # f"{citlali_executables['path', 'version', 'timestamp']}"
                )
        else:
            pass
        citlali_exec = self._citlali_exec = citlali_executables[0]
        self.logger.debug(f"use citlali executable: {citlali_exec}")

    def __repr__(self):
        return f'{self.__class__.__name__}(version={self.version})'

    @property
    def exec_path(self):
        return self._citlali_exec.path

    @property
    def version(self):
        return self._citlali_exec.version

    def get_default_config(self):
        return self._citlali_exec.get_default_config()

    def check_for_update(self):
        return self._citlali_exec.check_for_update()

    def run(self, *args, **kwargs):
        return self._citlali_exec.run(*args, **kwargs)

    @classmethod
    def find_citlali_executables(
            cls, path=None, version=None, use_env_path=True):
        """Return a list of `CitlaliExec` objects that satisfy the version
        constraints."""
        exec_name = 'citlali'

        if path is None:
            path = []
        elif isinstance(path, (str, pathlib.Path)):
            path = [path]
        else:
            pass
        if use_env_path:
            # check if the executable is in env path
            exec_path = shutil.which(exec_name, path=None)
            if exec_path is not None:
                path.append(exec_path)

        # now go over each item in path and get executable for checking
        paths = dict()

        def _check_and_add_path(exec_path):
            try:
                exec = CitlaliExec(path=exec_path)
                paths[exec_path] = exec
            except Exception:
                cls.logger.debug(
                    f"skip invalid cialali exec {exec_path}")

        cls.logger.debug(
            f"find citali executables in paths:\n{pformat_yaml(path)}")
        for p in path:
            p = ensure_abspath(p)
            if p.is_dir():
                _check_and_add_path(shutil.which('citlali', path=p))
            elif p.is_file() and os.access(p, os.X_OK):
                _check_and_add_path(p)
            else:
                cls.logger.debug(f'skip invalid path {p}')
        cls.logger.debug(
            f"found {len(paths)} citlali executables:\n{pformat_yaml(paths)}")
        # now check each executable with version
        if version is None:
            cls.logger.debug("skip checking executable versions")
            return list(paths.values())
        valid_execs = list()
        cls.logger.debug("check executable versions")
        for e in paths.values():
            if e.check_version(version):
                valid_execs.append(e)
                cls.logger.debug(f"{e} version satisfies {version}")
            else:
                cls.logger.debug(f"{e} version does not satisfies {version}")
        return valid_execs

    @timeit
    def proc_context(self, config):
        """Return a `CitlaliProc` that run reduction for given input dataset.
        """
        return CitlaliProc(citlali=self, config=config)


class CitlaliProc(object):
    """A context class for running Citlali."""

    logger = get_logger()

    def __init__(self, citlali, config):
        self._citlali = citlali
        self._config = replace(
            config,
            low_level=self._resolve_low_level_config(config.low_level)
            )

    @property
    def citlali(self):
        return self._citlali

    @property
    def config(self):
        return self._config

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def __call__(
            self, dataset, output_dir,
            log_level='INFO', logger_func=None):
        # resolve the dataset to input items
        tbl = dataset.index_table
        grouped = tbl.group_by(
            ['obsnum', 'subobsnum', 'scannum', 'master', 'repeat'])
        self.logger.debug(f"collected {len(grouped)} raw obs")
        for key, group in zip(grouped.groups.keys, grouped.groups):
            self.logger.debug(
                    '****** obs name={obsnum}_{subobsnum}_{scannum} '
                    '*******'.format(**key))
            self.logger.debug(f'{group}\n')
        input_items = [
            self._resolve_input_item(d, output_dir, cal_items=self.config.cal_items) for d in grouped.groups]
        # create low level config object and dump to file
        # the low level has been resolved to a dict when __init__ is called
        cfg = deepcopy(self.config.low_level)
        rupdate(cfg, {
            'inputs': input_items,
            'runtime': {
                # TODO get rid of the trailing slash
                'output_dir': output_dir.as_posix() + '/'
                }
            })
        cfg_hl = self._resolve_high_level_config(self.config)
        rupdate(cfg, cfg_hl)
        self.logger.debug(
                f'resolved low level config:\n{pformat_yaml(cfg)}')
        # show the high level config entries that over writes the low level
        # values
        chl = Cut(cfg_hl)
        cll = Cut(self.config.low_level)
        updated_entries = []
        for key in get_nested_keys(cfg_hl):
            if key not in cll:
                continue
            new = chl[key]
            old = cll[key]
            if new == old:
                continue
            updated_entries.append((key, old, new))
        if updated_entries:
            updated_entries = Table(
                rows=updated_entries,
                names=['low_level_config_key', 'default', 'updated'])
            self.logger.info(
                f"low level config entries overwitten by "
                f"high level config:\n\n"
                f"{updated_entries}\n")
        name = input_items[0]['meta']['name']
        output_name = f'citlali_o{name}_c{len(input_items)}.yaml'
        cfg_filepath = output_dir.joinpath(output_name)
        with open(cfg_filepath, 'w') as fo:
            fo.write(pformat_yaml(cfg))
            # yaml_dump(cfg, fo)
        success = self._citlali.run(
            cfg_filepath, log_level=log_level, logger_func=logger_func)
        # TODO implement the logic to locate the generated output files
        # which will be used to create data prod object.
        if success:
            # return output_dir
            return ToltecDataProd.collect_from_dir(output_dir).select(
                'id == id.max()')
        raise RuntimeError(
            f"failed to run {self.citlali} with config file {cfg_filepath}")

    def _resolve_low_level_config(self, low_level):
        """Return a low-level config dict from low_level config entry."""
        if low_level is None:
            return self.citlali.get_default_config()
        if isinstance(low_level, Path):
            with open(low_level, 'r') as fo:
                return yaml_load(fo)
        # a dict already
        return low_level

    def _resolve_high_level_config(self, high_level):
        """Return a low level config dict from high level dict."""
        # image_frame_params
        cfg = Cut(dict())
        pixel_size = high_level.image_frame_params.pixel_size
        if pixel_size is not None:
            cfg.setdefault(
                'mapmaking.pixel_size_arcsec', pixel_size.to_value(u.arcsec))
        return cfg.data

    @classmethod
    def _resolve_cal_items(cls, index_table, cal_items):
        """Build list of cal_items that are applicable to index_table
        """
        df = index_table[[c for c in index_table.colnames if len(index_table[c].shape) == 1]].to_pandas()
        if not cal_items:
            # a list of empty lists
            return [dict() for _ in range(len(df))]
        check = np.zeros((len(df), len(cal_items)), dtype=bool)
        for j, cal_item in enumerate(cal_items):
            # check the "select" against the index_table
            cond = cal_item.get("select", None)
            if cond is not None:
                is_applicable = df.eval(cond).to_numpy(dtype=bool)
            else:
                is_applicable = np.ones((len(df), ), dtype=bool)
            check[:, j] = is_applicable
        # build the list for each entry, note that we run rupdate
        # to merge the same type of cal_items
        result = list()
        for i in range(len(df)):
            resolved_cal_items = dict()
            for j, cal_item in enumerate(cal_items):
                cal_item = cal_item.copy()
                cal_item.pop("select", None)
                cal_item_type = cal_item['type']
                if check[i, j]:
                    if cal_item_type not in resolved_cal_items:
                        resolved_cal_items[cal_item_type] = dict()
                    rupdate(resolved_cal_items[cal_item_type], cal_item)
            # convert to list
            result.append(resolved_cal_items)
        return result

    @classmethod
    def _resolve_input_item(cls, index_table, output_dir, cal_items=None):
        """Return an citlali input list entry from index table."""
        tbl = index_table
        d0 = tbl[0]
        meta = {
            'name': f'{d0["obsnum"]}_{d0["subobsnum"]}_{d0["scannum"]}'
            }
        data_items = list()
        # note there we just build the calitem for the first entry
        # since all entries are the same
        cal_items = cls._resolve_cal_items(
            index_table[:1], cal_items.copy() or list())[0]
        for entry in tbl:
            instru = entry['instru']
            interface = entry['interface']
            source = entry['source']
            extra = dict()
            if instru == 'toltec':
                c = data_items
            elif interface == 'lmt':
                c = data_items
                source = _fix_tel(source, output_dir)
            elif interface == 'hwp':
                c = data_items
            elif interface == 'apt':
                c = cal_items
                # TODO implement in citlali the proper
                # ecsv handling
                source = _fix_apt(source, output_dir)
                extra = {'type': 'array_prop_table'}
            else:
                continue
            item = dict({
                    'filepath': source,
                    'meta': {
                        'interface': interface
                        }
                    }, **extra)
            if isinstance(c, list):
                c.append(item)
            elif isinstance(c, dict):
                c[item['type']] = item
            else:
                raise ValueError("invalid item list to construct")
        if 'array_prop_table' not in cal_items:
            # build the apt with all network tones
            cal_items['array_prop_table'] = {
                'filepath': _make_apt(data_items, output_dir),
                'meta': {
                    'interface': 'apt'
                    },
                'type': 'array_prop_table'
                }
        cls.logger.debug(
                f"collected input item name={meta['name']} "
                f"n_data_items={len(data_items)} "
                f"n_cal_items={len(cal_items)}")
        # this is a hack. TODO fix the proper ordering of data items
        data_items = sorted(
                data_items,
                key=lambda d: (
                    int(d['meta']['interface'][6:])
                    if d['meta']['interface'].startswith('toltec')
                    else -1)
                )
        return {
                'meta': meta,
                'data_items': data_items,
                # make cal_items a list as expected by the citlali
                'cal_items': list(cal_items.values()),
                }

def _fix_tel(source, output_dir):
    # This is to recompute the ParAngAct, SourceRaAct and SourceDecAct from the tel.nc file
    logger = get_logger()
    from netCDF4 import Dataset
    from astropy.time import Time
    from astropy.coordinates import SkyCoord
    from tollan.utils.fmt import pformat_yaml
    from tolteca.simu.toltec.toltec_info import toltec_info
    source_new = output_dir.joinpath(Path(source).name.replace('.nc', '_recomputed.nc')).as_posix()
    if source_new != source:
        try:
            shutil.copy(source, source_new)
        except Exception:
            raise ValueError("unable to create recomputed tel.nc")
    else:
        raise ValueError("invalid tel.nc filename")

    observer = toltec_info['site']['observer']
    tnc = Dataset(source_new, mode='a')
    tel_time = Time(tnc['Data.TelescopeBackend.TelTime'][:], format='unix')
    tel_t0 = tel_time[0]
    tel_t = tel_time - tel_t0

    tel_az = tnc['Data.TelescopeBackend.TelAzAct'][:] << u.rad
    tel_alt = tnc['Data.TelescopeBackend.TelElAct'][:] << u.rad
    tel_az_cor = tnc['Data.TelescopeBackend.TelAzCor'][:] << u.rad
    tel_alt_cor = tnc['Data.TelescopeBackend.TelElCor'][:] << u.rad
    tel_az_tot = tel_az - (tel_az_cor) / np.cos(tel_alt)
    tel_alt_tot = tel_alt - (tel_alt_cor)
    altaz_frame = observer.altaz(time=tel_time)
    tel_icrs_astropy = SkyCoord(tel_az_tot, tel_alt_tot, frame=altaz_frame).transform_to('icrs')
    # update variables and save
    pa = observer.parallactic_angle(time=tel_time, target=tel_icrs_astropy)
    pa_orig = tnc['Data.TelescopeBackend.ActParAng'][:] << u.rad
    ra_orig = tnc['Data.TelescopeBackend.SourceRaAct'][:] << u.rad
    dec_orig = tnc['Data.TelescopeBackend.SourceDecAct'][:] << u.rad

    tnc['Data.TelescopeBackend.ActParAng'][:] = pa.radian
    tnc['Data.TelescopeBackend.SourceRaAct'][:] = tel_icrs_astropy.ra.radian
    tnc['Data.TelescopeBackend.SourceDecAct'][:] = tel_icrs_astropy.dec.radian
    tnc.sync()
    tnc.close()
    # make some diagnostic info
    def stat_change(d, d_orig, unit, name):
        dd = (d - d_orig).to_value(unit)
        logger.info(f"{name} changed with diff ({unit}): min={dd.max()} max={dd.min()} mean={dd.mean()} std={np.std(dd)}")
    stat_change(pa, pa_orig, u.deg, 'ActParAng') 
    stat_change(tel_icrs_astropy.ra, ra_orig, u.arcsec, 'SourceRaAct') 
    stat_change(tel_icrs_astropy.dec, dec_orig, u.arcsec, 'SourceDecAct') 
    return source_new 


def _fix_apt(source, output_dir):
    # this is a temporary fix to make citlali work with the
    # apt
    tbl = Table.read(source, format='ascii.ecsv')
    if all(tbl[c].dtype == float for c in tbl.colnames):
        # by-pass it when it is already all float
        return source
    tbl_new = Table()

    def get_uid(uid):
        return int('1' + str(uid).replace("_", '').replace("-", ""))

    # tbl_new['uid'] = np.array([get_uid(uid) for uid in tbl['uid']], dtype='d')
    tbl_new['uid'] = np.arange(len(tbl), dtype='d')
    tbl_new['nw'] = np.array(tbl['nw'], dtype='d')
    tbl_new['fg'] = np.array(tbl['fg'], dtype='d')
    tbl_new['pg'] = np.array(tbl['pg'], dtype='d')
    tbl_new['ori'] = np.array(tbl['ori'], dtype='d')
    tbl_new['array'] = np.array(tbl['array'], dtype='d')
    tbl_new['flxscale'] = np.array(tbl['flxscale'], dtype='d')
    tbl_new['x_t'] = tbl['x_t'].quantity.to_value(u.arcsec)
    tbl_new['x_t_err'] = 0.
    tbl_new['y_t'] = tbl['y_t'].quantity.to_value(u.arcsec)
    tbl_new['y_t_err'] = 0.
    tbl_new['pa_t'] = tbl['pa_t'].quantity.to_value(u.radian)
    tbl_new['pa_t_err'] = 0.
    tbl_new['a_fwhm'] = tbl['a_fwhm'].quantity.to_value(u.arcsec)
    tbl_new['a_fwhm_err'] = 0.
    tbl_new['b_fwhm'] = tbl['b_fwhm'].quantity.to_value(u.arcsec)
    tbl_new['b_fwhm_err'] = 0.
    tbl_new['angle'] = 0.
    tbl_new['angle_err'] = 0.
    tbl_new['amp'] = 1.
    tbl_new['amp_err'] = 0.
    tbl_new['responsivity'] = tbl['responsivity'].quantity.to_value(u.pW ** -1)
    tbl_new['flag'] = 1.
    tbl_new['sens'] = 1.
    tbl_new['sig2noise'] = 1.
    tbl_new['converge_iter'] = 0.
    tbl_new['derot_elev'] = 0.
    tbl_new['loc'] = -1.
    if 'loc' in tbl.colnames:
        tbl_new['loc'] = np.array(tbl['loc'], dtype='d')
    for c in ["tone_freq", "x_t_raw", "y_t_raw", "x_t_derot", "y_t_derot"]:
        if c not in tbl.colnames:
            tbl_new[c] = 0.
        else:
            tbl_new[c] = np.array(tbl[c], dtype='d')

    source_new = output_dir.joinpath(Path(source).name.replace('.ecsv', '_trimmed.ecsv')).as_posix()
    tbl_new.write(source_new, format='ascii.ecsv', overwrite=True)
    return source_new


def _make_apt(data_items, output_dir):

    data_items_by_interface = {
            data_item['meta']['interface']: data_item
            for data_item in data_items
            }

    def _make_nw_apt(data_item):
        interface = data_item['meta']['interface']
        if not interface.startswith('toltec'):
            return
        nw = toltec_info[interface]['nw']
        array_name = toltec_info[interface]['array_name']
        filepath = data_item['filepath']
        bod = BasicObsData(source=filepath).open()
        kids_model = bod.get_model_params_table()
        tbl = Table(kids_model)
        # prefix all columns with kids_model_
        for c in tbl.colnames:
            tbl.rename_column(c, f'kids_model_{c}')
        tbl['uid'] = -1.
        tbl['nw'] = float(nw)
        tbl['fg'] = -1.
        tbl['pg'] = -1.
        tbl['ori'] = -1.
        tbl['loc'] = -1.
        tbl['array'] = float(toltec_info[array_name]['index'])
        tbl['flxscale'] = 1.
        tbl['x_t'] = 0.
        tbl['y_t'] = 0.
        tbl['pa_t'] = 0.
        tbl['a_fwhm'] = 0.
        tbl['b_fwhm'] = 0.
        return tbl

    tbl = list()
    for data_item in data_items:
        t = _make_nw_apt(data_item)
        if t is None:
            continue
        tbl.append(t)
    tbl = sorted(
            tbl,
            key=lambda t: t[0]['nw']
            )
    tbl = vstack(tbl)
    outname = Path(data_items_by_interface['toltec0']['filepath'].replace(data_items_by_interface['toltec0']['meta']['interface'], 'apt').replace('.nc', '.ecsv')).name
    outname = output_dir.joinpath(outname)
    tbl.write(outname, format='ascii.ecsv', overwrite=True)
    return outname


# High level config classes
# TODO some of these config can be made more generic and does not needs to
# be citlali specific.


@add_schema
@dataclass
class ImageFrameParams(object):
    """Params related to 2D image data shape and WCS."""
    pixel_size: u.Quantity = field(
        default=None,
        metadata={
            'description': 'The pixel size of image.',
            'schema': PhysicalTypeSchema('angle')
            }
        )


@add_schema
@dataclass
class CitlaliConfig(object):
    """The high-level config for Citlali."""

    low_level: Union[None, Path, dict] = field(
        default=None,
        metadata={
            'description': 'The low level config used as the base.',
            'schema': Or(RelPathSchema(), dict)
            }
        )
    image_frame_params: ImageFrameParams = field(
        default_factory=ImageFrameParams,
        metadata={
            'description':
            'The params related to the output image data shape and WCS.'
            }
        )
    cal_items: list = field(
        default_factory=list,
        metadata={
            'description': 'Additional cal items passed to input.'
            }
        )
    class Meta:
        schema = {
            'ignore_extra_keys': False,
            'description': 'The high level config for Citlali.'
            }
