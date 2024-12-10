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
from schema import Or, Optional, Schema
from dataclasses import dataclass, field, replace
from typing import Union
from tollan.utils.dataclass_schema import add_schema
from tollan.utils import rupdate
from tollan.utils import call_subprocess_with_live_output
from tollan.utils.log import get_logger, timeit
from tollan.utils import ensure_abspath
from tollan.utils.fmt import pformat_yaml
from tollan.utils.nc import ncstr

from .base import PipelineEngine, PipelineEngineError
from ...utils import get_user_data_dir
from ...utils.misc import get_nested_keys
from ...utils.common_schema import RelPathSchema, PhysicalTypeSchema
from ...utils.runtime_context import yaml_load
from ...datamodels.toltec.data_prod import ToltecDataProd
from ...datamodels.toltec.basic_obs_data import BasicObsData
from ...datamodels.io.toltec.tel import LmtTelFileIO
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
            log_level='INFO', logger_func=None, dry_run=False):
        cfg = self._prepare_citlali_config(dataset, output_dir)
        input_items = cfg['inputs']
        name = input_items[0]['meta']['name']
        output_name = f'citlali_o{name}_c{len(input_items)}.yaml'
        cfg_filepath = output_dir.joinpath(output_name)
        with open(cfg_filepath, 'w') as fo:
            fo.write(pformat_yaml(cfg))
            # yaml_dump(cfg, fo)
        if dry_run:
            logger_func(f"** DRY RUN **: citlali low level config: {cfg_filepath}")
            logger_func(f"dry run's done.")
            return None
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

    def _prepare_citlali_config(
            self, dataset, output_dir,
            ):
        
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
        def _has_data_item(d):
            return any(i.startswith('toltec') for i in d['interface'])
        input_items = [
            self._resolve_input_item(d, output_dir, cal_items_low_level=self.config.cal_items, cal_objs=self.config.cal_objs) for d in grouped.groups if _has_data_item(d)]
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
        return cfg

    def _resolve_low_level_config(self, low_level):
        """Return a low-level config dict from low_level config entry."""
        if low_level is None:
            if self.citlali is not None:
                return self.citlali.get_default_config()
            return {}
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
    def _resolve_pointing_offsets(cls, ppts, teldata):
        logger = get_logger()

        obsnum = teldata.nc_node.variables['Header.Dcs.ObsNum'][:].item()
        def _get_obsnum(ppt):
            return int(ppt.meta['obsnum'])
        # sort ppts by obsnum and get the closest ones to the current
        ppts = sorted(ppts, key=_get_obsnum)
        cal_obsnums = np.array(list(map(_get_obsnum, ppts)))
        logger.debug(f"resolve pointing offsets for {teldata} {obsnum=} using ppt of obsnums={cal_obsnums}")

        idx_pre = np.where(cal_obsnums <= obsnum)[0]
        if len(idx_pre) == 0:
            obsnum_pre = None
            ppt_pre = None
        else:
            obsnum_pre = cal_obsnums[idx_pre[-1]]
            ppt_pre = ppts[idx_pre[-1]]

        idx_post = np.where(cal_obsnums >= obsnum)[0]
        if len(idx_post) == 0:
            obsnum_post = None
            ppt_post = None
        else:
            obsnum_post = cal_obsnums[idx_post[0]]
            ppt_post = ppts[idx_post[0]]
        if obsnum_post == obsnum_pre:
            logger.debug(f"pre- and post- pointing resolve to the same obsnum, use pre only")
            obsnum_post = None
            ppt_post = None
        logger.debug(f"pointing offset for {obsnum=}: {obsnum_pre=} {obsnum_post=}")


        daz_tel_user = ((
                teldata.nc_node.variables['Header.PointModel.AzUserOff'][:].item()
                + teldata.nc_node.variables['Header.PointModel.AzPaddleOff'][:].item()
                ) << u.rad).to(u.arcsec)
        dalt_tel_user = ((
                teldata.nc_node.variables['Header.PointModel.ElUserOff'][:].item()
                + teldata.nc_node.variables['Header.PointModel.ElPaddleOff'][:].item()
                ) << u.rad).to(u.arcsec)

        logger.debug(f"tel {obsnum=} {daz_tel_user=!s} {dalt_tel_user=!s}")

        def _get_altaz_cor_arcsec(ppt, teldata):
            obsnum = _get_obsnum(ppt)
            daz_raw = np.mean(ppt['x_t'] << u.arcsec)
            dalt_raw = np.mean(ppt['y_t'] << u.arcsec)
            daz_user = ((
                    ppt.meta['Header.PointModel.AzUserOff']
                    + ppt.meta['Header.PointModel.AzPaddleOff']) << u.rad).to(u.arcsec)
            dalt_user = ((
                    ppt.meta['Header.PointModel.ElUserOff']
                    + ppt.meta['Header.PointModel.ElPaddleOff']) << u.rad).to(u.arcsec)
            logger.debug(f"ppt {obsnum=} {daz_raw=!s} {dalt_raw=!s} {daz_user=!s} {dalt_user=!s}")
            daz_adjust = daz_user - daz_tel_user
            dalt_adjust = dalt_user - dalt_tel_user
            logger.debug(f"adjust ppt offset {daz_adjust=!s} {dalt_adjust=!s}")
            # note here is a sign flip to go from measured offsets to
            # offset corrections.
            az_cor = -(daz_raw + daz_adjust)
            alt_cor = -(dalt_raw + dalt_adjust)
            logger.debug(f"pointing offset correction to use: {az_cor=!s} {alt_cor=!s}")
            mjd = ppt.meta["mjd"]
            return az_cor.to_value(u.arcsec), alt_cor.to_value(u.arcsec), mjd

        # compose the dict
        altaz_cor_arcsec = []
        if ppt_pre is not None:
            altaz_cor_arcsec.append(_get_altaz_cor_arcsec(ppt_pre, teldata))
        if ppt_post is not None:
            altaz_cor_arcsec.append(_get_altaz_cor_arcsec(ppt_post, teldata))
        az_cor_arcsec, alt_cor_arcsec, mjd = zip(*altaz_cor_arcsec)
        result = {
                'pointing_offsets': [
                    {'axes_name': 'az', 'value_arcsec': az_cor_arcsec},
                    {'axes_name': 'alt', 'value_arcsec': alt_cor_arcsec},
                    {'modified_julian_date': mjd},
                    ],
                'type': 'astrometry',
                'select': f'obsnum == {obsnum}'
                }
        logger.info(f"resolved pointing offset for {obsnum=} from {obsnum_pre=} {obsnum_post=}:\n{pformat_yaml(result)}")
        return [result]

    @classmethod
    def _resolve_cal_objs(cls, tel_index, cal_objs):
        if cal_objs is None:
            # resolve to a list of empty items
            return [dict() for _ in range(len(tel_index))]
        check = cls._check_select(
                tel_index,
                [cal_obj.select for cal_obj in cal_objs]
                )
        # load all cal_objs, this is a nested list
        # we need to keep this way to apply the select cond.
        caldata_objs_list = [cal_obj.load_data_objs() for cal_obj in cal_objs]
        # build the list for each entry, note that we run rupdate
        # to merge the same type of cal_items
        resolvers = {
                "ppt": cls._resolve_pointing_offsets
                }
        result = list()
        for i in range(len(tel_index)):
            teldata = LmtTelFileIO(tel_index[i]['source'])
            # this is a flat list of all applicable data objs
            caldata_objs_applicable = []
            for j, caldata_objs in enumerate(caldata_objs_list):
                if check[i, j]:
                    caldata_objs_applicable.extend(caldata_objs)
            for resolver_key, resolver_func in resolvers.items():
                # filter the applicable list with the resolver type
                caldata_items = [d['data'] for j, d in enumerate(caldata_objs_applicable) if d['type'] == resolver_key]
                if not caldata_items:
                    # no applicable caldata, skip
                    continue
                result.extend(resolver_func(caldata_items, teldata))
        return result

    @classmethod
    def _check_select(cls, index_table, conds):
        # build a n_entry x n_cond matrix recording the applicable status
        # of the select conds.
        df = index_table[[c for c in index_table.colnames if len(index_table[c].shape) == 1 and c != 'ut']].to_pandas()
        check = np.zeros((len(df), len(conds)), dtype=bool)
        for j, cond in enumerate(conds):
            # check the "select" against the index_table
            if cond is not None:
                is_applicable = df.eval(cond).to_numpy(dtype=bool)
            else:
                is_applicable = np.ones((len(df), ), dtype=bool)
            check[:, j] = is_applicable
        return check

    @classmethod
    def _resolve_cal_items(cls, tel_index, cal_items):
        """Build list of cal_items that are applicable to index_table
        """
        logger = get_logger()
        if cal_items is None:
            # resolve to a list of empty items
            return [dict() for _ in range(len(tel_index))]

        # logger.debug(f'resolve cal_items:\n{pformat_yaml(cal_items)}')
        check = cls._check_select(
                tel_index,
                [cal_item.get('select', None) for cal_item in cal_items]
                )
        # build the list for each entry, note that we run rupdate
        # to merge the same type of cal_items
        result = list()
        for i in range(len(tel_index)):
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
    def _resolve_input_item(cls, index_table, output_dir, cal_items_low_level=None, cal_objs=None):
        """Return an citlali input list entry from index table."""
        logger = get_logger()
        tbl = index_table
        d0 = tbl[0]
        meta = {
            'name': f'{d0["obsnum"]}_{d0["subobsnum"]}_{d0["scannum"]}'
            }
        # resolve data items first
        # any cal items resolved in data items will override
        # cal items
        data_items = list()
        cal_items_override = []
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
            elif interface == 'hwpr':
                c = data_items
            elif interface == 'apt':
                c = cal_items_override
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

        # resolve cal items
        # for this we need the tel file index
        tel_index = index_table[index_table['interface'] == 'lmt']
        cal_items =[]
        if cal_items_low_level is not None:
            cal_items.extend(cal_items_low_level)
        # note these cal_items get overriden by the resolved calobjs
        if cal_objs is not None:
            cal_items.extend(cls._resolve_cal_objs(tel_index, cal_objs))
        # don't forget the cal_items resolved from data_items
        cal_items.extend(cal_items_override)

        # finally for all of them combined
        # note this resovles to list of one dict
        cal_items = cls._resolve_cal_items(tel_index, cal_items)[0]
        logger.debug(f"all resolved cal_items:\n{pformat_yaml(cal_items)}")

        # validate the calitems.
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
    import netCDF4
    from netCDF4 import Dataset
    from astropy.time import Time
    from astropy.coordinates import SkyCoord
    from tollan.utils.fmt import pformat_yaml
    from tolteca.simu.toltec.toltec_info import toltec_info
    from tolteca.simu.toltec.models import pa_from_coords
    source_new = output_dir.joinpath(Path(source).name.replace('.nc', '_recomputed.nc')).as_posix()
    if Path(source_new).exists():
        return source_new
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
    tel_altaz = SkyCoord(tel_az_tot, tel_alt_tot, frame=altaz_frame)
    tel_icrs_astropy = tel_altaz.transform_to('icrs')

    # 20230328 It seems that there are some rotation in final maps.
    # we switch the pa from using astroplan observer to that used in the
    # simulator, directly computating with the RA Dec Az and Alt.
    # update variables and save
    # pa = observer.parallactic_angle(time=tel_time, target=tel_icrs_astropy)
    pa = pa_from_coords(
        observer=observer,
        coords_altaz=tel_altaz,
        coords_icrs=tel_icrs_astropy)

    pa_orig = tnc['Data.TelescopeBackend.ActParAng'][:] << u.rad
    ra_orig = tnc['Data.TelescopeBackend.SourceRaAct'][:] << u.rad
    dec_orig = tnc['Data.TelescopeBackend.SourceDecAct'][:] << u.rad

    tnc['Data.TelescopeBackend.ActParAng'][:] = pa.radian
    tnc['Data.TelescopeBackend.SourceRaAct'][:] = tel_icrs_astropy.ra.radian
    tnc['Data.TelescopeBackend.SourceDecAct'][:] = tel_icrs_astropy.dec.radian

    def _setstr(nc, k, s, dim=128):
        dim_name = k + "_len"
        nc.createDimension(dim_name, dim)
        v = nc.createVariable(k, 'S1', (dim_name, ))
        v[:] = netCDF4.stringtochar(np.array([s], dtype=f'S{dim}'))
        return v
    if 'Header.Dcs.ObsGoal' not in tnc.variables:
        _setstr(tnc, 'Header.Dcs.ObsGoal', 'Science')
    if ncstr(tnc.variables['Header.Dcs.ObsPgm']) == "Map" and "Header.Map.MapCoord" not in tnc.variables:
        _setstr(tnc, 'Header.Map.MapCoord', 'Az')
    if 'Header.Source.Epoch' not in tnc.variables:
        _setstr(tnc, 'Header.Source.Epoch', 'J2000')

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
    logger = get_logger()
    tbl = Table.read(source, format='ascii.ecsv')
    if all(tbl[c].dtype in [float, np.float32] for c in tbl.colnames):
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
    tbl_new['loc'] = np.array(tbl['loc'], dtype='d')
    tbl_new['array'] = np.array(tbl['array'], dtype='d')
    tbl_new['x_t'] = tbl['x_t'].quantity.to_value(u.arcsec)
    tbl_new['y_t'] = tbl['y_t'].quantity.to_value(u.arcsec)

    # these are items that does not show up in the some apt
    for c, unit, defval in [
            ("tone_freq", u.Hz, 0.),
            ('responsivity', u.pW  ** -1, 1.),
            ('flxscale', u.mJy / u.beam, 1.),
            ("x_t_err", u.arcsec, 0.),
            ("y_t_err", u.arcsec, 0.),
            ("x_t_raw", u.arcsec, 0.),
            ("y_t_raw", u.arcsec, 0.),
            ("x_t_derot", u.arcsec, 0.),
            ("y_t_derot", u.arcsec, 0.),
            ("pa_t", u.radian, 0.),
            ("pa_t_err", u.radian, 0.),
            ("a_fwhm", u.arcsec, 0.),
            ("a_fwhm_err", u.arcsec, 0.),
            ("b_fwhm", u.arcsec, 0.),
            ("b_fwhm_err", u.arcsec, 0.),
            ("angle", u.radian, 0.),
            ("angle_err", u.radian, 0.),
            ("amp", None, 1.),
            ("amp_err", None, 0.),
            ("sens", u.mJy * u.s ** (1/2), 1.),
            ("derot_elev", u.radian, 0.),
            ("sig2noise", None, 0.),
            ("converge_iter", None, 0.),
            ]:
        logger.debug(f"fix col {c=} {unit=} {defval=}")
        if c not in tbl.colnames:
            tbl_new[c] = defval
        else:
            if unit is not None and tbl[c].unit is not None:
                c_value = tbl[c].quantity.to_value(unit)
            elif unit is not None and tbl[c].unit is None:
                logger.debug(f"assume {unit=} for apt column {c}")
                c_value = tbl[c]
            else:
                c_value = tbl[c]
            tbl_new[c] = np.array(c_value, dtype='d')
        if unit is not None:
            tbl_new[c].unit = unit

    flag = tbl['flag']
    if hasattr(flag, 'filled'):
        flag = flag.filled(1.)
    tbl_new['flag'] = 1. * flag
    # add required meta
    tbl_new.meta['Radesys'] = 'altaz'
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
        tbl['responsivity'] = 1.
        tbl['sens'] = 1.
        tbl['derot_elev'] = 0.
        tbl['x_t'] = 0.
        tbl['x_t_err'] = 0.
        tbl['y_t'] = 0.
        tbl['y_t_err'] = 0.
        tbl['pa_t'] = 0.
        tbl['a_fwhm'] = 0.
        tbl['a_fwhm_err'] = 0.
        tbl['b_fwhm'] = 0.
        tbl['b_fwhm_err'] = 0.
        tbl['amp'] = 0.
        tbl['amp_err'] = 0.
        tbl['angle'] = 0.
        tbl['angle_err'] = 0.
        tbl['converge_iter'] = 1.
        tbl['flag'] = 0.
        tbl['sig2noise'] = 1.

        tbl['x_t_raw'] = 0.
        tbl['x_t_raw_err'] = 0.
        tbl['y_t_raw'] = 0.
        tbl['y_t_raw_err'] = 0.
        tbl['x_t_derot'] = 0.
        tbl['x_t_derot_err'] = 0.
        tbl['y_t_derot'] = 0.
        tbl['y_t_derot_err'] = 0.


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
class CalObj(object):
    """Calibration Object."""
    path: Path = field(
            metadata={
                "description": "Calobj path.",
                "schema": RelPathSchema(),
                })
    select: Union[None, str] = field(
        default=None,
        metadata={
            "desription": "The expression to select data this cal file is applicable to",
            })

    def load_data_objs(self):
        path = self.path
        if path.is_dir():
            # todo properly handle the loading of calobj from data prod folder
            cal_patterns = {
                    'ppt': '**/ppt_*.ecsv',
                    }
            caldata_objs = list()
            for caldata_obj_key, cal_pattern in cal_patterns.items():
                f = list(path.glob(cal_pattern))
                if len(f) != 1:
                    raise ValueError(f"invalid calobj path {path}")
                f = f[0]
                caldata_objs.append({
                        'data': Table.read(f, format='ascii.ecsv'),
                        'type': caldata_obj_key,
                        'select': self.select
                        })
            return caldata_objs
        raise NotImplementedError()


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
    cal_objs: list = field(
        default_factory=list,
        metadata={
            'description': 'Additional calibration objects passed to input.',
            'schema': Schema([CalObj.schema]),
            })
    class Meta:
        schema = {
            'ignore_extra_keys': False,
            'description': 'The high level config for Citlali.'
            }
