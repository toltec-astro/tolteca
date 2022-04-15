#!/usr/bin/env python


import numpy as np
from dataclasses import dataclass, field

from astropy.io import fits
import astropy.units as u

from tollan.utils.dataclass_schema import add_schema
from tollan.utils.log import get_logger, timeit
from tollan.utils.fmt import pformat_yaml

from ....simu.toltec.toltec_info import toltec_info
from ....datamodels.dp.base import DataProd, DataItemKind
from ....datamodels.toltec.data_prod import ScienceDataProd
from ... import steps_registry


def _make_minkasi_maps(tod_names, pixsize=1 << u.arcsec):

    from minkasi_wrapper import minkasi

    todvec = minkasi.TodVec()

    for fname in tod_names:
        with timeit('read tod from fits'):
            dat = minkasi.read_tod_from_fits(fname)
        with timeit("guess common mode"):
            # truncate_tod chops samples from the end to make
            # the length happy for ffts
            # minkasi.truncate_tod(dat)
            # sometimes we have faster sampled data than we need.
            # this fixes that.  You don't need to, though.
            # minkasi.downsample_tod(dat)
            # since our length changed, make sure we have a happy length
            # minkasi.truncate_tod(dat)

            # figure out a guess at common mode #and (assumed) linear detector
            # drifts/offset drifts/offsets are removed, which is important for
            # mode finding.  CM is *not* removed.
            dd, pred2, cm = minkasi.fit_cm_plus_poly(dat['dat_calib'], full_out=True)
            dat['dat_calib'] = dd
        tod = minkasi.Tod(dat)
        todvec.add_tod(tod)

    # make a template map with desired pixel size an limits that cover the data
    # todvec.lims() is MPI-aware and will return global limits, not just
    # the ones from private TODs
    lims = todvec.lims()
    pixsize_rad = pixsize.to_value(u.rad)
    map = minkasi.SkyMap(lims, pixsize_rad)

    # once we have a map, we can figure out the pixellization of the data.
    # Save that so we don't have to recompute.  Also calculate a noise model.
    # The one here (and currently the only supported one) is to rotate the data
    # into SVD space, then smooth the power spectrum of each mode.  Other
    # models would not be hard to implement.  The smoothing could do with a bit
    # of tuning as well.
    for tod in todvec.tods:
        ipix = map.get_pix(tod)
        tod.info['ipix'] = ipix
        tod.set_noise(minkasi.NoiseSmoothedSVD)
        # tod.set_noise(minkasi.NoiseCMWhite)
        # tod.set_noise_smoothed_svd()

    # get the hit count map.  We use this as a preconditioner
    # which helps small-scale convergence quite a bit.
    hits = minkasi.make_hits(todvec, map)
    hits_mapset = minkasi.Mapset()
    hits_mapset.add_map(hits)

    # setup the mapset. In general this can have many things
    # in addition to map(s) of the sky, but for now we'll just
    # use a single skymap.
    mapset = minkasi.Mapset()
    mapset.add_map(map)

    # make A^T N^1 d.  TODs need to understand what to do with maps but maps
    # don't necessarily need to understand what to do with TODs, hence putting
    # make_rhs in the vector of TODs. Again, make_rhs is MPI-aware, so this
    # should do the right thing if you run with many processes.
    rhs = mapset.copy()
    todvec.make_rhs(rhs)

    # this is our starting guess. Default to starting at 0,
    # but you could start with a better guess if you have one.
    x0 = rhs.copy()
    x0.clear()

    # preconditioner is 1/ hit count map.  helps a lot for
    # convergence.
    precon = mapset.copy()
    tmp = hits.map.copy()
    ii = tmp > 0
    tmp[ii] = 1.0/tmp[ii]
    # precon.maps[0].map[:]=numpy.sqrt(tmp)
    precon.maps[0].map[:] = tmp[:]

    # run PCG!
    mapset_out = minkasi.run_pcg(rhs, x0, todvec, precon, maxiter=30)
    return (mapset_out, hits_mapset)


class MinkasiExecutor():

    logger = get_logger()

    def __call__(self, dp, output_dir):

        self.logger.debug(
            f"run minkasi for dp={dp}, output_dir={output_dir}")
        ctod = dp[
            dp['kind'] == DataItemKind.CalibratedTimeOrderedData]
        if len(ctod) == 0:
            raise ValueError('data product has no calibrated tod item.')
        ctod = ctod.index_table[0]
        cache_dir = output_dir.joinpath('minkasi_cache')
        if not cache_dir.exists():
            cache_dir.mkdir()
        minkasi_tod_paths = self._convert_nc_to_fits(
            ctod['filepath'], cache_dir=cache_dir)
        data_items = list() 
        for i, (array_name, tod_path) in enumerate(minkasi_tod_paths.items()):
            image_name = (
                f"{ctod['filepath'].stem}_{array_name}_minkasi_map.fits")
            hits_image_name = (
                f"{ctod['filepath'].stem}_{array_name}_minkasi_map_hits.fits")
            image_path = output_dir.joinpath(image_name)
            hits_image_path = output_dir.joinpath(hits_image_name)
            mapset, hits_mapset = _make_minkasi_maps([tod_path])
            mapset.maps[0].write(image_path)
            hits_mapset.maps[0].write(hits_image_path)
            data_items.append({
                'array_name': array_name,
                'kind': DataItemKind.Image,
                'filepath': image_path
                })
        meta = dp.meta
        index = {
            'data_items': data_items,
            'meta': meta
            }
        return ScienceDataProd(source=index)

    @classmethod
    def _convert_nc_to_fits(
            cls,
            ctod_path,
            cache_dir):
        # here we check if in the cache if it is already converted.
        # if so, return the list of files
        minkasi_tod_paths = dict()
        cache_valid = True
        for array_name in toltec_info['array_names']:
            cache_filename = f'{ctod_path.stem}_{array_name}_minkasi_tod.fits'
            p = minkasi_tod_paths[array_name] = cache_dir.joinpath(
                cache_filename)
            if not p.exists():
                cache_valid = False

        if cache_valid:
            return minkasi_tod_paths
        # do the convert if cache is missing
        from netCDF4 import Dataset

        cls.logger.debug(f"load calibrated tod in netCDF format: {ctod_path}")
        nc_node = Dataset(ctod_path)
        nc_vars = dict(
            data=nc_node['DATA'],
            dx=nc_node['DX'],
            dy=nc_node['DY'],
            elev=nc_node['ELEV'],
            flag=nc_node['FLAG'],
            pixid=nc_node['PIXID'],
            time=nc_node['TIME'],
            array=nc_node['ARRAYID']
            )
        cls.logger.debug(f"data shape: {np.shape(nc_vars['data'])}")
        t_unix = nc_vars['time']
        cls.logger.debug(f"t_exp={t_unix[-1] - t_unix[0]} s")

        for array_name in toltec_info['array_names']:
            array_index = toltec_info[array_name]['index']
            array = nc_vars['array'][:]
            m = (array == array_index)
            n_dets = m.sum()
            if n_dets == 0:
                cls.logger.debug(f"no data for array_name={array_name}")
                continue
            # load the data
            tod_data = {
                    "data": nc_vars['data'][:, m],
                    "dx": nc_vars['dx'][:, m],
                    "dy": nc_vars['dy'][:, m],
                    'elev': nc_vars['elev'][:],
                    'pixid': nc_vars['pixid'][m],
                    'time': nc_vars['time'][:],
                    }
            minkasi_tod_path = minkasi_tod_paths[array_name]
            cls.logger.debug(
                f'make minkasi tod file for array_name={array_name} '
                f'n_detectors={n_dets}')
            hdulist = cls._make_minkasi_tod_fits(tod_data=tod_data)
            hdulist.writeto(minkasi_tod_path, overwrite=True)
        cls.logger.debug(
            f"finished making minkasi tod fits: "
            f"{pformat_yaml(minkasi_tod_paths)}")
        return minkasi_tod_paths

    @classmethod
    def _make_minkasi_tod_fits(cls, tod_data):
        data = tod_data['data']
        dx = tod_data['dx']
        dy = tod_data['dy']
        cls.logger.info(f"make minkasi map with tod shape={data.shape}")

        n_smps, n_dets = data.shape
        # expand the 1d variables to 2d
        pixid = np.outer(np.ones(n_smps), tod_data['pixid'])
        elev = np.outer(tod_data['elev'], np.ones(n_dets))
        time = np.outer(tod_data['time'], np.ones(n_dets))

        data_f = data.flatten(order='F')
        dx_f = dx.flatten(order='F')
        dy_f = dy.flatten(order='F')
        pixid_f = pixid.flatten(order='F')
        elev_f = elev.flatten(order='F')
        time_f = time.flatten(order='F')

        col_pixid = fits.Column(
            name='PIXID', format='D', array=pixid_f)
        col_dx = fits.Column(
            name='DX', format='D', array=dx_f)
        col_dy = fits.Column(
            name='DY', format='D', array=dy_f)
        col_elev = fits.Column(
            name='ELEV', format='D', array=elev_f)
        col_time = fits.Column(
            name='TIME', format='D', array=time_f)
        col_fnu = fits.Column(
            name='FNU', format='D', array=data_f)
        col_ufnu = fits.Column(
            name='UFNU', format='D', array=np.zeros(len(data_f)))
        hdu = fits.BinTableHDU.from_columns([
            col_pixid, col_dx,
            col_dy,
            col_elev,
            col_time,
            col_fnu,
            col_ufnu])
        return hdu


@steps_registry.register('minkasi')
@add_schema
@dataclass
class MinkasiStepConfig():
    '''
    The config class for the minkasi map maker.
    '''

    enabled: bool = field(
        default=True,
        metadata={
            'description': 'Enable/disable this pipeline step.'
            }
        )

    def __post_init__(self):
        self.logger = get_logger()

    def __call__(self, cfg):
        return MinkasiExecutor()

    def run(self, cfg, inputs=None):
        """Run this reduction step."""
        if inputs is None:
            inputs = cfg.load_input_data()
        dps = [
            input for input in inputs
            if (
                isinstance(input, DataProd)
                and (input.data_item_kinds
                     & DataItemKind.CalibratedTimeOrderedData))
            ]
        if len(dps) == 0:
            self.logger.debug("no valid input for this step, skip")
            return None

        assert len(dps) == 1
        dp = dps[0]
        output_dir = cfg.get_or_create_output_dir()
        minkasi_executor = self(cfg)
        return minkasi_executor(
            dp=dp,
            output_dir=output_dir,
            )
