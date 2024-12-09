
import numpy as np
import datetime
import astropy.units as u
import astropy.constants as const

from tollan.utils.log import timeit, get_logger
import toast

from .models import _get_default_passbands
from ...common import lmt_info

# load the atmosphere tools
try:
    from toast.atm import (
        atm_absorption_coefficient_vec, 
        atm_atmospheric_loading_vec
    )
    have_atm_utils = True
except ImportError as err:
    have_atm_utils = False
    raise err

__all__ = ['ToastAtmosphereSimulation']

class ToastAtmosphereSimulation(object):
    """ toast Atmosphere Slabs 
        TODO: kwargs the inputs to toast.atm.AtmSim
    """
    def __init__(self, t0, tmin, tmax, azmin, azmax, elmin, elmax, cachedir=None):
        self.t0    = t0
        self.tmin  = tmin
        self.tmax  = tmax
        self.azmin = azmin
        self.azmax = azmax
        self.elmin = elmin
        self.elmax = elmax
        self.cachedir = cachedir

        self.site_height = u.Quantity(lmt_info['location']['height'])

        self.logger = get_logger()

    @timeit
    def generate_simulation(self, setup_params_dict):
        """ Generates with parameters
        """
        self.atm_slabs = self._generate_toast_atm_slabs(
            self.t0, self.tmin, self.tmax, 
            self.azmin, self.azmax, 
            self.elmin, self.elmax, **setup_params_dict
        )
        self._bandpass_calculations()

    @staticmethod
    def spectrum_convolution(freqs, spectrum, throughput):
        """ Convolve the provided spectrum with the detector bandpass
        Args:
            freqs(array of floats):  Spectral bin locations
            spectrum(array of floats):  Spectral bin values
            throughput(array of floats): throughput of the bandpass
        Returns:
            (array):  The bandpass-convolved spectrum
        """
        # interpolation not needed since we use the same array
        freqs = freqs.to_value(u.GHz)

        # norm the bandpass
        norm = toast._libtoast.integrate_simpson(freqs, throughput)
        throughput /= norm

        # convolved the data
        convolved = toast._libtoast.integrate_simpson(
            freqs, spectrum * throughput
        )
        return convolved

    @timeit
    def _absorption_coefficient(self, bandpass):
        absorption = atm_absorption_coefficient_vec(
            self.site_height.to_value(u.meter),
            self.sim_weather.air_temperature.to_value(u.Kelvin),
            self.sim_weather.surface_pressure.to_value(u.Pa),
            self.sim_weather.pwv.to_value(u.mm),
            bandpass[0].to_value(u.GHz),
            bandpass[-1].to_value(u.GHz),
            len(bandpass),
        )    
        return absorption

    @timeit
    def _atmospheric_loading(self, bandpass):
        loading = atm_atmospheric_loading_vec(
            self.site_height.to_value(u.meter),
            self.sim_weather.air_temperature.to_value(u.Kelvin),
            self.sim_weather.surface_pressure.to_value(u.Pa),
            self.sim_weather.pwv.to_value(u.mm),
            bandpass[0].to_value(u.GHz),
            bandpass[-1].to_value(u.GHz),
            len(bandpass),
        )
        return loading

    @timeit
    def _bandpass_calculations(self):
        # TODO: make this faster
        
        self.absorption = dict()
        self.loading = dict()

        pb = _get_default_passbands()
        for band in ['a1100', 'a1400', 'a2000']:

            # get the bandpass/throughput
            bandpass_freqs = np.array(pb[band]['f'][:]) * u.GHz
            bandpass_throughput = np.array(pb[band]['throughput'])
            
            # calculate the absorption/loading 
            # (around 15s each) (toast function so requires a bit more investigation )
            absorption = self._absorption_coefficient(bandpass_freqs)
            loading= self._atmospheric_loading(bandpass_freqs)

            # calculate the convolution
            absorption_det = self.spectrum_convolution(bandpass_freqs, absorption, bandpass_throughput)
            loading_det = self.spectrum_convolution(bandpass_freqs, loading, bandpass_throughput)

            # store it for later use
            self.absorption[band] = absorption_det
            self.loading[band]= loading_det

    def _generate_toast_atm_slabs(
        self, 
        t0, 
        tmin, 
        tmax, 
        azmin, 
        azmax, 
        elmin, 
        elmax, 
        lmin_center,
        lmin_sigma,
        lmax_center,
        lmax_sigma, 
        z0_center, 
        z0_sigma, 
        zatm, 
        zmax,
        w_sigma,
        wdir_sigma,
        T0_sigma,
        nelem_sim_max,
        median_weather,
        rmin,
        rmax,
        scale,
        xstep,
        ystep,
        zstep,
        key1,
        key2,
        mpi_comm=None
    ):
        """Creates the atmosphere models using multiple slabs
        """

        # sets toast to also produce debugging messages 
        # import toast.utils
        # toast_env = toast.utils.Environment.get()
        # toast_env.set_log_level('DEBUG')

        # Starting slab parameters (thank you Ted)
        # rmin  =  0 * u.meter
        # rmax  =  100 * u.meter
        # scale =  10.0
        # xstep =  5 * u.meter
        # ystep =  5 * u.meter
        # zstep =  5 * u.meter

        # RNG state
        # key1     = 0
        # key2     = 0
        counter1 = 0
        counter2 = 0

        # obtain the weather information
        self.sim_weather = toast.weather.SimWeather(
            time = t0.to_datetime(timezone=datetime.timezone.utc),
            name="LMT", median_weather=median_weather
        )
        self.T0_center   = self.sim_weather.air_temperature
        self.wx          = self.sim_weather.west_wind
        self.wy          = self.sim_weather.south_wind
        w_center    = np.sqrt(self.wx ** 2 + self.wy ** 2)
        wdir_center = np.arctan2(self.wy, self.wx)

        self.logger.debug(f'initial seed: {key1=} {key2=} {counter1=} {counter2=}')
        self.logger.debug(f'weather params (1): {self.sim_weather.air_temperature=} {self.sim_weather.south_wind=} {self.sim_weather.west_wind=}')
        self.logger.debug(f'weather params (2): {w_center=} {wdir_center=}')

        # dict of atmosphere slabs
        atm_slabs = dict()

        # generate slabs until rmax > rmax_threshold
        rmax_threshold = 100000 * u.meter
        while rmax < rmax_threshold:
            slab_id = f'{key1}_{key2}_{counter1}_{counter2}'
            self.logger.debug(f'slab_id: {slab_id}')
            toast_atmsim_model = toast.atm.AtmSim(
                azmin=azmin, azmax=azmax,
                elmin=elmin, elmax=elmax,
                tmin=tmin, tmax=tmax,
                lmin_center=lmin_center,
                lmin_sigma=lmin_sigma,
                lmax_center=lmax_center,
                lmax_sigma =lmax_sigma,  
                w_center=w_center,
                w_sigma=w_sigma, #0 * (u.km / u.second), # w_sigma
                wdir_center=wdir_center,
                wdir_sigma=wdir_sigma, #0 * u.radian, #
                z0_center=z0_center,  #,2000 * u.meter,
                z0_sigma=z0_sigma,  #0 * u.meter,
                T0_center=self.T0_center,
                T0_sigma=T0_sigma, #10 * u.Kelvin, # 
                zatm=zatm, 
                zmax=zmax, 
                xstep=xstep,
                ystep=ystep,
                zstep=zstep,
                nelem_sim_max=nelem_sim_max,
                comm=mpi_comm,
                key1=key1,
                key2=key2,
                counterval1=counter1,
                counterval2=counter2,
                cachedir=self.cachedir,
                rmin=rmin,
                rmax=rmax,
                node_comm=None,
                node_rank_comm=None
            )
            
            # simulate the atmosphere
            use_cache = False
            if self.cachedir is not None:
                use_cache = True
            err = toast_atmsim_model.simulate(use_cache=use_cache)
            if err != 0:
                raise RuntimeError("toast atmosphere generation failed")
            
            # include in stack
            atm_slabs[slab_id] = toast_atmsim_model
            
            # use a new RNG stream for each slab
            counter1 += 1

            # decrease resolution as we increase altitude
            rmin   = u.Quantity(rmax)
            rmax  *= scale
            xstep *= np.sqrt(scale)
            ystep *= np.sqrt(scale)
            zstep *= np.sqrt(scale)

        return atm_slabs
