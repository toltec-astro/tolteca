
import numpy as np
import datetime
import astropy.units as u
import astropy.constants as const

from tollan.utils.log import timeit, get_logger
import toast

from . import get_default_passbands
from .lmt import info as site_info

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
    def __init__(self, t0, tmin, tmax, azmin, azmax, elmin, elmax):
        self.t0    = t0
        self.tmin  = tmin
        self.tmax  = tmax
        self.azmin = azmin
        self.azmax = azmax
        self.elmin = elmin
        self.elmax = elmax

        self.site_height = u.Quantity(site_info['site']['location']['height'])

    @timeit
    def generate_simulation(self):
        """ Generates with parameters
        """
    
        self.atm_slabs = self._generate_toast_atm_slabs(
            self.t0, self.tmin, self.tmax, 
            self.azmin, self.azmax, 
            self.elmin, self.elmax
        )

        self._bandpass_calculations()

    @staticmethod
    def spectrum_convolution(freqs, spectrum, throughput, rj):
        """ Convolve the provided spectrum with the detector bandpass
        Args:
            freqs(array of floats):  Spectral bin locations
            spectrum(array of floats):  Spectral bin values
            throughput(array of floats): throughput of the bandpass
            rj(bool):  Input spectrum is in Rayleigh-Jeans units and
                should be converted into thermal units for convolution
        Returns:
            (array):  The bandpass-convolved spectrum
        """
        # interpolation not needed since we use the same array
        freqs = freqs.to_value(u.GHz)
        # spectrum_det = np.interp(freq_ghz, freqs.to_value(u.GHz), spectrum)

        # TODO: this is always true
        if rj: 
            # From brightness to thermodynamic units
            TCMB = 2.72548
            x = const.h.value * freqs * 1e9 / const.k_B.value / TCMB
            rj2cmb = (x / (np.exp(x / 2) - np.exp(-x / 2))) ** -2
            spectrum *= rj2cmb

        convolved = toast._libtoast.integrate_simpson(
            freqs, spectrum * throughput
        )

        # convert back to brightness units after so it will 
        # multiply correctly with the observe result.
        cmb_equiv = u.thermodynamic_temperature(np.mean(freqs) * u.GHz, (2.72548 * u.Kelvin))
        rj_equiv = u.brightness_temperature(np.mean(freqs) * u.GHz)

        return (convolved * u.mK).to(u.MJy / u.sr, equivalencies=cmb_equiv).to(u.K, equivalencies=rj_equiv).value

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
        
        self.absorption = dict()
        self.loading = dict()

        pb = get_default_passbands()
        for band in ['a1100', 'a1400', 'a2000']:

            # get the bandpass/throughput
            self.bandpass_freqs = np.array(pb[band]['f'][:]) * u.GHz
            self.bandpass_throughput = np.array(pb[band]['throughput'])

            # calculate the absorption/loading
            absorption = self._absorption_coefficient(self.bandpass_freqs)
            loading= self._atmospheric_loading(self.bandpass_freqs)

            # calculate the convolution
            absorption_det = self.spectrum_convolution(self.bandpass_freqs, absorption, self.bandpass_throughput, rj=True)
            loading_det = self.spectrum_convolution(self.bandpass_freqs, loading, self.bandpass_throughput, rj=True)

            # store it for later use
            self.absorption[band] = absorption_det
            self.loading[band]= loading_det

    def _generate_toast_atm_slabs(self, t0, tmin, tmax, azmin, azmax, elmin, elmax, mpi_comm=None):
        """Creates the atmosphere models using multiple slabs
        Currently, only the parameters that define the time ranges, azimuth ranges, 
        elevation ranges are exposed (by necessity)
        """

        # Starting slab parameters (thank you Ted)
        rmin  =  0 * u.meter
        rmax  =  100 * u.meter
        scale =  10.0
        xstep =  5 * u.meter
        ystep =  5 * u.meter
        zstep =  5 * u.meter

        # RNG state
        key1     = 0
        key2     = 0
        counter1 = 0
        counter2 = 0

        # obtain the weather information
        # TODO: separate out the weather to its own method
        self.sim_weather = toast.weather.SimWeather(
            time = t0.to_datetime(timezone=datetime.timezone.utc),
            name="LMT"
        )
        T0_center   = self.sim_weather.air_temperature
        wx          = self.sim_weather.west_wind
        wy          = self.sim_weather.south_wind
        w_center    = np.sqrt(wx ** 2 + wy ** 2)
        wdir_center = np.arctan2(wy, wx)

        # dict of atmosphere slabs
        atm_slabs = dict()

        # generate slabs until rmax > 100000 meters
        # TODO: eventually expose these
        while rmax < 100000 * u.meter:
            slab_id = f'{key1}_{key2}_{counter1}_{counter2}'
            toast_atmsim_model = toast.atm.AtmSim(
                azmin=azmin, azmax=azmax,
                elmin=elmin, elmax=elmax,
                tmin=tmin, tmax=tmax,
                lmin_center=0.01 * u.meter,
                lmin_sigma=0.001 * u.meter,
                lmax_center=10.0 * u.meter,
                lmax_sigma =10.0 * u.meter,
                w_center=w_center,
                w_sigma=0 * (u.km / u.second),
                wdir_center=wdir_center,
                wdir_sigma=0 * u.radian,
                z0_center=2000 * u.meter,
                z0_sigma=0 * u.meter,
                T0_center=T0_center,
                T0_sigma=0 * u.Kelvin,
                zatm=40000.0 * u.meter,
                zmax=2000.0 * u.meter,
                xstep=xstep,
                ystep=ystep,
                zstep=zstep,
                nelem_sim_max=10000,
                comm=mpi_comm,
                key1=key1,
                key2=key2,
                counterval1=counter1,
                counterval2=counter2,
                cachedir=None,
                rmin=rmin,
                rmax=rmax,
                node_comm=None,
                node_rank_comm=None
            )
            
            # simulate the atmosphere
            err = toast_atmsim_model.simulate(use_cache=False)
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