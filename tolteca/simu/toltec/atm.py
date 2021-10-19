
import numpy as np
import datetime
import astropy.units as u

from tollan.utils.log import timeit, get_logger
import toast

# load the optional (not for us) atmosphere tools
have_atm_utils = None
if have_atm_utils is None:
    try:
        from toast.atm import atm_absorption_coefficient_vec, atm_atmospheric_loading_vec
        have_atm_utils = True
    except ImportError:
        have_atm_utils = False

__all__ = ['ToastAtmosphereSlabs']

class ToastAtmosphereSlabs(object):
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

    @timeit
    def generate_slabs(self):
        """ Generates with parameters
        """
        self.atm_slabs_dict = self._generate_toast_atm_slabs(
            self.t0, self.tmin, self.tmax, 
            self.azmin, self.azmax, 
            self.elmin, self.elmax
        )

    @timeit
    def absload_calc(self):
        pass
        # self.bandpass = {
        #     'a1100':{
        #         'absorption': ,
        #         'loading': ,
        #         'bandpass': ,
        #     }, 
        #     'a1400':{
        #         'absorption': ,
        #         'loading':  ,
        #         'bandpass':              
        #     }, 
        #     'a2000':{
        #         'absorption': ,
        #         'loading':  ,    
        #         'bandpass':           
        #     }
        # }
        # self.bandpass_absorption = 
        # self.bandpass_loading = 

        # # TODO: I think this is already somwhere else that I can just import
        # lmt_height = 4640.0 * u.m
        # for band in ['a1100', 'a1400', 'a2000']:
        #     absorption = atm_absorption_coefficient_vec(
        #         lmt_height.to_value(u.meter),
        #         self.sim_weather.air_temperature.to_value(u.Kelvin),
        #         self.sim_weather.surface_pressure.to_value(u.Pa),
        #         self.sim_weather.pwv.to_value(u.mm),
        #         bandpass.freqs[0].to_value(u.GHz),
        #         bandpass.freqs[-1].to_value(u.GHz),
        #         len(bandpass.freqs),
        #     )

        #     loading = atm_atmospheric_loading_vec(
        #         lmt_height.to_value(u.meter),
        #         self.sim_weather.air_temperature.to_value(u.Kelvin),
        #         self.sim_weather.surface_pressure.to_value(u.Pa),
        #         self.sim_weather.pwv.to_value(u.mm),
        #         bandpass.freqs[0].to_value(u.GHz),
        #         bandpass.freqs[-1].to_value(u.GHz),
        #         len(bandpass.freqs),
        #     )

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
        atm_slabs_dict = dict()

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
            atm_slabs_dict[slab_id] = toast_atmsim_model
            
            # use a new RNG stream for each slab
            counter1 += 1

            # decrease resolution as we increase altitude
            rmin   = u.Quantity(rmax)
            rmax  *= scale
            xstep *= np.sqrt(scale)
            ystep *= np.sqrt(scale)
            zstep *= np.sqrt(scale)

        # return atm_slabs_list
        return atm_slabs_dict