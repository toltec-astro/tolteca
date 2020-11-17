#! /usr/bin/env python


from astropy import units as u
from astropy.time import Time
from tolteca.simu import SimulatorRuntime
from astroquery.utils import parse_coordinates
import numpy as np


if __name__ == "__main__":

    import sys

    # given that we have a workdir setup, we can use the path to init
    # a simulator runtime object, which parses the contents of the workdir
    # and has the methods to inspect them
    # here we assume the script is called as
    # python plot_mapping.py /path/to/simu/workdir
    simrt = SimulatorRuntime(sys.argv[1])

    print(simrt)

    # if in the simrt.config the simulator is configured (key 'simu' exists),
    # one can get the # mapping model via:
    m_obs = simrt.get_mapping_model()

    # m_obs is an subclass of `astropy.modeling.Model` that evaluates
    # the telescope pointing spec at any time.
    # in this example, we have setup a raster scan in the config so
    # the returned m_obs is a SkyRasterScanModel
    # one can get the timeseries of coords by calling `evaluate_at` with
    # some target or just call the object for the offsets

    # here we use the astroquery.parse_coordinates to parse the coord
    # from string for convenience but target could be just specified
    # as an astropy.coordinates.SkyCoord object as
    # target = SkyCoord(180. * u.deg, -20 * u.deg)
    target = parse_coordinates('M31')  # or '180d -20d', '11h21m13s 0d1m2s'
    # target = parse_coordinates('5h0m0s 0d0m0s')

    print(f"ref coord: {target.to_string('hmsdms')}")

    # now we make an array of times for the evaluation

    t_total = m_obs.get_total_time()

    t = np.arange(0, t_total.to_value(u.s), 0.5) * u.s

    n_pts = t.size

    print(f"create {n_pts} pointings")

    # offsets is a tuple (lon, lat) of length t that is the mapping pattern
    offsets = m_obs(t)

    # obs_coords is a of SkyCoord array that has the same shape as t
    # which is the mapping pattern with absolute coordinates
    # the frame of these coords is the same as the frame of the target
    # coord, in this case the pattern is defined in the equitorial
    obs_coords = m_obs.evaluate_at(target, t)

    # we can also construct the instrument simulator
    # object which has the information of the instrument
    # if instru is toltec, the returned object is an instance of
    # tolteca.simu.toltec.ToltecObsSimulator

    simulator = simrt.get_instrument_simulator()

    # we can access the array property table via
    apt = simulator.table

    print(apt)

    # we can get the a1100 subset
    # the other two arrays are named a1400 and a2000
    apt_a1100 = apt[apt['array_name'] == 'a1100']

    # the array propery table contains the projected array locations in
    # the toltec frame, which are delta Az (column name x_t)
    # and delta El (column name y_t)

    # now we can plot all these information

    import matplotlib.pyplot as plt

    fig = plt.figure(constrained_layout=True)

    # the pointing offsets
    ax = fig.add_subplot(2, 1, 1)
    ax.set_aspect('equal')
    ax.plot(
            offsets[0].to_value(u.arcmin),
            offsets[1].to_value(u.arcmin),
            color='red')
    ax.set_xlabel('lon. offset (arcmin)')
    ax.set_ylabel('lat. offset (arcmin)')

    # we can overlay the array footprint on the first pointing
    # we can plot the polarimetry groups with different marker
    for pg, marker in [(0, '+'), (1, 'x')]:
        m = apt_a1100['pg'] == pg
        ax.plot(
            (offsets[0][0] + apt_a1100[m]['x_t']).to_value(u.arcmin),
            (offsets[0][1] + apt_a1100[m]['y_t']).to_value(u.arcmin),
            marker=marker, linestyle='none'
            )
    # the sky coords, which we need an fiducial wcs object

    from astropy.wcs.utils import celestial_frame_to_wcs

    w = celestial_frame_to_wcs(target.frame)
    # set the crval to target
    w.wcs.crval = np.array([target.ra.degree, target.dec.degree])

    ax = fig.add_subplot(2, 1, 2, projection=w)
    ax.set_aspect('equal')

    ax.plot(
        obs_coords.ra, obs_coords.dec,
        transform=ax.get_transform('icrs'),
        color='red',
        )

    # to overlay the array footprint, we need do some coord transform
    # using the sky project model, which require a ref coord and an obs time
    # we can transform the toltec frame offset to the equitorial via
    # frame='icrs'
    # frame='native' can be used to transform to the altaz.
    m_proj = simulator.get_sky_projection_model(
            ref_coord=obs_coords[0], time_obs=Time('2020-11-17T00:00:00'))
    a_ra, a_dec = m_proj(apt_a1100['x_t'], apt_a1100['y_t'], frame='icrs')
    # a_az, a_alt = m_proj(apt_a1100['x_t'], apt_a1100['y_t'], frame='native')

    for pg, marker in [(0, '+'), (1, 'x')]:
        m = apt_a1100['pg'] == pg
        ax.plot(
            a_ra[m], a_dec[m],
            marker=marker, linestyle='none',
            transform=ax.get_transform('icrs'),
            )

    plt.show()
