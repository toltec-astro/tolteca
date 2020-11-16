#! /usr/bin/env python


from astropy import units as u
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
    sim = SimulatorRuntime(sys.argv[1])

    print(sim)

    # if in the sim.config the simulator is configured (key 'simu' exists),
    # one can get the # mapping model via:
    m_obs = sim.get_mapping_model()

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
    obs_coords = m_obs.evaluate_at(target, t)

    # now we can plot them

    import matplotlib.pyplot as plt

    fig = plt.figure(tight_layout=True)

    # the offsets
    ax = fig.add_subplot(2, 1, 1)
    ax.set_aspect('equal')
    ax.plot(
            offsets[0].to_value(u.arcmin),
            offsets[1].to_value(u.arcmin),
            color='red')
    ax.set_xlabel('lon. offset (arcmin)')
    ax.set_ylabel('lat. offset (arcmin)')

    # the sky coords, which we need an fiducial wcs object

    from astropy.wcs.utils import celestial_frame_to_wcs

    w = celestial_frame_to_wcs(obs_coords.frame)

    ax = fig.add_subplot(2, 1, 2, projection=w)
    ax.set_aspect('equal')

    ax.plot(
        obs_coords.ra, obs_coords.dec,
        transform=ax.get_transform('icrs'),
        color='red',
        )
    plt.show()
