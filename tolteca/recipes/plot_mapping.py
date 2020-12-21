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

    # an example to create an in-memory simrt with updated config
    simrt = SimulatorRuntime.from_config(simrt.config, {'new_stuff': 'abc'})

    # if in the simrt.config the simulator is configured (key 'simu' exists),
    # one can get the # mapping model via:
    m_obs = simrt.get_mapping_model()

    # we can also construct the instrument simulator
    # object which has the information of the instrument
    # if instru is toltec, the returned object is an instance of
    # tolteca.simu.toltec.ToltecObsSimulator

    simulator = simrt.get_instrument_simulator()

    # we can access the array property table via
    apt = simulator.table

    # we can get the a1100 subset
    # the other two arrays are named a1400 and a2000
    apt_a1100 = apt[apt['array_name'] == 'a1100']

    # the array propery table contains the projected array locations in
    # the toltec frame, which are delta Az (column name x_t)
    # and delta El (column name y_t)

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
    # coord.

    # so here we need to check the ref_frame setting so that we transform
    # the target coord to the ref_frame.
    # this transformation will need the absolute time if the ref_frame
    # is AltAz
    t0 = m_obs.t0  # specified in the 60_simu.yaml t0

    ref_frame = simulator.resolve_sky_map_ref_frame(
            ref_frame=m_obs.ref_frame,
            time_obs=t0 + t)
    # we now can generate the obs_coords in ref_frame
    # by first get the target coords in ref_frame and
    # evaluate the sky offset map model
    target_in_ref_frame = target.transform_to(ref_frame)
    obs_coords = m_obs.evaluate_at(target_in_ref_frame, t)

    # and we can convert the obs_coords to other frames if needed
    obs_coords_icrs = obs_coords.transform_to('icrs')

    # and we can convert the obs_coords to other frames if needed
    obs_coords_altaz = obs_coords.transform_to('altaz')

    # now we can plot all these information

    import matplotlib.pyplot as plt

    fig = plt.figure(constrained_layout=True)

    # the pointing offsets
    ax = fig.add_subplot(3, 1, 1)
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

    # we can plot in the ref_frame

    ax = fig.add_subplot(3, 1, 2)
    ax.set_aspect('equal')

    ax.plot(
        obs_coords.data.lon.degree, obs_coords.data.lat.degree,
        color='red',
        )

    # we can plot in the icrs
    w = celestial_frame_to_wcs(target.frame)
    # set the crval to target
    w.wcs.crval = np.array([target.ra.degree, target.dec.degree])

    ax = fig.add_subplot(3, 1, 3, projection=w)
    ax.set_aspect('equal')

    ax.plot(
        obs_coords_icrs.ra, obs_coords_icrs.dec,
        transform=ax.get_transform('icrs'),
        color='red',
        )

    # to plot a point in the toltec frame, one need to use the sky projection
    # model, which accepts a ref_coord and a time_obs.
    # the model could be a model set, i.e., a vector of coord and time_obs
    # are given.
    # below show an example of geting the trajectory of a point in the toltec
    # frame with offsets (10, 10) arcsec to the boresight.

    # note that the ref_coord is the vector of positions in the mapping
    # pattern. time_obs is the vector of absolute times in the mapping pattern
    # the created m_proj is a "modelset" of size npts, which expects
    # inputs to be vectors of size npts in its first axis (axis 0).
    m_proj = simulator.get_sky_projection_model(
            ref_coord=obs_coords, time_obs=Time('2020-11-17T00:00:00') + t)

    # the trajectory of a point in the toltec frame, with coordinate
    # (10, 10) arcsec, i.e., an fixed point attched on the TolTEC and offseted
    # from the boresight by 10", 10"
    # the shape has to be (npts, ) because we are eveluating this point
    # along the npts times of the mapping pattern.
    x_t = np.full((n_pts, ), 10.) << u.arcsec
    y_t = np.full((n_pts, ), 10.) << u.arcsec
    a_ra, a_dec = m_proj(x_t, y_t, frame='icrs')

    # this plot the trajectory of this point.
    ax.plot(
        a_ra, a_dec,
        transform=ax.get_transform('icrs'),
        color='blue',
        )

    # to overlay the array footprint, we need do some coord transform
    # using the sky project model, which require a ref coord and an obs time
    # we can transform the toltec frame offset to the equitorial via
    # frame='icrs'
    # frame='native' can be used to transform to the altaz.

    # this transformation uses the same logic as above, except that we now
    # have n_detectors points in the toltec frame, instead of just pone.
    # the input the m_proj model should be shape (n_pts, n_detectors)

    # this is computationally expensive. So below we only plot the
    # full array at the first position of the mapping pattern. This
    # can be done by creating m_proj with a scalar ref_coord and time_obs,
    # i.e., n_pts == 1 in the following code
    m_proj = simulator.get_sky_projection_model(
            ref_coord=obs_coords[0], time_obs=Time('2020-11-17T00:00:00'))

    # because we used scalar ref_coord and time_obs, the m_proj is no longer
    # a modelset. It is a simple model, which we can feed the list of
    # detector offsets in toltec frame to get the array locations in equitorial
    # coords.
    a_ra, a_dec = m_proj(apt_a1100['x_t'], apt_a1100['y_t'], frame='icrs')

    # a_az, a_alt = m_proj(apt_a1100['x_t'], apt_a1100['y_t'], frame='native')

    for pg, marker in [(0, '+'), (1, 'x')]:
        m = apt_a1100['pg'] == pg
        ax.plot(
            a_ra[m], a_dec[m],
            marker=marker, linestyle='none',
            transform=ax.get_transform('icrs'),
            )

    # just for the sake of demonstration, we show how to do the full
    # evalution of all detectors over the entire mapping pattern:
    # because its takes too long for all n_pts, we just demonstrate
    # it for the last 2 points

    # m_proj is now a modelset of size n_pts
    m_proj = simulator.get_sky_projection_model(
            ref_coord=obs_coords[-2:],
            time_obs=Time('2020-11-17T00:00:00') + t[-2:])

    # repeat the array positions so that the shape is (n_pts, n_detectors)
    x_t = np.tile(apt_a1100['x_t'], (2, 1))
    y_t = np.tile(apt_a1100['y_t'], (2, 1))

    # evaluate
    # this times very long time.
    a_ra, a_dec = m_proj(x_t, y_t, frame='icrs')
    # a_ra and a_dec have shape (n_pts, n_detectors)

    print(a_ra.shape)

    plt.show()
