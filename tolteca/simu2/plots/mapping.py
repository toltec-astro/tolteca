#!/usr/bin/env python

from tollan.utils.log import get_logger
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.wcs.utils import celestial_frame_to_wcs


def plot_mapping(
        simulator,
        mapping,
        target_name=None,
        show=True
        ):
    logger = get_logger()

    if target_name is None:
        target_name = str(mapping.target)

    logger.info(f"plot mapping {mapping}")

    t_pattern = mapping.t_pattern

    t = np.arange(0, t_pattern.to_value(u.s), 0.5) << u.s

    n_pts = t.size
    logger.debug(f"create {n_pts} sampling points")

    # this is to show the mapping in offset unit
    mapping_offset = mapping.offset_mapping_model
    dlon, dlat = mapping_offset(t)

    # then evaluate the mapping model in mapping.ref_frame
    obs_coords = mapping.evaluate_coords(t)

    # and we can convert the obs_coords to other frames if needed
    obs_coords_icrs = obs_coords.transform_to('icrs')

    # and we can convert the obs_coords to other frames if needed
    obs_coords_altaz = obs_coords.transform_to('altaz')

    # now we can plot all these information

    fig = plt.figure(figsize=(12, 8), tight_layout=True)
    gs = fig.add_gridspec(ncols=3, nrows=1)

    ax = fig.add_subplot(gs[0])
    ax.set_aspect('equal')
    ax.plot(
            dlon.to_value(u.arcmin),
            dlat.to_value(u.arcmin),
            color='red')
    ax.set_xlabel('lon. offset (arcmin)')
    ax.set_ylabel('lat. offset (arcmin)')

    # we can overlay the array footprint on the first pointing
    # we can plot the polarimetry groups with different marker
    apt = simulator.array_prop_table
    # makes sure the relevant columns are present
    if {'pg', 'array'}.issubset(set(apt.colnames)):
        apt_0 = apt[apt['array'] == 0]
        for pg, marker in [(0, '+'), (1, 'x')]:
            mask = apt_0['pg'] == pg
            ax.plot(
                (dlon[0] + apt_0[mask]['x_t']).to_value(u.arcmin),
                (dlat[0] + apt_0[mask]['y_t']).to_value(u.arcmin),
                marker=marker, linestyle='none'
                )
    else:
        logger.debug("skipped plot array overlay.")

    # the sky coords, which we may need an fiducial wcs object
    ax = fig.add_subplot(gs[1])
    # target_altaz = mapping.target.transform_to(
    #     resolve_sky_coords_frame(
    #         'altaz', observer=mapping.observer, time_obs=mapping.t0))
    # ax.set_aspect(np.cos(target_altaz.alt.radian))
    # ax.set_aspect(1. / np.cos(target_altaz.alt.radian))
    ax.plot(
        obs_coords_altaz.az.degree, obs_coords_altaz.alt.degree,
        color='red',
        )
    ax.set_xlabel('Az')
    ax.set_ylabel('Alt')

    # we can plot in the icrs
    target_icrs = mapping.target.transform_to('icrs')
    w = celestial_frame_to_wcs(target_icrs.frame)
    # set the crval to target
    w.wcs.crval = np.array([target_icrs.ra.degree, target_icrs.dec.degree])
    ax = fig.add_subplot(gs[2], projection=w)
    ax.set_aspect('equal')
    ax.plot(
        obs_coords_icrs.ra, obs_coords_icrs.dec,
        transform=ax.get_transform('icrs'),
        color='red',
        )
    fig.suptitle(
            f"{mapping.name} of {target_name} at "
            f"{mapping.t0.to_value('iso')}")
    fig.canvas.draw()  # fix layout
    if show:
        plt.show()

    return fig, gs
