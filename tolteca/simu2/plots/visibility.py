#!/usr/bin/env python


from ..toltec import site_info
from tollan.utils.log import get_logger
import numpy as np
import astropy.units as u
from astroplan import FixedTarget
from astroplan import (AltitudeConstraint, AtNightConstraint)
from astroplan import observability_table
from astroplan.plots import plot_sky
import matplotlib.pyplot as plt


__all__ = ['plot_visibility']


def plot_visibility(t0, targets, target_names=None, show=True):

    logger = get_logger()

    obs = site_info['observer']

    logger.debug(f"{obs=}")

    if target_names is None:
        target_names = [str(target) for target in targets]
    targets = [
            FixedTarget(coord=target,  name=name)
            for target, name in zip(targets, target_names)]

    constraints = [
            AltitudeConstraint(20*u.deg, 90*u.deg),
            AtNightConstraint.twilight_civil()
            ]

    time_grid = t0 + (np.arange(0, 24, 0.5) << u.h)

    summary = observability_table(
            constraints, obs, targets, times=time_grid)

    logger.info(f'Visibility of targets on day of {t0}\n{summary}')

    m_good = np.logical_and.reduce([
            constraint(
                obs, targets,
                times=time_grid,
                grid_times_targets=True)
            for constraint in constraints])

    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(ncols=1, nrows=2,
                          height_ratios=[4, 1])
    ax = fig.add_subplot(gs[0], projection='polar')
    ax.set_title(
            f"Visibility of targets on "
            f"{t0.to_value('iso', subfmt='date')}")

    name_width = max([len(target.name) for target in targets])
    for i, target in enumerate(targets):
        m = m_good[i]
        n = ~m
        name_padded = f'{{:{name_width}s}}'.format(target.name)
        if n.sum() > 0:
            if m.sum() == 0:
                # add label for not observable
                label = f'{name_padded} Not Observable'
            else:
                label = None
            plot_sky(
                    target, obs, time_grid[n], ax=ax,
                    style_kwargs={
                        'marker': 'x',
                        'label': label,
                        'color': f'C{i % 10}',
                        }
                    )
        if m.sum() > 0:
            t1, t2 = time_grid[m][0], time_grid[m][-1]
            plot_sky(
                    target, obs, time_grid[m], ax=ax,
                    style_kwargs={
                        'marker': 'o',
                        'color': f'C{i % 10}',
                        'label': '{} UT {}-{}'.format(
                            name_padded,
                            t1.datetime.strftime("%H:%M"),
                            t2.datetime.strftime("%H:%M")
                            )
                        }
                    )

    fig.legend(
            prop={'family': 'monospace'},
            loc='upper left', bbox_to_anchor=(0.1, 0.9))

    ax = fig.add_subplot(gs[1])

    extent = [-0.5, -0.5 + len(time_grid), -0.5, -0.5 + len(targets)]

    ax.imshow(m_good, extent=extent)

    ax.set_yticks(range(len(targets)))
    ax.set_yticklabels(reversed([target.name for target in targets]))

    ax.set_xticks(range(len(time_grid)))
    ax.set_xticklabels([t.datetime.strftime("%H:%M") for t in time_grid])
    ax.set_xticks(np.arange(extent[0], extent[1]), minor=True)
    ax.set_yticks(np.arange(extent[2], extent[3]), minor=True)

    ax.grid(which='minor', color='#8c8c8c', linestyle='-', linewidth=1)
    ax.tick_params(axis='x', which='minor', bottom='off')
    ax.tick_params(axis='y', which='minor', left='off')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_xlabel('Time on {0} UTC'.format(time_grid[0].datetime.date()))

    if show:
        plt.show()

    return fig, gs
