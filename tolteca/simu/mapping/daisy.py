#!/usr/bin/env python

import astropy.units as u
import numpy as np


def setup_radial_scan(phi0=0, dphi0=1, dphi1=10, dphi2=30, dphi3=None):
    """
    Scan params.

    Parameters
    ----------
    phi0 : u.Quantity
        The angle of starting point.

    dphi0 : u.Quantity
        The angle between spokes

    dphi1 : u.Quantity
        The angle of each scan

    dphi2 : u.Quantity
        The step angle between scans

    dphi3 : double, optional
        If set, this is the offset of steps around circle.
    """
    petals_per_scan = int(dphi1.to_value(u.deg) / dphi0.to_value(u.deg))
    if dphi2 <= (1 << u.deg):
        raise ValueError("scan step too small")
    scans_per_circle = int(180. / np.abs(dphi2.to_value(u.deg)))

    # this is the fraction of scanned area in the circle
    # note that this may have overlapping scans
    # when diph2 < dphi1
    fill_factor = (dphi1.to_value(u.deg) * scans_per_circle) / 180.

    # the number of circles to cover in full
    circles_per_map = int(1 / fill_factor)

    if dphi3 is None:
        dphi3 = dphi2 * fill_factor

    return (
        phi0, dphi0, dphi1, dphi2, dphi3,
        petals_per_scan, scans_per_circle, circles_per_map
        )


def _rot_xy(x0, vx0, ax0, y0, vy0, ay0, omega, t, delta):
    """
    Rotate on the sky.

    Parameters
    ----------
    x0, vx0, ax0, y0, vy0, ay0 :
        The instantaneous state.

    omega :
        The angular frequency of the rotation.
    t :
        The time of the observation
    delta :
        The phase to add to the rotation.
    """
    # import pdb
    # pdb.set_trace()
    st = np.sin(omega * t + delta)
    ct = np.cos(omega * t + delta)

    x1 = x0 * ct - y0 * st
    y1 = x0 * st + y0 * ct

    omega_v = omega.to(1 / u.s, equivalencies=u.dimensionless_angles())

    vx1 = (vx0 - y0 * omega_v) * ct - (vy0 + x0 * omega_v) * st
    vy1 = (vx0 - y0 * omega_v) * st + (vy0 + x0 * omega_v) * ct

    vy1_v = vy1.to(1 / u.s, equivalencies=u.dimensionless_angles())

    ax1 = (ax0 - 2 * vy0 * omega_v - x0 * omega_v ** 2 + x0 * vy1_v ** 2
           ) * ct - (
               ay0 + 2 * vx0 * omega_v - y0 * omega_v ** 2 + y0 * vy1_v ** 2
               ) * st
    ay1 = (ax0 - 2 * vy0 * omega_v - x0 * omega_v ** 2) * st + \
        (ay0 + 2 * vx0 * omega_v - y0 * omega_v ** 2) * ct

    return x1, vx1, ax1, y1, vy1, ay1


def _step_const_v_1d(dt, x0, v, x_lim):
    """
    Evaluate daisy pattern step at `x0` with step `dt`.

    Parameters
    ----------
    dt :
        The step size.
    x0 :
        The initial position, signed.
    v :
        The velocity, signed
    x_lim :
        The limit of x such that -x_lim < x < x_lim.
    """
    v_unit = v.unit
    a_unit = v_unit / u.s

    if x_lim.value < 0:
        raise ValueError("x_lim has to be positive.")
    x1 = dt * v + x0
    v1 = v
    a1 = 0 << a_unit

    if np.abs(x1) > x_lim:
        # set next step to limit and stop
        x1 = x_lim if x1.value > 0 else -x_lim
        v1 = 0 << v_unit
        a1 = 0 << a_unit
    return x1, v1, a1


def _step_one_over_r_1d(dt, x0, v_lim, x_lims, reverse=False):
    """
    Evaluate daisy pattern step at `x0`, with step `dt`.

    """
    x_min, x_max = x_lims
    if reverse:
        s = -1
    else:
        s = 1

    def _v(x):
        if np.abs(x) < x_min:
            return s * v_lim * (1 - x ** 2 / (3 * x_min ** 2))
        return s * 2 / 3 * v_lim * x_min / np.abs(x)

    def _a(x, v):
        if np.abs(x) < x_min:
            return -2 * s * v_lim * x * v / (3 * x_min ** 2)
        return -v ** 2 / x

    def _dx_term(x):
        return dt * _v(x)

    k1 = _dx_term(x0) / 2
    k2 = _dx_term(x0 + k1 / 2)
    k3 = _dx_term(x0 + k1 / 2 + k2 / 2)
    k4 = _dx_term(x0 + k1 / 2 + k2 / 2 + k3)
    x1 = x0 + (k1 + 2 * (k2 + k3) + k4) / 6
    v1 = _v(x1)
    a1 = _a(x1, v1)
    # check x range
    if np.abs(x1) > x_max:
        x1 = x_max if x1 > 0 else -x_max
        v1 = 0
        a1 = 0
    return x1, v1, a1


def eval_const_v_2d(t, speed, size, omega, delta, dphi0, f_smp):
    """
    Evaluate daisy pattern position at t.

    Parameters
    ----------
    t :
        The time.
    speed :
        The scan speed.
    size :
        The diameter of the pattern.
    omega :
        The rotation rate.
    delta :
        Added phase to rotation.
    """
    # start at center
    r_max = size / 2.
    x_unit = r_max.unit
    v = speed
    v_unit = v.unit
    a_unit = v.unit / u.s
    # make sampling points
    dt = (1 / f_smp)
    ts = np.arange(0, t.to_value(u.s), dt.to_value(u.s)) << u.s
    rs = np.empty(ts.shape, dtype=np.double) << x_unit
    vrs = np.empty(ts.shape, dtype=np.double) << v_unit
    ars = np.empty(ts.shape, dtype=np.double) << a_unit
    deltas = np.empty(ts.shape, dtype=np.double) << delta.unit
    rs[0] = 0 << x_unit
    vrs[0] = v
    ars[0] = 0 << a_unit
    deltas[0] = delta

    x2s = np.empty(ts.shape, dtype=np.double) << x_unit
    vx2s = np.empty(ts.shape, dtype=np.double) << v_unit
    ax2s = np.empty(ts.shape, dtype=np.double) << a_unit
    y2s = np.empty(ts.shape, dtype=np.double) << x_unit
    vy2s = np.empty(ts.shape, dtype=np.double) << v_unit
    ay2s = np.empty(ts.shape, dtype=np.double) << a_unit
    x2s[0] = y2s[0] = 0 << x_unit
    vx2s[0] = vy2s[0] = 0 << v_unit
    ax2s[0] = ay2s[0] = 0 << a_unit

    # run the pattern for i>=1
    for i in range(1, len(ts)):
        x1, v1, a1 = _step_const_v_1d(dt, rs[i - 1], vrs[i-1], r_max)
        if v1.value != 0:
            d1 = deltas[i - 1]
            x2, vx2, ax2, y2, vy2, ay2 = _rot_xy(
                x1, v1, a1,
                0 << x_unit, 0 << v_unit, 0 << a_unit,
                omega, ts[i], d1)
        else:
            # reverse with offset of petal
            d1 = deltas[i - 1] + dphi0
            x2, vx2, ax2, y2, vy2, ay2 = _rot_xy(
                x1, v1, a1,
                0 << x_unit, 0 << v_unit, 0 << a_unit,
                omega, ts[i], d1)
            v1 = -vrs[i-1]
            vx2 = vy2 = 0 << v_unit
            ax2 = ay2 = 0 << a_unit
        rs[i] = x1
        vrs[i] = v1
        ars[i] = a1
        deltas[i] = d1
        x2s[i] = x2
        vx2s[i] = vx2
        ax2s[i] = ax2
        y2s[i] = y2
        vy2s[i] = vy2
        ay2s[i] = ay2
    return locals()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    f_smp = 10 << u.Hz

    t_exp = 100 << u.s
    speed = 40 << u.arcsec / u.s
    size = 10 << u.arcmin
    omega = 1 << u.deg / u.s
    delta = 30 << u.deg
    dphi0 = 10 << u.deg

    info = eval_const_v_2d(t_exp, speed, size, omega, delta, dphi0, f_smp)

    fig, axes = plt.subplots(1, 1, constrained_layout=True, squeeze=False)

    ax = axes[0, 0]
    ax.set_aspect('equal')
    ax.set_xlabel('lon. offset (arcmin)')
    ax.set_ylabel('lat. offset (arcmin)')

    ax.plot(
            info['x2s'].to_value(u.arcmin),
            info['y2s'].to_value(u.arcmin),
            linestyle='-',
            marker=None,
        )
    ax.scatter(
            info['x2s'].to_value(u.arcmin),
            info['y2s'].to_value(u.arcmin),
            c=info['ts'].to_value(u.s),
        )
    plt.show()
