"""Numerical phase fitter retained from Sam Walker's ``kid_phase_fit``.

Source: https://github.com/swalker5/kid_phase_fit
Revision: 4f4c9a71e51d88e41980c5932bd973722a6906a0.

The circle and linear/nonlinear phase calculations retain the upstream
operation order. Standalone file readers, plotting and console output are
excluded. This private kernel is exercised against the upstream implementation
by the drive-fit parity probe; application configuration belongs to the caller.
"""

# Preserve upstream numerical names, signatures and statement order for parity.
# ruff: noqa: N803, E741, FBT002, PLR0913, PLR0917, PLR0915, C901, PLR0912
# ruff: noqa: PLR2004, RET504, RET505, SIM102, SIM105, SIM108, PLR1730

import math

import numpy as np
import scipy.stats
from scipy import linalg
from scipy.optimize import curve_fit


def phase2(z):
    x = np.angle(z)
    x_cp = np.copy(x)
    y = np.unwrap(x_cp)
    return y


def removecable(f, z, tau, verbose=False, showplot=False):
    if tau > 0:
        tau = -1.0 * tau
    if f[0] > 10000000.0:
        f = f / 1000000000.0
    _f1 = f[0]
    z1 = z * np.exp(-1j * 2.0 * np.pi * f * tau)
    return z1


def find_nearest_indice(x, val):
    idx = np.searchsorted(x, val, side="left")
    if idx > 0 and (
        idx == len(x) or math.fabs(val - x[idx - 1]) < math.fabs(val - x[idx])
    ):
        return idx - 1
    else:
        return idx


def trimdata(f, z, f0, Q, numspan=1):
    f0id = find_nearest_indice(f, f0)
    idspan = f0 / Q / (f[1] - f[0])
    idstart = int(np.floor(np.amax([f0id - idspan * numspan, 0])))
    if idstart != 0:
        idstart = idstart + 1
    idstop = int(np.floor(np.amin([f0id + idspan * numspan, len(f)]))) + 1
    if idstop > len(f):
        idstop = len(f)
    fb = f[idstart:idstop]
    zb = z[idstart:idstop]
    return (fb, zb)


def smoothdata2(y, N):
    if N % 2 == 0:
        N = N - 1
    y_out = np.convolve(y, np.ones(N, dtype=int), mode="valid") / N
    r = np.arange(1, N - 1, 2)
    start = np.cumsum(y[: N - 1])[::2] / r
    stop = (np.cumsum(y[:-N:-1])[::2] / r)[::-1]
    y_smooth = np.concatenate((start, y_out, stop))
    return y_smooth


def estpara3(f, z, verbose=False, **keywords):
    if "result0" in keywords:
        result0 = keywords["result0"]
        f0g = result0["f0"]
        Qg = result0["Q"]
        _v = np.amin(np.abs(f - f0g))
        idf0 = np.argmin(np.abs(f - f0g))
        iddf = np.abs(
            find_nearest_indice(f, f0g - f0g / Qg / 2.0)
            - find_nearest_indice(f, f0g + f0g / Qg / 2.0)
        )
    else:
        z1 = np.mean(z[:10])
        z2 = np.mean(z[len(f) - 11 : len(f)])
        zr = (z1 + z2) / 2.0
        _dmax = np.amax(np.abs(z - z1) + np.abs(z - z2))
        idf0 = np.argmax(np.abs(z - z1) + np.abs(z - z2))
        z0 = z[idf0]
        f0g = f[idf0]
        _zcg = (zr + z0) / 2.0
        rg = np.abs(zr - z0) / 2.0
        zz1 = z[: idf0 + 1]
        _r01 = np.amin(np.abs(np.abs(zz1 - z0) - np.sqrt(2.0) * rg))
        id3db1 = np.argmin(np.abs(np.abs(zz1 - z0) - np.sqrt(2.0) * rg))
        zz2 = z[idf0 + 1 : len(z)]
        _r02 = np.amin(np.abs(np.abs(zz2 - z0) - np.sqrt(2.0) * rg))
        id3db2 = np.argmin(np.abs(np.abs(zz2 - z0) - np.sqrt(2.0) * rg))
        iddf = np.amax([np.abs(idf0 - id3db1), np.abs(id3db2)])
        Qg = f0g / (2.0 * iddf * (f[1] - f[0]))
    f0g = float(f0g)
    Qg = float(Qg)
    idf0 = int(idf0)
    iddf = int(iddf)
    return (f0g, Qg, idf0, iddf)


class Circlefit:
    """Circle fit to complex I/Q samples."""

    def __init__(self, z, scaling=False):
        """Compute the fit for the supplied complex samples."""
        self.z = z
        self.scaling = scaling
        self.r, self.zc, self.residue = self.nsphere_fit_test(scaling=self.scaling)

    def nsphere_fit_test(self, scaling):
        """Return the circle radius, center and least-squares residual."""
        x = np.column_stack((self.z.real, self.z.imag))
        x = np.asarray(x)
        n = x.shape[-1]
        x = x.reshape(-1, n)
        m = x.shape[0]
        B = np.empty((m, n + 1), dtype=x.dtype)
        X = B[:, :-1]
        X[:] = x
        B[:, -1] = 1
        if scaling:
            xmin = X.min()
            xmax = X.max()
            scale = 0.5 * (xmax - xmin)
            offset = 0.5 * (xmax + xmin)
            X -= offset
            X /= scale
        d = np.sum(X**2, axis=-1)
        y, residue, _rank_B, _B_sing = linalg.lstsq(
            B, d, overwrite_a=True, overwrite_b=True
        )
        zc = 0.5 * y[:-1]
        r = np.sqrt(y[-1] + np.sum(zc**2))
        if scaling:
            r *= scale
            zc *= scale
            zc += offset
        zc_complex = zc[0] + 1j * zc[1]
        return (r, zc_complex, residue)


class ResonanceFitterSingleTone:
    """Linear and nonlinear phase fits after cable-delay correction."""

    def __init__(
        self,
        f,
        z,
        tau,
        numspan=2,
        tone_freq_lo=0,
        window_width=0,
        pherr_threshold_num=10,
        pherr_threshold=0.2,
        verbose=False,
        **keywords,
    ):
        """Compute the fit for the supplied complex samples."""
        self.f = f
        self.z = z
        self.tau = tau
        self.tone_freq_lo = tone_freq_lo
        self.numspan = numspan
        self.ang = phase2(self.z)
        self.pherr_threshold_num = int(pherr_threshold_num)
        self.pherr_threshold = pherr_threshold
        self.verbose = verbose
        self.window_width = window_width
        if "weight_type" in keywords:
            self.use_weight = True
            self.weight_type = keywords["weight_type"]
            if self.window_width == 0:
                self.use_weight = False
        else:
            self.use_weight = False
        if "result0" in keywords:
            self.result0 = keywords["result0"]
            self.result = self.fits21simnlin()
        else:
            self.result = self.fits21sim()
        self.z1 = removecable(self.f, self.z, self.tau, verbose=self.verbose)
        if "result0" in keywords:
            self.f0g, self.Qg, self.idf0, self.iddf = estpara3(
                self.f, self.z1, verbose=self.verbose, result0=self.result0
            )
        else:
            self.f0g, self.Qg, self.idf0, self.iddf = estpara3(
                self.f, self.z1, verbose=self.verbose
            )
        self.id1 = np.amax([self.idf0 - 2 * self.iddf, 0])
        self.id2 = np.amin([self.idf0 + 2 * self.iddf, len(self.f)])
        if self.id2 < len(self.f):
            self.id2 = self.id2 + 1
        circle_obj = Circlefit(self.z1[self.id1 : self.id2])
        self.r = float(circle_obj.r)
        self.zc = circle_obj.zc
        self.c_residue = circle_obj.residue
        self.theta_zc = np.angle(self.zc)
        self.z2 = (self.zc - self.z1) * np.exp(-1j * self.theta_zc)
        self.ft, self.zt = trimdata(self.f, self.z2, self.f0g, self.Qg, self.numspan)
        self.f0gid = find_nearest_indice(self.f, self.f0g)

    def fphaselin(self, ft, Q, f0, phi):
        xg = ft / f0 - 1.0
        ang = phi + 2.0 * np.arctan(-2.0 * Q * xg)
        return ang

    def calcphasebounds(self, Q, f0, phi, fmargin=0.02, phimargin=0.05):
        bounds = (
            (1000.0, f0 - fmargin * f0, phi - phimargin * np.abs(phi)),
            (10000000.0, f0 + fmargin * f0, phi + phimargin * np.abs(phi)),
        )
        return bounds

    def windowlorentz(self, f):
        f_c = self.tone_freq_lo
        x = 2.0 / self.window_width * (f - f_c)
        uncertainty = np.sqrt(1.0 + np.square(x))
        return uncertainty

    def windowgauss(self, f):
        f_c = self.tone_freq_lo
        x = 2.0 / self.window_width * (f - f_c)
        uncertainty = np.exp(np.square(x) / 2.0)
        return uncertainty

    def fitphasetest(self, ft, angt, Q, f0, phi, **keywords):
        p0 = [Q, f0, phi]
        if self.use_weight:
            if self.weight_type == "lorentz":
                uncertainty_angt = self.windowlorentz(ft)
            elif self.weight_type == "gauss":
                uncertainty_angt = self.windowgauss(ft)
            else:
                uncertainty_angt = np.ones(np.shape(ft))
        else:
            uncertainty_angt = np.ones(np.shape(ft))
        if "bounds" in keywords:
            bounds = keywords["bounds"]
            popt, pcov = curve_fit(
                self.fphaselin, xdata=ft, ydata=angt, p0=p0, bounds=bounds
            )
        elif self.use_weight:
            popt, pcov = curve_fit(
                self.fphaselin,
                xdata=ft,
                ydata=angt,
                p0=p0,
                sigma=uncertainty_angt,
                absolute_sigma=False,
            )
        else:
            popt, pcov = curve_fit(self.fphaselin, xdata=ft, ydata=angt, p0=p0)
        fit_result = self.fphaselin(ft, popt[0], popt[1], popt[2])
        p0_result = self.fphaselin(ft, p0[0], p0[1], p0[2])
        return (popt, pcov, fit_result, p0_result, uncertainty_angt)

    def fitphase2(self, f, z, f0g, Qg, numspan=2, showplot=False, use_bounds=False):
        zinf = (z[0] + z[-1]) / 2.0
        ezinf = zinf / np.abs(zinf)
        z1_1 = z / -ezinf
        ang = phase2(z1_1) + np.angle(-ezinf)
        hnumsmopts = int(np.floor(f0g / Qg / 3.0 / (f[1] - f[0])))
        hnumsmopts = int(np.amax([np.amin([hnumsmopts, 20]), 0]))
        if hnumsmopts - 1 < 0:
            hnumsmopts_l = 0
        else:
            hnumsmopts_l = hnumsmopts - 1
        N = int(hnumsmopts * 2)
        zm = smoothdata2(z, N)
        zm_inf = (zm[0] + zm[-1]) / 2.0
        ezminf = zm_inf / np.abs(zm_inf)
        zm1_1 = zm / -ezminf
        angm = phase2(zm1_1) + np.angle(-ezminf)
        dangm = angm[hnumsmopts_l : len(angm)] - angm[0 : len(angm) - hnumsmopts + 1]
        _fd = f[hnumsmopts_l : len(angm)] - f[0 : len(angm) - hnumsmopts + 1]
        _dangmin = np.amin(dangm[hnumsmopts_l : len(dangm) - hnumsmopts + 1])
        idx = np.argmin(dangm[hnumsmopts_l : len(dangm) - hnumsmopts + 1])
        f0 = f[idx + hnumsmopts]
        fstep = f[1] - f[0]
        idx = find_nearest_indice(f, f0)
        wid = int(np.floor(f0 / Qg / 3.0 / fstep))
        fmid = f[np.amax([idx - wid, 0]) : np.amin([idx + wid + 1, len(f)])]
        xmid = (f0 - fmid) / f0
        angmid = ang[np.amax([idx - wid, 0]) : np.amin([idx + wid + 1, len(f)])]
        slope, intercept, _r, _p, _se = scipy.stats.linregress(xmid, angmid)
        phi = intercept
        Q = np.abs(slope / 4.0)
        ft, zt = trimdata(f, z, f0, Q, numspan)
        ft, angt = trimdata(f, ang, f0, Q, numspan)
        if use_bounds:
            bounds = self.calcphasebounds(Q, f0, phi)
            popt, pcov, fit_result, p0_result, uncertainty_angt = self.fitphasetest(
                ft, angt, Q, f0, phi, bounds=bounds
            )
        else:
            popt, pcov, fit_result, p0_result, uncertainty_angt = self.fitphasetest(
                ft, angt, Q, f0, phi
            )
        fphasetest = self.fphaselin(ft, popt[0], popt[1], popt[2])
        pherr = angt - fphasetest
        return (
            popt,
            pcov,
            pherr,
            fit_result,
            p0_result,
            ft,
            zt,
            angt,
            ang,
            uncertainty_angt,
            ezinf,
        )

    def fits21sim(self, showplot=False, use_bounds=False):
        result = {}
        result["tau"] = self.tau
        numspan = self.numspan
        z1 = removecable(self.f, self.z, self.tau, verbose=self.verbose)
        f0g, Qg, idf0, iddf = estpara3(self.f, z1, verbose=self.verbose)
        id1 = np.amax([idf0 - 2 * iddf, 0])
        id2 = np.amin([idf0 + 2 * iddf, len(self.f)])
        if id2 < len(self.f):
            id2 = id2 + 1
        circle_obj = Circlefit(z1[id1:id2])
        result["r"] = float(circle_obj.r)
        result["zc"] = circle_obj.zc
        result["c_residue"] = circle_obj.residue
        theta_zc = np.angle(result["zc"])
        z2 = (result["zc"] - z1) * np.exp(-1j * theta_zc)
        if use_bounds:
            (
                popt,
                pcov,
                pherr,
                fit_result,
                x0_result,
                ft,
                zt,
                angt,
                ang,
                uncertainty_angt,
                ezinf,
            ) = self.fitphase2(self.f, z2, f0g, Qg, numspan, use_bounds=use_bounds)
        else:
            (
                popt,
                pcov,
                pherr,
                fit_result,
                x0_result,
                ft,
                zt,
                angt,
                ang,
                uncertainty_angt,
                ezinf,
            ) = self.fitphase2(self.f, z2, f0g, Qg, numspan)
        result["popt"] = popt
        result["Q"] = popt[0]
        result["f0"] = popt[1]
        result["phi"] = popt[2]
        result["pcov"] = pcov
        result["pherr"] = pherr
        result["popt_err"] = np.sqrt(np.diag(pcov))
        result["fit_result"] = fit_result
        result["fit_result2"] = self.fphaselin(
            self.f, result["Q"], result["f0"], result["phi"]
        )
        result["fit_resultt"] = self.fphaselin(
            ft, result["Q"], result["f0"], result["phi"]
        )
        result["x0_result"] = x0_result
        result["ft"] = ft
        result["zt"] = zt
        result["ang"] = ang
        result["angt"] = angt
        result["uncertainty_angt"] = uncertainty_angt
        result["ezinf"] = ezinf
        idft1 = find_nearest_indice(self.f, ft[0])
        widft = len(ft)
        zt_n = self.z[idft1 : idft1 + widft]
        ft_n = self.f[idft1 : idft1 + widft]
        result["zt_n"] = zt_n
        result["ft_n"] = ft_n
        result["zd"] = (
            -result["zc"]
            / np.abs(result["zc"])
            * np.exp(1j * result["phi"])
            * result["r"]
            * 2.0
        )
        result["zinf"] = result["zc"] - result["zd"] / 2.0
        result["Qc"] = np.abs(result["zinf"]) / np.abs(result["zd"]) * result["Q"]
        result["Qi"] = 1.0 / (1.0 / result["Q"] - 1.0 / result["Qc"])
        result["zf0"] = result["zinf"] + result["zd"]
        l = (
            np.real(result["zf0"] * np.conj(result["zinf"]))
            / np.abs(result["zinf"]) ** 2
        )
        result["Qi0"] = result["Q"] / l
        result["f00id"] = find_nearest_indice(self.f, result["f0"])
        result["f0_corr"] = result["f0"]
        result["f0id"] = result["f00id"]
        iq_model = 1.0 - result["Q"] / result["Qc"] * np.exp(1j * result["phi"]) / (
            1.0 + 2j * result["Q"] * (self.f / result["f0"] - 1.0)
        )
        iq_modelt = 1.0 - result["Q"] / result["Qc"] * np.exp(1j * result["phi"]) / (
            1.0 + 2j * result["Q"] * (result["ft"] / result["f0"] - 1.0)
        )
        result["iq_model"] = iq_model
        result["iq_modelt"] = iq_modelt
        result["A_mag"] = result["r"]
        result["A_magt"] = result["r"]
        ang_to_iq = (
            result["A_mag"]
            * (
                np.cos(result["fit_result2"] - np.angle(-result["ezinf"]))
                + 1j * np.sin(result["fit_result2"] - np.angle(-result["ezinf"]))
            )
            * np.abs(result["iq_model"])
        )
        angt_to_iq = (
            result["A_magt"]
            * (
                np.cos(result["fit_resultt"] - np.angle(-result["ezinf"]))
                + 1j * np.sin(result["fit_resultt"] - np.angle(-result["ezinf"]))
            )
            * np.abs(result["iq_modelt"])
        )
        result["ang_to_z2"] = ang_to_iq * -result["ezinf"]
        result["angt_to_z2"] = angt_to_iq * -result["ezinf"]
        result["ang_to_z1"] = result["zc"] - result["ang_to_z2"] / np.exp(
            -1j * np.angle(result["zc"])
        )
        result["angt_to_z1"] = result["zc"] - result["angt_to_z2"] / np.exp(
            -1j * np.angle(result["zc"])
        )
        result["ang_to_z"] = result["ang_to_z1"] / np.exp(
            -1j * 2.0 * np.pi * (self.f / 1000000000.0) * self.tau
        )
        result["angt_to_z"] = result["angt_to_z1"] / np.exp(
            -1j * 2.0 * np.pi * (result["ft"] / 1000000000.0) * self.tau
        )
        return result

    def fphasenlin(self, f, Q, alpha, f00, phi, r, z):
        A = z - r * np.exp(1j * (phi + np.pi))
        f0 = f00 - alpha * np.abs(A) ** 2
        x = (f - f0) / f0
        ang = phi + 2.0 * np.arctan(-2.0 * Q * x)
        return ang

    def fphasenlin2(self, f, Q, alpha, f00, phi, r, z):
        A = z - r * np.exp(1j * (phi + np.pi))
        f0 = f00 - alpha * np.abs(A) ** 2
        x = (f - f0) / f0
        ang = phi + 2.0 * np.arctan(-2.0 * Q * x)
        return (ang, x)

    def calcphasenlinbounds(self, Q, alpha, f0, phi, fmargin=0.02, phimargin=0.05):
        bounds = (
            (1000.0, -np.inf, f0 - fmargin * f0, phi - phimargin * np.abs(phi)),
            (10000000.0, np.inf, f0 + fmargin * f0, phi + phimargin * np.abs(phi)),
        )
        return bounds

    def fitphasenlintest2(self, ft, angt, Q, alpha, f0, phi, r, zt, **keywords):
        p0 = [Q, alpha, f0, phi]
        if self.use_weight:
            if self.weight_type == "lorentz":
                uncertainty_angt = self.windowlorentz(ft)
            elif self.weight_type == "gauss":
                uncertainty_angt = self.windowgauss(ft)
            else:
                uncertainty_angt = np.ones(np.shape(ft))
        else:
            uncertainty_angt = np.ones(np.shape(ft))
        if "bounds" in keywords:
            bounds = keywords["bounds"]
            if "err" in keywords:
                err = keywords["err"]
                popt, pcov = curve_fit(
                    lambda f_lamb, a, b, c, d: self.fphasenlin(
                        f_lamb, a, b, c, d, r, zt
                    ),
                    ft,
                    angt,
                    p0=p0,
                    sigma=err,
                    bounds=bounds,
                )
            else:
                try:
                    popt, pcov = curve_fit(
                        lambda f_lamb, a, b, c, d: self.fphasenlin(
                            f_lamb, a, b, c, d, r, zt
                        ),
                        ft,
                        angt,
                        p0=p0,
                        bounds=bounds,
                    )
                except RuntimeError:
                    pass
        elif "err" in keywords:
            err = keywords["err"]
            popt, pcov = curve_fit(
                lambda f_lamb, a, b, c, d: self.fphasenlin(f_lamb, a, b, c, d, r, zt),
                ft,
                angt,
                p0=p0,
                sigma=err,
            )
        elif self.use_weight:
            try:
                popt, pcov = curve_fit(
                    lambda f_lamb, a, b, c, d: self.fphasenlin(
                        f_lamb, a, b, c, d, r, zt
                    ),
                    ft,
                    angt,
                    p0=p0,
                    sigma=uncertainty_angt,
                    absolute_sigma=False,
                )
                fit_result, xt = self.fphasenlin2(
                    ft, popt[0], popt[1], popt[2], popt[3], r, zt
                )
            except RuntimeError:
                popt = [1.0, 1.0, 1.0, 1.0]
                pcov = [1.0, 1.0, 1.0, 1.0]
                fit_result = 1.0
                xt = 1.0
        else:
            try:
                popt, pcov = curve_fit(
                    lambda f_lamb, a, b, c, d: self.fphasenlin(
                        f_lamb, a, b, c, d, r, zt
                    ),
                    ft,
                    angt,
                    p0=p0,
                )
                fit_result, xt = self.fphasenlin2(
                    ft, popt[0], popt[1], popt[2], popt[3], r, zt
                )
            except RuntimeError:
                popt = [1.0, 1.0, 1.0, 1.0]
                pcov = [1.0, 1.0, 1.0, 1.0]
                fit_result = 1.0
                xt = 1.0
        p0_result = self.fphasenlin(ft, p0[0], p0[1], p0[2], p0[3], r, zt)
        return (popt, pcov, fit_result, p0_result, xt, uncertainty_angt)

    def fitphasenlin_c2(self, f, z, estv, numspan=2, use_bounds=False):
        zinf = (z[0] + z[-1]) / 2.0
        ezinf = zinf / np.abs(zinf)
        z1_1 = z / -ezinf
        ang = phase2(z1_1) + np.angle(-ezinf)
        ft, angt = trimdata(f, ang, estv["f0"], estv["Q"], numspan)
        ft, zt = trimdata(f, z, estv["f0"], estv["Q"], numspan)
        f00g = estv["f0"]
        Qgg = estv["Q"]
        z1_n = np.mean(zt[:5])
        z2_n = np.mean(zt[-5:])
        _v = np.amax(np.abs(zt - z1_n) + np.abs(zt - z2_n))
        idf0 = np.argmax(np.abs(zt - z1_n) + np.abs(zt - z2_n))
        phig = estv["phi"]
        alphag = np.abs(ft[idf0] - f00g) / (2.0 * estv["r"]) ** 2
        if use_bounds:
            bounds = self.calcphasenlinbounds(Qgg, alphag, f00g, phig)
            popt, pcov, fit_result, x0_result, xt, uncertainty_angt = (
                self.fitphasenlintest2(
                    ft, angt, Qgg, alphag, f00g, phig, estv["r"], zt, bounds=bounds
                )
            )
        else:
            popt, pcov, fit_result, x0_result, xt, uncertainty_angt = (
                self.fitphasenlintest2(ft, angt, Qgg, alphag, f00g, phig, estv["r"], zt)
            )
        if np.mean(popt) == 1.0:
            flag = 1
        else:
            flag = 0
        var = np.sum((angt - fit_result) ** 2) / (angt.shape[0] - 1)
        err = np.ones(angt.shape[0]) * np.sqrt(var)
        pherr = angt - fit_result
        _f00 = popt[2]
        return (
            popt,
            pcov,
            pherr,
            fit_result,
            x0_result,
            ft,
            zt,
            angt,
            ang,
            var,
            err,
            xt,
            flag,
            uncertainty_angt,
            ezinf,
        )

    def fits21simnlin(self, use_bounds=False, showplot=False):
        numspan = self.numspan
        result = {}
        result["tau"] = self.tau
        z1 = removecable(self.f, self.z, self.tau, verbose=self.verbose)
        f0g, Qg, idf0, iddf = estpara3(
            self.f, z1, verbose=self.verbose, result0=self.result0
        )
        id1 = np.amax([idf0 - 2 * iddf, 0])
        id2 = np.amin([idf0 + 2 * iddf, len(self.f)])
        if id2 < len(self.f):
            id2 = id2 + 1
        circle_obj = Circlefit(z1[id1:id2])
        result["r"] = float(circle_obj.r)
        result["zc"] = circle_obj.zc
        result["c_residue"] = circle_obj.residue
        estv = {}
        estv["f0"] = f0g
        estv["Q"] = Qg
        estv["r"] = result["r"]
        estv["phi"] = self.result0["phi"]
        theta_zc = np.angle(result["zc"])
        z2 = (result["zc"] - z1) * np.exp(-1j * theta_zc)
        if use_bounds:
            (
                popt,
                pcov,
                pherr,
                fit_result,
                x0_result,
                ft,
                zt,
                angt,
                ang,
                var,
                err,
                xt,
                flag,
                uncertainty_angt,
                ezinf,
            ) = self.fitphasenlin_c2(self.f, z2, estv, numspan, use_bounds=use_bounds)
        else:
            (
                popt,
                pcov,
                pherr,
                fit_result,
                x0_result,
                ft,
                zt,
                angt,
                ang,
                var,
                err,
                xt,
                flag,
                uncertainty_angt,
                ezinf,
            ) = self.fitphasenlin_c2(self.f, z2, estv, numspan)
        result["popt"] = popt
        result["Q"] = popt[0]
        result["alpha"] = popt[1]
        result["f0"] = popt[2]
        result["phi"] = popt[3]
        result["pcov"] = pcov
        result["pherr"] = pherr
        result["popt_err"] = np.sqrt(np.diag(pcov))
        result["fit_result"] = fit_result
        result["fit_result2"] = self.fphasenlin(
            self.f,
            result["Q"],
            result["alpha"],
            result["f0"],
            result["phi"],
            result["r"],
            z2,
        )
        result["fit_resultt"] = self.fphasenlin(
            ft,
            result["Q"],
            result["alpha"],
            result["f0"],
            result["phi"],
            result["r"],
            zt,
        )
        result["x0_result"] = x0_result
        result["ft"] = ft
        result["zt"] = zt
        result["ang"] = ang
        result["angt"] = angt
        result["uncertainty_angt"] = uncertainty_angt
        result["ezinf"] = ezinf
        result["fit_var"] = var
        result["fit_err"] = err
        result["x"] = xt
        idft1 = find_nearest_indice(self.f, ft[0])
        widft = len(ft)
        zt_n = self.z[idft1 : idft1 + widft]
        ft_n = self.f[idft1 : idft1 + widft]
        result["zt_n"] = zt_n
        result["ft_n"] = ft_n
        result["zd"] = (
            -result["zc"]
            / np.abs(result["zc"])
            * np.exp(1j * result["phi"])
            * result["r"]
            * 2
        )
        result["zinf"] = result["zc"] - result["zd"] / 2.0
        Qc_part = np.abs(
            np.abs(result["zc"]) / (2.0 * result["r"] * np.exp(1j * result["phi"]))
            + 0.5
        )
        result["Qc"] = Qc_part * result["Q"]
        result["Qi"] = 1.0 / (1.0 / result["Q"] - 1.0 / result["Qc"])
        result["bif"] = (
            result["alpha"] * (2.0 * result["r"]) ** 2 / (result["f0"] / result["Q"])
        )
        result["dff0"] = result["alpha"] * (2.0 * result["r"]) ** 2 / result["f0"]
        result["zf0"] = result["zinf"] + result["zd"]
        result["bif_err"] = np.sqrt(
            (4.0 * result["r"] ** 2 * result["Q"] / result["f0"]) ** 2
            * result["popt_err"][1] ** 2
            + (
                -4.0
                * result["r"] ** 2
                * result["alpha"]
                * result["Q"]
                / result["f0"] ** 2
            )
            ** 2
            * result["popt_err"][2] ** 2
            + (4.0 * result["r"] ** 2 * result["alpha"] / result["f0"]) ** 2
            * result["popt_err"][0] ** 2
        )
        result["Qc_err"] = np.sqrt(
            (
                np.abs(
                    np.abs(result["zc"])
                    * -1j
                    / (2.0 * result["r"] * np.exp(1j * result["phi"]))
                )
                * result["Q"]
            )
            ** 2
            * result["popt_err"][3] ** 2
            + np.abs(
                np.abs(result["zc"]) / (2.0 * result["r"] * np.exp(1j * result["phi"]))
                + 0.5
            )
            ** 2
            * result["popt_err"][0] ** 2
        )
        result["Qi_err"] = np.sqrt(
            (Qc_part / (Qc_part - 1.0)) ** 2 * result["popt_err"][0] ** 2
            + (
                -1.0
                / (1.0 / result["Q"] - 1.0 / (Qc_part * result["Q"])) ** 2
                * (1.0 / (Qc_part**2 * result["Q"]))
                * np.abs(
                    -1j
                    * np.abs(result["zc"])
                    / (2.0 * result["r"] * np.exp(1j * result["phi"]))
                )
            )
            ** 2
            * result["popt_err"][3] ** 2
        )
        result["zd_err"] = np.sqrt(
            (
                -result["zc"]
                / np.abs(result["zc"])
                * 2.0
                * result["r"]
                * 1j
                * np.exp(1j * result["phi"])
            )
            ** 2
            * result["popt_err"][3] ** 2
        )
        result["zinf_err"] = np.sqrt(
            (
                result["zc"]
                / np.abs(result["zc"])
                * 1j
                * np.exp(1j * result["phi"])
                * result["r"]
            )
            ** 2
            * result["popt_err"][3] ** 2
        )
        l = (
            np.real(result["zf0"] * np.conj(result["zinf"]))
            / np.abs(result["zinf"]) ** 2
        )
        result["l"] = l
        result["l_err"] = np.sqrt(
            np.real(result["zc"] / result["zinf"] ** 2) ** 2
            * np.abs(result["zd_err"]) ** 2
        )
        result["Qi0"] = result["Q"] / l
        result["Qi0_err"] = np.sqrt(
            result["popt_err"][0] ** 2 / result["l"] ** 2
            + result["Q"] ** 2 / result["l"] ** 4 * result["l_err"] ** 2
        )
        result["f00id"] = find_nearest_indice(self.f, result["f0"])
        result["f0_corr"] = result["f0"] - result["alpha"] * (2.0 * estv["r"]) ** 2
        result["f0id"] = find_nearest_indice(self.f, result["f0_corr"])
        if flag == 0:
            if (
                sum(np.abs(result["pherr"]) > self.pherr_threshold)
                > self.pherr_threshold_num
            ):
                flag = 2
        result["flag"] = flag
        iq_model = 1.0 - result["Q"] / result["Qc"] * np.exp(1j * result["phi"]) / (
            1.0 + 2j * result["Q"] * (self.f / result["f0_corr"] - 1.0)
        )
        iq_modelt = 1.0 - result["Q"] / result["Qc"] * np.exp(1j * result["phi"]) / (
            1.0 + 2j * result["Q"] * (result["ft"] / result["f0_corr"] - 1.0)
        )
        result["iq_model"] = iq_model
        result["iq_modelt"] = iq_modelt
        result["A_mag"] = result["r"]
        result["A_magt"] = result["r"]
        ang_to_iq = result["A_mag"] * (
            np.cos(result["fit_result2"] - np.angle(-result["ezinf"]))
            + 1j * np.sin(result["fit_result2"] - np.angle(-result["ezinf"]))
        )
        angt_to_iq = result["A_magt"] * (
            np.cos(result["fit_resultt"] - np.angle(-result["ezinf"]))
            + 1j * np.sin(result["fit_resultt"] - np.angle(-result["ezinf"]))
        )
        result["ang_to_z2"] = ang_to_iq * -result["ezinf"]
        result["angt_to_z2"] = angt_to_iq * -result["ezinf"]
        result["ang_to_z1"] = result["zc"] - result["ang_to_z2"] / np.exp(
            -1j * np.angle(result["zc"])
        )
        result["angt_to_z1"] = result["zc"] - result["angt_to_z2"] / np.exp(
            -1j * np.angle(result["zc"])
        )
        result["ang_to_z"] = result["ang_to_z1"] / np.exp(
            -1j * 2.0 * np.pi * (self.f / 1000000000.0) * self.tau
        )
        result["angt_to_z"] = result["angt_to_z1"] / np.exp(
            -1j * 2.0 * np.pi * (result["ft"] / 1000000000.0) * self.tau
        )
        return result
