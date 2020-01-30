#! /usr/bin/env python
import numpy as np
# from astropy import units as u
from astropy.modeling.mappings import Identity
from astropy.modeling import fix_inputs
from . import (
        # ResonanceCircleComplex,
        ResonanceCircleInv,
        ResonanceCircleComplex,
        ResonanceCircleComplexInv,
        ResonanceCircleSweepComplex,
        ResonanceCircleProbeComplex,
        ReadoutIQToComplex,
        OpticalDetune,
        # InstrumentalDetune,
        ResonanceCircleQrInv
        )
from astropy import log


class KidsSimulator(object):
    """Class that make simulated kids data."""

    def __init__(self, fr=None, Qr=None,
                 background=None, responsivity=None):
        self._Qr = Qr
        self._fr = fr
        self._background = background
        self._responsivity = responsivity

        # make models
        m_r = ResonanceCircleQrInv()
        m_x = Identity(1)
        m_p_probe = OpticalDetune(
                background=self._background,
                responsivity=self._responsivity)
        m_iq = (ReadoutIQToComplex() | ResonanceCircleComplex())

        self._x_sweep = fix_inputs((m_r & m_x) | m_iq, {
            'Qr': self._Qr
            })
        self._f_sweep = ResonanceCircleSweepComplex(fr=self._fr, Qr=self._Qr)
        self._x_probe = self._x_sweep
        self._f_probe = ResonanceCircleProbeComplex()
        self._Qr2r = m_r
        self._p2x = m_p_probe
        self._p_probe = (m_r & m_p_probe) | m_iq
        self._iq2rx = ResonanceCircleInv()
        self._iq2rxcomplex = ResonanceCircleComplexInv()

        m_info = ['summary of kids simulator models:', ]
        sep = '-*' * 40
        m_info.append(f"{sep}\nx sweep:\n{self._x_sweep}")
        m_info.append(f"{sep}\nf sweep:\n{self._f_sweep}")
        m_info.append(f"{sep}\nx probe:\n{self._x_probe}")
        m_info.append(f"{sep}\nf_probe:\n{self._f_probe}")
        m_info.append(f"{sep}\np_probe:\n{self._p_probe}")
        m_info.append(f"{sep}")
        log.info('\n'.join(m_info))

    @property
    def fwhm_x(self):
        """Return the resonance FWHM in unit of x."""
        return 1. / self._Qr

    @property
    def fwhm_f(self):
        """Return the resonance FWHM in unit of Hz."""
        return self.fwhm_x * self._fr

    def sweep_x(self, n_steps=None, n_fwhms=None, xlim=None):
        """Return a resonance circle sweep."""
        if n_fwhms is None and xlim is None:
            raise ValueError("n_fwhms or xlim is required.")
        if xlim is None:
            xlim = (-0.5 * n_fwhms * self.fwhm_x,
                    0.5 * n_fwhms * self.fwhm_x)
        # get grid
        xs = np.linspace(*xlim, n_steps)
        iqs = self._x_sweep(xs)
        return xs, iqs

    def probe_p(self, pwrs):
        """Return detector response for given optical power."""
        # self._p2x.background = background
        # self._p2x.responsivity = responsivity
        rs = np.full(pwrs.shape, self._Qr2r(self._Qr))
        xs = self._p2x(pwrs)
        iqs = self._x_probe(xs)
        return rs, xs, iqs

    def solve_x(self, *args):
        """Return x for given detector response."""
        if len(args) == 1:
            # complex
            return self._iq2rxcomplex(args[0])
        return self._iq2rx(*args)
