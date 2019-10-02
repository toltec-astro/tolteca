#! /usr/bin/env python

import matplotlib.colors as mc
import numpy as np
import re


class Palette(object):

    black = "#000000"
    white = "#ffffff"
    blue = "#73cef4"
    green = "#bdffbf"
    orange = "#ffa500"
    purple = "#af00ff"
    red = "#ff6666"
    yellow = "#ffffa0"

    @staticmethod
    def _is_hex(c):
        m = re.match("^#(?:[0-9a-fA-F]{3}){1,2}$", c)
        return m is not None

    @classmethod
    def _color(self, c):
        if isinstance(c, str):
            if self._is_hex(c):
                return c
            if hasattr(self, c) and self._is_hex(getattr(self, c)):
                return getattr(self, c)
        if isinstance(c, np.ndarray) and len(c) in (3, 4):
            if c.dtype in (np.float_, np.double):
                return c
            else:
                return c / 255.
        raise ValueError(f"unknown color {c}")

    def rgb(self, c):
        return np.array(mc.to_rgb(self._color(c)))

    def irgb(self, c):
        return (self.rgb(c) * 255.).astype(np.uint8)

    def hex(self, c):
        return mc.to_hex(self._color(c))

    def blend(self, c1, c2, a, fmt='hex'):
        c1 = self.rgb(c1)
        c2 = self.rgb(c2)
        c = c1 * (1. - a) + c2 * a
        return getattr(self, fmt)(c)
