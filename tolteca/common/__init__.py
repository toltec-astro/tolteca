#!/usr/bin/env python


def get_instru_info(instru):
    """Return assort info dict for given instrument."""

    if instru == 'toltec':
        from .toltec import info
        return info
    if instru == 'lmt':
        from .lmt import info
        return info
    raise ValueError(f"invalid instrument key {instru}")
