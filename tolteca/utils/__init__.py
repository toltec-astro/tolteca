#!/usr/bin/env python

from pathlib import Path


def get_pkg_data_path():
    return Path(__file__).parent.parent.joinpath("data")
