from pathlib import Path

import appdirs

__all__ = ["get_pkg_data_dir", "get_user_data_dir"]


def get_pkg_data_dir():
    """Return the package data path."""
    return Path(__file__).parent.parent.joinpath("data")


def get_user_data_dir():
    """Return the user space data path."""
    return Path(appdirs.user_data_dir("tolteca", "toltec"))
