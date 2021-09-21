#!/usr/bin/env python


import appdirs
from pathlib import Path
from art import text2art


__all__ = [
    'get_pkg_data_path', 'get_user_data_dir', ]


def get_pkg_data_path():
    """Return the package data path."""
    return Path(__file__).parent.parent.joinpath("data")


def get_user_data_dir():
    return Path(appdirs.user_data_dir('tolteca', 'toltec'))


def make_ascii_banner(title, subtitle):
    """Return a textual banner."""
    title_lines = text2art(title, "avatar").split('\n')
    pad = 4
    width = len(title_lines[0]) + pad * 2

    st_line = f'{{:^{width}s}}'.format(subtitle)

    return '\n{}\n{}\n{}\n{}\n'.format(
        '.~' * (width // 2),
        '\n'.join([f'{" " * pad}{title_line}' for title_line in title_lines]),
        st_line,
        '~.' * (width // 2)
        )
