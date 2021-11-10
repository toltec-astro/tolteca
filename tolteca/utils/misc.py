#!/usr/bin/env python


import appdirs
from pathlib import Path
from art import text2art


__all__ = [
    'get_pkg_data_path', 'get_user_data_dir', 'get_nested_keys']


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


def get_nested_keys(data):
    """Return key list for nested structure.

    See: https://github.com/ducdetronquito/scalpl/issues/21#issue-700007850
    """
    _null_key = object()

    def nested_key(data):
        if isinstance(data, list):
            keys = []
            for i in range(len(data)):
                result = nested_key(data[i])
                if isinstance(result, list):
                    if isinstance(data[i], dict):
                        keys.extend(["[%d].%s" % (i, item) for item in result])
                    elif isinstance(data[i], list):
                        keys.extend(["[%d]%s" % (i, item) for item in result])
                elif result is _null_key:
                    keys.append("[%d]" % i)
            return keys
        elif isinstance(data, dict):
            keys = []
            for key, value in data.items():
                result = nested_key(value)
                if isinstance(result, list):
                    if isinstance(value, dict):
                        keys.extend(["%s.%s" % (key, item) for item in result])
                    elif isinstance(value, list):
                        keys.extend(["%s%s" % (key, item) for item in result])
                elif result is _null_key:
                    keys.append("%s" % key)
            return keys
        else:
            return _null_key
    keys = nested_key(data)
    if keys is not _null_key:
        return keys
    raise ValueError("invalid data type")
