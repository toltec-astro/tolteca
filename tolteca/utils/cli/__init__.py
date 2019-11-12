#! /usr/bin/env python

from ... import version, PROGNAME, DESCRIPTION


def cli_header():
    return (f"{PROGNAME} v{version.version} {version.timestamp}"
            f" - {DESCRIPTION}")
