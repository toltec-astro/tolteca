#! /usr/bin/env python

"""Flask entry point."""

if __package__:
    from . import create_app  # noqa: F401
else:
    # this is to work around the connexion api resolver problem
    from tolteca.web import create_app  # noqa: F401


def decode_token(*args):
    print(*args)
    return "abc"
