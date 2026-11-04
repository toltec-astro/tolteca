"""Instrument-independent data-model infrastructure.

Core imports do not load instrument registrations. Registry-backed APIs still
use available registrations, including those loaded after the core import.
Use the root entry-point modules to enable built-in LMT and TolTEC support.
"""
