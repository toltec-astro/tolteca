"""
Note that in Python, `a[1]` is equivalent to `a[(1,)]`.
"""

import pytest

from advindexparser import parse_slice


def test_parse_slice():
    with pytest.raises(ValueError):
        parse_slice('')
    with pytest.raises(ValueError):
        parse_slice('  ')
    assert parse_slice(':') == (slice(None, None, None),)
    assert parse_slice('1') == (1,)
    assert parse_slice('1,') == (1,)
    assert parse_slice('[1]') == ([1],)
    assert parse_slice('[1,]') == ([1],)
    with pytest.raises(ValueError):
        parse_slice('1,,')
    assert parse_slice('1,2') == (1, 2)
    assert parse_slice('1:,2') == (slice(1, None, None), 2)
    assert parse_slice('1::,2') == (slice(1, None, None), 2)
    assert parse_slice(':2:-1,2,...') == (slice(None, 2, -1), 2, ...)
    assert parse_slice('::,[1,2,-3],..., (5,7)') == (slice(None, None, None),
                                                     [1, 2, -3], ..., [5, 7])
