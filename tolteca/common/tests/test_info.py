#!/usr/bin/env python

import pytest
from .. import get_instru_info


def test_get_instru_info():
    info = get_instru_info('toltec')

    assert info['instru'] == 'toltec'
    assert info['name'] == 'TolTEC'

    info = get_instru_info('lmt')

    assert info['instru'] == 'lmt'
    assert info['name'] == 'LMT'

    with pytest.raises(ValueError, match='invalid instru'):
        info = get_instru_info('some_instru')
