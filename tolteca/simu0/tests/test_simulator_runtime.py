#! /usr/bin/env python

from .. import SimulatorRuntime


def test_simulator_runtime_none_persistent():

    from tolteca.simu import example_configs as cfgs
    cfg = cfgs['toltec_point_source']

    rc = SimulatorRuntime(config=cfg)

    assert rc.config['simu']['jobkey'] == cfg['simu']['jobkey']

    # check update config
    rc.update({
        'simu': {
            'jobkey': 'test_update'
            }
        }, overwrite=True)
    assert rc.config['simu']['jobkey'] != cfg['simu']['jobkey']
    assert rc.config['simu']['jobkey'] == 'test_update'
    simobj = rc.get_instrument_simulator()

    assert simobj.table
