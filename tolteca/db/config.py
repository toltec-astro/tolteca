#! /usr/bin/env python
from ..utils import get_pkg_data_path

DB_CONFIG = {
        'lmt_toltec': {
            'uri': 'mysql+mysqldb://tol:tirra@clipy:3306',
#            'uri': 'mysql+mysqldb://tol:tirra@127.0.0.1:3307',
            'tables_from_reflection': True
        },
        'debug_b': {
            'uri': f'sqlite+pysqlite:///'
                   f'{get_pkg_data_path().joinpath("debug.sqlite")}',
            'schema': 'toltecdatadb',
        }
    }
