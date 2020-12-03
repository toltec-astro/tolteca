#! /usr/bin/env python

import dash_bootstrap_components as dbc


# cache_config = {
#         "CACHE_TYPE": 'redis',
#         "CACHE_DEFAULT_TIMEOUT": 60 * 5,  # second
#         "CACHE_KEY_PREFIX": 'tolteca_simu_',
#         "CACHE_REDIS_URL": f"{redis_url}/0",
#         }
# ipc_config = {
#         'backends': {
#             'rejson': {
#                 'url': f"{redis_url}/1",
#                 },
#             }
#         }
dasha_config = {
        "title_text": "TolTEC Simulator",
        "template": "tolteca.simu.toltec.web.main",
        'EXTERNAL_STYLESHEETS': [
            # dbc.themes.MATERIA,
            # dbc.themes.YETI,
            dbc.themes.BOOTSTRAP,
            ],
        'ASSETS_IGNORE': 'bootstrap.*'
        }


# site runtime
_dasha_ext_module_parent = 'dasha.web.extensions'
extensions = [
    # {
    #     'module': f'{_dasha_ext_module_parent}.cache',
    #     'config': cache_config,
    #     },
    # {
    #     'module': f"{_dasha_ext_module_parent}.ipc",
    #     'config': ipc_config
    #     },
    {
        'module': f'{_dasha_ext_module_parent}.dasha',
        'config': dasha_config,
        },
    ]
