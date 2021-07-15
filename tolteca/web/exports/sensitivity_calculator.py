from .. import cache_config
import dash_bootstrap_components as dbc

_dasha_ext_module_parent = 'dasha.web.extensions'
extensions = [
    {
        'module': f'{_dasha_ext_module_parent}.cache',
        'config': cache_config,
        },
    {
        'module': f'{_dasha_ext_module_parent}.dasha',
        'config': {
            'template': 'tolteca.web.templates.sensitivity_calculator',
            'fluid': True,
            'className': 'my-2',
            'title_text': "Sensitivity Calculator",
            'EXTERNAL_STYLESHEETS': [
                dbc.themes.MATERIA,
                # dbc.themes.YETI,
                # dbc.themes.BOOTSTRAP,
                ],
            'ASSETS_IGNORE': 'bootstrap.*'
            },
        },
    ]
