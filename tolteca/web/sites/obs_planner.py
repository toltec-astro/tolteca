import dash_bootstrap_components as dbc

DASHA_SITE = {
    'extensions': [
        {
            'module': 'dasha.web.extensions.dasha',
            'config': {
                'template': 'tolteca.web.templates.obs_planner:ObsPlanner',
                'EXTERNAL_STYLESHEETS': [
                    dbc.themes.MATERIA,
                    ],
                'ASSETS_IGNORE': 'bootstrap.*'
                },
            },
        ]
    }
