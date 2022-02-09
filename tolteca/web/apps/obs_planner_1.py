import os
import dash_bootstrap_components as dbc

from .. import env_mapper
from .. site_config import db_config
from ...cli.web import web_config

if web_config:
    # invoked via tolteca web CLI interface
    # TODO implement running app via tolteca web CLI
    raise NotImplementedError
else:
    # invoked directly with dasha. We use the default config loader
    # to load some predefined envs from any env file
    from ...utils import default_config_loader
    config_loader = default_config_loader
    env = config_loader.get_env()
    import pdb
    pdb.set_trace()
    # update os env
    os.environ.update(env)


#

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
                'site_name'
                },
            },
        ]
    }
