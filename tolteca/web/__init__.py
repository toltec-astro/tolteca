#! /usr/bin/env python

"""DashA site for TolTECA."""

from tollan.utils.env import env_registry
from tolteca.datamodels.fs.toltec import ToltecDataFileStore
from ..utils import get_user_data_dir
import dash_bootstrap_components as dbc


env_prefix = 'TOLTECA'

env_registry.register(
        f"{env_prefix}_DB_TOLTEC_URL",
        "The TolTEC database url.",
        'mysql+mysqldb://localhost:3306')
env_registry.register(
        f"{env_prefix}_DB_TOLTEC_USERLOG_TOOL_URL",
        "The TolTEC database url for user log tool.",
        'mysql+mysqldb://localhost:3306')
env_registry.register(
        f"{env_prefix}_DB_TOLTECA_URL",
        "The TolTECA database url",
        'mysql+mysqldb://localhost:3306')
env_registry.register(
        f"{env_prefix}_FS_TOLTEC_ROOTPATH",
        "The root path to TolTEC data files",
        get_user_data_dir())
env_registry.register(
        f"{env_prefix}_LMT_OCS3_URL",
        "The OCS3_URL",
        'socket://localhost:61559')
env_registry.register(
        f"{env_prefix}_FS_TOLTEC_HK_ROOTPATH",
        "The root path to TolTEC housekeeping data files",
        get_user_data_dir())

# resource providers
db_toltec_url = env_registry.get(f"{env_prefix}_DB_TOLTEC_URL")
db_toltec_userlog_tool_url = env_registry.get(
        f"{env_prefix}_DB_TOLTEC_USERLOG_TOOL_URL")
db_tolteca_url = env_registry.get(f"{env_prefix}_DB_TOLTECA_URL")
redis_url = "redis://localhost:6379"
fs_toltec_rootpath = env_registry.get(f"{env_prefix}_FS_TOLTEC_ROOTPATH")
fs_toltec_hk_rootpath = env_registry.get(f"{env_prefix}_FS_TOLTEC_HK_ROOTPATH")
lmt_ocs3_url = env_registry.get(f"{env_prefix}_LMT_OCS3_URL")
toltec_datastore = ToltecDataFileStore(fs_toltec_rootpath)


# site configs
db_config = {
        "SQLALCHEMY_TRACK_MODIFICATIONS": False,
        "SQLALCHEMY_BINDS": {
            'toltec': db_toltec_url,
            'toltec_userlog_tool': db_toltec_userlog_tool_url,
            'tolteca': db_tolteca_url,
            'default': db_tolteca_url,
            }
        }
cache_config = {
        "CACHE_TYPE": 'redis',
        "CACHE_DEFAULT_TIMEOUT": 60 * 5,  # second
        "CACHE_KEY_PREFIX": 'tolteca_',
        "CACHE_REDIS_URL": f"{redis_url}/0",
        }
ipc_config = {
        'backends': {
            'rejson': {
                'url': f"{redis_url}/1",
                },
            }
        }
celery_config = {
        "CELERY_RESULT_BACKEND": f"{redis_url}/7",
        "CELERY_RESULT_EXPIRES": 24 * 60 * 60,  # second  == 1day
        "CELERY_BROKER_URL": f"{redis_url}/7",
        }
dasha_config = {
        "title_text": "TolTECA",
        "template": "dasha.web.templates.slapdash",
        'EXTERNAL_STYLESHEETS': [
            # dbc.themes.MATERIA,
            # dbc.themes.YETI,
            dbc.themes.BOOTSTRAP,
            ],
        'ASSETS_IGNORE': 'bootstrap.*'
        }


# tolteca payloads
# from .tasks import toltecdb  # noqa: E402


def load_task_modules():
    # from .tasks import kidsview  # noqa: F401
    from .tasks import kidsreduce  # noqa: F401
    from .tasks import ocs3  # noqa: F401


celery_config.update(
        # tasks=toltecdb.tasks,
        post_init_app=load_task_modules
        )

dasha_config.update(
        pages=[
                {
                    "template": "tolteca.web.templates.toltecdashboard",
                    "route_name": "toltecdashboard",
                    'title_text': "Dashboard",
                    'title_icon': 'fas fa-tachometer-alt',
                },
                {
                    "template": "tolteca.web.templates.logview",
                    "route_name": "logview",
                    "title_text": 'Log View',
                    'title_icon': 'far fa-sticky-note',
                },
                {
                    "template": "tolteca.web.templates.beammap",
                    "route_name": "beammap",
                    'title_text': "Beammap",
                    'title_icon': 'fas fa-layer-group',
                },
                {
                    "template": "tolteca.web.templates.dichroic",
                    "route_name": "dichroic",
                    "title_text": 'Dichro Temp',
                    'title_icon': 'fas fa-thermometer-half',
                },
                {
                    "template": "tolteca.web.templates.hkview",
                    "route_name": "hkview",
                    "title_text": 'HK View',
                    'title_icon': 'fas fa-thermometer-half',
                },
                {
                    "template": "tolteca.web.templates.kids_explorer",
                    "route_name": "kids_explorer",
                    "title_text": 'KIDs Explorer',
                    'title_icon': 'fas fa-icicles',
                },
                {
                    "template": "tolteca.web.templates.vna_explorer",
                    "route_name": "vna_explorer",
                    "title_text": 'VNA Explorer',
                    'title_icon': 'fas fa-icicles',
                },
                {
                    "template": "tolteca.web.templates.tune_explorer",
                    "route_name": "tune_explorer",
                    "title_text": 'TUNE Explorer',
                    'title_icon': 'fas fa-icicles',
                },
                {
                    "template": "tolteca.web.templates.noise_explorer",
                    "route_name": "noise_explorer",
                    "title_text": 'Noise Explorer',
                    'title_icon': 'fas fa-wave-square',
                },
                {
                    "template": "tolteca.web.templates.toltecdb",
                    "route_name": "toltecdb",
                    'title_text': "(dbg) Databases",
                    'title_icon': 'fas fa-table',
                },
                {
                    "template": "tolteca.web.templates.taskview",
                    "route_name": "taskview",
                    "title_text": '(dbg) Beat Schedule',
                    "title_icon": 'fas fa-tasks'
                },
                {
                    "template": "tolteca.web.templates.kidsreduceview",
                    "route_name": "kidsreduce",
                    "title_text": '(dbg) TRS Info',
                },
            ],
        )


# site runtime
_dasha_ext_module_parent = 'dasha.web.extensions'
extensions = [
    {
        'module': f'{_dasha_ext_module_parent}.db',
        'config': db_config,
        },
    {
        'module': f'{_dasha_ext_module_parent}.cache',
        'config': cache_config,
        },
    {
        'module': f"{_dasha_ext_module_parent}.ipc",
        'config': ipc_config
        },
    {
        'module': f"{_dasha_ext_module_parent}.celery",
        'config': celery_config
        },
    {
        'module': f'{_dasha_ext_module_parent}.dasha',
        'config': dasha_config,
        },
    ]

# from .defaults import create_server  # noqa: F401
