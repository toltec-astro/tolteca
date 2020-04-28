#! /usr/bin/env python

"""DashA site for TolTECA."""

from tollan.utils.env import env_registry
from pathlib import Path
from tolteca.fs.toltec import ToltecDataFileStore
from ..utils import get_user_data_dir


env_registry.register(
        "TOLTECA_TOLTEC_DB_URL", "The toltec database url",
        'mysql+mysqldb://localhost:3306')
env_registry.register(
        "TOLTECA_TOLTEC_DATA_ROOTPATH",
        "The root path to toltec data files",
        get_user_data_dir())

# service provider settings
tolteca_app_db_filename = 'tolteca_app_db.sqlite'
tolteca_app_db_path = Path(__file__).with_name(
        tolteca_app_db_filename).resolve().as_posix()
tolteca_toltec_db_url = env_registry.get("TOLTECA_TOLTEC_DB_URL")
tolteca_redis_url = "redis://localhost:6379"
tolteca_toltec_data_rootpath = env_registry.get("TOLTECA_TOLTEC_DATA_ROOTPATH")
tolteca_toltec_datastore = ToltecDataFileStore(tolteca_toltec_data_rootpath)


# site configs
db_config = {
        "SQLALCHEMY_TRACK_MODIFICATIONS": False,
        "SQLALCHEMY_BINDS": {
            'appdb': f"sqlite:////{tolteca_app_db_path}",
            'toltecdb': tolteca_toltec_db_url
            }
        }
cache_config = {
        "CACHE_TYPE": 'redis',
        "CACHE_DEFAULT_TIMEOUT": 60 * 5,  # second
        "CACHE_KEY_PREFIX": 'tolteca_',
        "CACHE_REDIS_URL": f"{tolteca_redis_url}/0",
        }
ipc_config = {
        'backends': {
            'rejson': {
                'url': f"{tolteca_redis_url}/1",
                },
            'redis': {
                'url': f"{tolteca_redis_url}/2",
                },
            'cache': {}
            }
        }
celery_config = {
        "CELERY_RESULT_BACKEND": f"{tolteca_redis_url}/3",
        "CELERY_RESULT_EXPIRES": 0,  # second
        "CELERY_BROKER_URL": f"{tolteca_redis_url}/3",
        }
dasha_config = {
        "TITLE": "TolTECA",
        "template": "dasha.web.templates.slapdash",
        }


# tolteca payloads
from .tasks import toltecdb  # noqa: E402


def load_task_modules():
    from .tasks import kidsview  # noqa: F401
    from .tasks import kidsreduce  # noqa: F401


celery_config.update(
        tasks=toltecdb.tasks,
        post_init_app=load_task_modules
        )

dasha_config.update(
        pages=[
                {
                    "template": "dasha.web.templates.viewgrid",
                    "route_name": "toltecdb",
                    "views": toltecdb.views,
                    'title_text': "Databases",
                    'title_icon': 'fas fa-table',
                },
                {
                    "template": "tolteca.web.templates.kidsview",
                    "route_name": "kidsview",
                    'title_text': "Kids View",
                    'title_icon': 'fas fa-chart-line',
                    'update_interval': 1000.,
                },
                {
                    "template": "tolteca.web.templates.taskview",
                    "route_name": "taskview",
                },
                {
                    "template": "tolteca.web.templates.kidsreduceview",
                    "route_name": "kidsreduce",
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
