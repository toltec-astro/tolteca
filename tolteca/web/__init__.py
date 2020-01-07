# import prance
# from pathlib import Path
# from flask import Flask

import connexion
# from . import config
from tolteca.utils.log import timeit
from tolteca.utils.cli.click_log import init as init_log


@timeit
def create_server(default_config=None):
    app = connexion.FlaskApp(
            __name__,
            specification_dir='backend/api',
            )
    app.add_api(
            'main.yaml',
            resolver=connexion.RestyResolver('tolteca.web.backend.api'))

    # def get_bundled_specs(main_file):
    #     parser = prance.ResolvingParser(
    #              main_file.absolute().as_posix(),
    #              lazy=True,
    #              backend='openapi-spec-validator')
    #     parser.parse()
    #     return parser.specification

    # app.add_api(get_bundled_specs(
    #     Path(__file__).parent.joinpath("backend/api/main.yaml")))

    # get flask server
    server = app.app
    server.connexion_app = app
    server.config.from_object(default_config)
    # server.config.from_object(config)
    init_log(level='DEBUG' if server.debug else 'INFO')
    return server
