# import prance
# from pathlib import Path
# from flask import Flask

import connexion
from . import config
from tolteca.utils.log import timeit


def create_app():

    @timeit
    def setup_backend():
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

        server.config.from_object(config)

        # server db backend
        from . import backend
        backend.init_app(server)
        return server

    server = setup_backend()

    # dash/plotly app
    from . import frontend
    frontend.init_app(server)
    return server
