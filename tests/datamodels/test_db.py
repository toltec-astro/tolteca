from tolteca_datamodels.db.core import (
    DB,
    DPDB_BIND_NAME,
    DBConfig,
    current_dpdb,
    db_context,
)


def test_db_config():
    dbc = DBConfig.model_validate({})
    assert dbc.binds[0].name == DPDB_BIND_NAME

    dbc = DBConfig.model_validate(
        {
            "binds": [
                {
                    "name": "test1",
                    "url": "mysql://test1",
                },
                {
                    "name": "test2",
                    "url": "sqlite:///test2",
                },
                {
                    "name": "dpdb",
                    "url": "sqlite:///test3",
                },
            ],
        },
    )
    assert dbc.binds[0].name == "test1"


def test_db():
    db = DB()
    assert db.config.dpdb.name == DPDB_BIND_NAME


def test_db_context():
    assert not current_dpdb.proxy_initialized()
    with db_context():
        assert current_dpdb.engine.url.render_as_string() == "sqlite://"
    assert not current_dpdb.proxy_initialized()
