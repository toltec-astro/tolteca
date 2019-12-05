#! /usr/bin/env python
"""This module provide automapped models from databases."""

from tolteca.utils.registry import Registry
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.ext.declarative import declared_attr


class AutoTableNameBase(object):

    @declared_attr
    def __tablename__(cls):
        return cls.__name__.lower()


class AutomapModelRegistry(Registry):

    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self._self_Base = automap_base(cls=AutoTableNameBase)

    @property
    def Base(self):
        return self._self_Base

    def load_models(self, db):

        logger = get_logger()

        _pluralizer = inflect.engine()

        def pluralize_collection(base, local_cls, referred_cls, constraint):

            referred_name = referred_cls.__tablename__
            pluralized = _pluralizer.plural(referred_name)
            return pluralized

        try:
            Base.prepare(
                    db.engine, reflect=True,
                    name_for_collection_relationship=pluralize_collection)
        except Exception as e:
            logger.error(f"unable to create model classes: {e}")
        else:
            # collect all models
            Base.classes.update(_models)
            db.Base = Base
            # db.metadata = Base.metadata
            # db.metadata.reflect(db.engine)
            # logger.debug(f"tables {pformat_dict(db.metadata.tables)}")
            logger.debug(f"model classes {pformat_dict(db.Base.classes._data)}")
            # logger.debug(f"model classes {pformat_dict(db.Base.classes._data)}")
