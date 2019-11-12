#! /usr/bin/env python

from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


def ClassFactory(name, tableName, BaseClass=db.Base, fks=None):
    tableArgs = [{'autoload': True, 'schema': 'mangadapdb'}]
    if fks:
        for fk in fks:
            tableArgs.insert(0, ForeignKeyConstraint([fk[0]], [fk[1]]))

    newclass = type(
        name, (BaseClass,),
        {'__tablename__': tableName,
         '__table_args__': tuple(tableArgs)})


class DebugObject(Base):
    pass



from __future__ import division, print_function

import re

import marvin.db.models.DataModelClasses as datadb
import numpy as np
from astropy.io import fits
from marvin.core.caching_query import RelationshipCache
from marvin.db.database import db
from marvin.utils.datamodel.dap import datamodel
from sqlalchemy import Float, ForeignKeyConstraint, and_, case, cast, select
from sqlalchemy.engine import reflection
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.inspection import inspect
from sqlalchemy.orm import configure_mappers, relationship
from sqlalchemy.schema import Column
from sqlalchemy.types import Integer


def cameliseClassname(tableName):
    """Produce a camelised class name."""

    return str(tableName[0].upper() +
               re.sub(r'_([a-z])',
               lambda m: m.group(1).upper(), tableName[1:]))



