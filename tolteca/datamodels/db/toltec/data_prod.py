#! /usr/bin/env python

import re
from tollan.utils.log import get_logger, logit, timeit
import sqlalchemy as sa
import sqlalchemy.sql.expression as se
from sqlalchemy.ext.automap import automap_base
from tollan.utils import odict_from_list
from tollan.utils.fmt import pformat_dict
from tollan.utils.db import conventions as c
from tollan.utils.db import TableDefList

from ...toltec.enums import RawObsType, RawObsMaster


__all__ = ['init_db', ]


_static_data = {
        'dp_raw_obs_master': [
            {
                'pk': v.value,
                'label': v.name,
                'desc': v.desc
                }
            for v in RawObsMaster
            ],
        'dp_raw_obs_type': [
            {
                'pk': v.value,
                'label': v.name,
                'desc': v.desc
                }
            for v in RawObsType
            ]
        }


def _make_proc_descriptor():

    return [
        sa.Column(
            'module', sa.String(1024), nullable=False,
            comment='The qualified name of the data proc module.'
            ),
        sa.Column(
            'version', sa.String(128), nullable=True,
            comment='The version of the data proc module.'
            ),
        sa.Column(
            'config', sa.JSON(none_as_null=False),
            nullable=False, default="null",
            comment='The config of the data proc module.'
            ),
        ]


def _make_table_defs():
    table_defs = odict_from_list([
        c.table_info_table(),
        c.client_info_table(),
        {
            'name': 'data_prod',
            'desc': 'The data products.',
            'columns': [
                c.pk(),
                c.fk('data_prod_type'),
                sa.Column(
                    'source_url', sa.String(4096), nullable=True,
                    comment='The URL to the data product.'),
                sa.Column(
                    'source', sa.JSON(none_as_null=True),
                    nullable=True,
                    default=None,
                    comment='The data product.'
                    ),
                c.created_at(),
                c.updated_at(),
                c.client_info_fk(),
                sa.CheckConstraint("""
( CASE WHEN source     IS NULL THEN 0 ELSE 1 END
+ CASE WHEN source_url IS NULL THEN 0 ELSE 1 END
) = 1"""),
                ]
            },
        {
            'name': 'data_prod_assoc',
            'desc': 'The associations between data products.',
            'columns': [
                c.pk(),
                c.fk('data_prod_assoc_type'),
                c.fk(
                    'data_prod_assoc_info',
                    comment='The details of this association.'
                    ),
                ]
            },
        {
            'name': 'data_prod_assoc_info',
            'desc': 'The details of the associations.',
            'columns': [
                c.pk(),
                sa.Column(
                    'context', sa.JSON(none_as_null=False),
                    nullable=False,
                    default="null",
                    comment='The contextual info of the association.'
                    ),
                c.created_at(),
                c.updated_at(),
                c.client_info_fk(),
                ]
            },
        {
            'name': 'data_prod_assoc_type',
            'desc': 'The types of of data associations.',
            'columns': [
                c.pk(),
                c.label(),
                c.desc(),
                ],
            'data': []
            },
        {
            'name': 'data_prod_type',
            'desc': 'The types of data products',
            'columns': [
                c.pk(),
                c.label(),
                c.desc(),
                sa.Column('level', sa.Integer, nullable=True),
                ],
            'data': []
            },
        {
            'name': 'content_type',
            'desc': 'A standard set of high level data content types',
            'columns': [
                c.pk(),
                c.label(),
                c.desc(),
                ],
            'data': [
                {'label': 'image', 'desc': 'Image type.'},
                {'label': 'table', 'desc': 'Table type.'},
                {'label': 'object', 'desc': 'Object type.'},
                ]
            },
        {
            'name': 'data_prod_type_content_type',
            'desc': 'Association table between data product types and content'
                    'types.',
            'columns': [
                c.pk(),
                c.fk('data_prod_type'),
                c.fk('content_type'),
                ]
            },
        # concrete data prod tables
        # each data prod table comes with its own association table(s).
        {
            'name': 'dp_raw_obs',
            'desc': 'Raw KIDs detector telemetry data as recorded by the data'
                    ' acquisition system. Each of this data product consists'
                    ' of multiple raw interface data files.',
            'parent': 'data_prod',
            'columns': [
                c.fk('dp_raw_obs_type'),
                c.fk('dp_raw_obs_master'),
                sa.Column(
                    'obsnum', sa.Integer, nullable=False,
                    comment='The id of the obs assigned by master.'
                    ),
                sa.Column(
                    'subobsnum', sa.Integer, nullable=False,
                    comment='The sub-id of the obs assigned by master.'
                    ),
                sa.Column(
                    'scannum', sa.Integer, nullable=False,
                    comment='The sub-sub-id of the obs assigned by master.'
                    ),
                sa.Column(
                    'repeat', sa.Integer, nullable=False,
                    comment='The degree of repeat of the original data stream.'
                    ),
                ]
            },
        {
            'name': 'dp_raw_obs_master',
            'desc': 'The entity that drives the acquisition of raw obs.',
            'columns': [
                c.pk(autoincrement=False),
                c.label(),
                c.desc(),
                ],
            'data': _static_data['dp_raw_obs_master']
            },
        {
            'name': 'dp_raw_obs_type',
            'desc': 'Raw KIDs detector data types.',
            'columns': [
                c.pk(autoincrement=False),
                c.label(),
                c.desc(),
                ],
            'data': _static_data['dp_raw_obs_type']
            },
        {
            'name': 'dpa_raw_obs_sweep_obs',
            'desc': 'The association between raw obs and its sweep obs.',
            'parent': 'data_prod_assoc',
            'columns': [
                c.fk(
                    'dp_raw_obs', name='dp_sweep_obs_pk',
                    comment='The sweep obs used in deriving the raw obs.',
                    ),
                c.fk(
                    'dp_raw_obs',
                    comment='The raw obs.',
                    ),
                # mysql does not allow this
                # sa.CheckConstraint('dp_raw_obs_pk != dp_sweep_obs_pk')
                ]
            },
        {
            'name': 'dpa_proc',
            'desc': 'The associations established from data processing.',
            'parent': 'data_prod_assoc',
            'columns': [
                c.fk(
                    'data_prod', name='in_data_prod_pk',
                    comment='The input data product.',
                    ),
                c.fk(
                    'data_prod', name='out_data_prod_pk',
                    comment='The output data product.',
                    ),
                # sa.CheckConstraint('in_data_prod_pk != out_data_prod_pk'),
                ] + _make_proc_descriptor(),
            },
        {
            'name': 'dp_basic_reduced_obs',
            'desc': 'This data product is structured similar to '
                    'that of the raw obs, but instead consists of processed '
                    'interface data rather than the raw data.',
            'parent': 'data_prod',
            'columns': [],
            },
        {
            'name': 'dpa_basic_reduced_obs_raw_obs',
            'desc': 'The associations between basic reduced obs and '
                    'its parent raw obs.',
            'parent': 'data_prod_assoc',
            'columns': [
                c.fk(
                    'dp_raw_obs',
                    comment='The raw obs that derives this basic reduced obs.',
                    unique=True),
                c.fk(
                    'dp_basic_reduced_obs',
                    comment='The basic reduced obs.',
                    unique=True),
                ] + _make_proc_descriptor(),
            },
        {
            'name': 'dp_named_group',
            'desc': 'This data product consists of an arbitrary set of'
                    ' data products with some metadata.',
            'parent': 'data_prod',
            'columns': [
                c.name(),
                sa.Column(
                    'meta', sa.JSON(none_as_null=False),
                    nullable=False,
                    default='null',
                    comment='The metadata of this group of data products.'
                    ),
                ],
            },
        {
            'name': 'dpa_named_group_data_prod',
            'desc': 'The associations between named group and its content'
                    ' data products.',
            'parent': 'data_prod_assoc',
            'columns': [
                c.fk(
                    'data_prod',
                    comment='The content data product.',
                    ),
                c.fk(
                    'dp_named_group',
                    comment='The named group the content belongs to.',
                    ),
                ],
            },
        {
            'name': 'dp_pointing',
            'desc': 'This data product consists of reduced pointing'
                    ' observation maps and catalogs, and a final pointing'
                    ' offset report.',
            'parent': 'data_prod',
            'columns': [
                sa.Column(
                    'lon_offset_deg', sa.Float, nullable=False,
                    comment='The longitudinal pointing offset in degree.',
                    ),
                sa.Column(
                    'lat_offset_deg', sa.Float, nullable=False,
                    comment='The latitudinal pointing offset in degree.',
                    ),
                sa.Column(
                    'ra_center_deg', sa.Float, nullable=False,
                    comment='The RA of the pointing obs center in degree.'
                    ),
                sa.Column(
                    'dec_center_deg', sa.Float, nullable=False,
                    comment='The Dec of the pointing obs center in degree.'
                    ),
                ],
            },
        {
            'name': 'dp_m2',
            'desc': 'This data product consists of reduced M2'
                    ' observation maps and catalogs. A typical'
                    ' use of this data product is to determine'
                    ' the best focus.',
            'parent': 'data_prod',
            'columns': [],
            },
        {
            'name': 'dp_m1',
            'desc': 'This data product consists of reduced M1'
                    ' observation maps and catalogs. A typical'
                    ' use of this data product is to determine'
                    ' the optimal astigmatism coefficient.',
            'parent': 'data_prod',
            'columns': [],
            },
        {
            'name': 'dp_beammap',
            'desc': 'This data product consists of reduced beammap'
                    ' observation maps and the generated'
                    ' array property tables.',
            'parent': 'data_prod',
            'columns': [],
            },
        {
            'name': 'dp_science',
            'desc': 'This data product consists of (coadded) science'
                    ' observation maps and catalogs, as well as'
                    ' additional diagnostic data/plots.',
            'parent': 'data_prod',
            'columns': [],
            },
        {
            'name': 'dp_autodrive',
            'desc': 'This data product is reduced from a set of sweeps'
                    ' to derive the optimal driving attenuations of the KIDs.',
            'parent': 'data_prod',
            'columns': [],
            },
        {
            'name': 'dp_fts',
            'desc': 'This data product is reduced from a set of FTS'
                    ' observations to derive the filter curves.',
            'parent': 'data_prod',
            'columns': [],
            },
        {
            'name': 'dp_optical_efficiency',
            'desc': 'This data product is reduced from a set of optical'
                    ' efficiency measurements.',
            'parent': 'data_prod',
            'columns': [],
            },
        {
            'name': 'dp_responsivity',
            'desc': 'This data product is reduced from a set of '
                    'responsivity measurements.',
            'parent': 'data_prod',
            'columns': [],
            },
        {
            'name': 'dp_blackbody_responsivity',
            'desc': 'This data product is reduced from a set of '
                    'blackbody responsivity measurements.',
            'parent': 'data_prod',
            'columns': [],
            },
        {
            'name': 'dp_loopback_if',
            'desc': 'This data product is reduced from a set of '
                    'loopback + IF measurements.',
            'parent': 'data_prod',
            'columns': [],
            },
        {
            'name': 'dp_loopback_roach',
            'desc': 'This data product is reduced from a set of '
                    'loopback + ROACH measurements.',
            'parent': 'data_prod',
            'columns': [],
            },
        ], key='name')

    # for tables with 'parent', we insert the pfk
    # also we collect table info to the table info table.
    for t in table_defs.values():
        table_defs[c.TABLE_INFO_TABLE_NAME]['data'].append({
            'name': t['name'],
            'desc': t['desc'],
            })
        t_parent = t.pop('parent', None)
        if t_parent is not None:
            parent_type_data = table_defs[f'{t_parent}_type']['data']
            if t_parent not in parent_type_data:
                parent_type_data.append({
                    'label': table_defs[t_parent]['name'],
                    'desc': 'The base type.',
                    })
            parent_type_data.append({
                'label': t['name'],
                'desc': t['desc'],
                })
            t['columns'] = [c.pfk(t_parent, name='pk'), ] + t['columns']
    return table_defs


def _validate_db(db):
    if db.engine.url.database == '':
        raise ValueError(
                f"database name shall be specified "
                f"in the db url {db.engine.url}")
    return db


def init_db(db, create_tables=False, recreate=False):
    """
    Populate tables for `db`.

    .. note::

        The `db` url shall have the database specified.

    Parameters
    ----------
    db : `tollan.utils.db.SqlaDB`
        The database to hold the tables.
    create_tables : bool
        If True, tables are created in the database.
    """
    _validate_db(db)

    logger = get_logger()

    table_defs = _make_table_defs()
    logger.debug(f"collected {len(table_defs)} tables")
    # populate db.metadata
    TableDefList(table_defs.values()).init_db(db)

    if not create_tables:
        return

    # populate the database
    engine = db.engine
    if recreate:
        with logit(logger.info, "delete all tables"):
            db.metadata.drop_all(engine)
    db.metadata.create_all(engine)

    # populate tables with static data
    for n, t in table_defs.items():
        tbl = db.metadata.tables[n]
        if 'data' in t:
            with engine.begin() as conn:
                # because the data is static, we just clear off the
                # existing data before populating it.
                if engine.dialect.name == 'mysql':

                    conn.execute(tbl.delete())

                    from sqlalchemy.dialects.mysql import insert
                    stmt_insert = insert(tbl)
                    stmt = stmt_insert.on_duplicate_key_update(
                            **{
                                x.name: x for x in stmt_insert.inserted
                                if not any([x.primary_key, x.unique])})
                    conn.execute(
                        stmt,
                        t['data']
                        )
                else:
                    conn.execute(tbl.delete())
                    conn.execute(
                        db.metadata.tables[n].insert(),
                        t['data']
                        )


@timeit
def init_orm(db):

    logger = get_logger()

    _t = db.metadata.tables

    session = db.Session()

    Base = automap_base(metadata=db.metadata)

    # query the label tables to get dispatch maps
    dispatch_labels = dict()

    for table_name in [
            'data_prod_type', 'data_prod_assoc_type',
            'dp_raw_obs_type', 'dp_raw_obs_master',
            ]:
        dispatch_labels[table_name] = odict_from_list(
                session.execute(
                    se.select([_t[table_name]])), key='label')
    logger.debug(
            f"dispatch_labels: "
            f"{pformat_dict(dispatch_labels)}")

    class DataProd(Base):
        __tablename__ = 'data_prod'
        __mapper_args__ = {
            'polymorphic_identity':
                dispatch_labels['data_prod_type']['data_prod']['pk'],
            'polymorphic_on': 'data_prod_type_pk'
            }

    class RawObs(DataProd):
        __tablename__ = 'dp_raw_obs'
        __mapper_args__ = {
            'polymorphic_identity':
                dispatch_labels['data_prod_type']['dp_raw_obs']['pk']
        }

    class BasicReducedObs(DataProd):
        __tablename__ = 'dp_basic_reduced_obs'
        __mapper_args__ = {
            'polymorphic_identity':
                dispatch_labels['data_prod_type']['dp_basic_reduced_obs']['pk']
        }

    class NamedGroup(DataProd):
        __tablename__ = 'dp_named_group'
        __mapper_args__ = {
            'polymorphic_identity':
                dispatch_labels['data_prod_type']['dp_named_group']['pk']
        }

    class DataProdAssoc(Base):
        __tablename__ = 'data_prod_assoc'
        __mapper_args__ = {
            'polymorphic_identity':
                dispatch_labels['data_prod_assoc_type'][
                    'data_prod_assoc']['pk'],
            'polymorphic_on': 'data_prod_assoc_type_pk'
            }

    class DataProdAssocInfo(Base):
        __tablename__ = 'data_prod_assoc_info'

    class NamedGroupDataProdAssoc(DataProdAssoc):
        __tablename__ = 'dpa_named_group_data_prod'
        __mapper_args__ = {
            'polymorphic_identity':
                dispatch_labels['data_prod_assoc_type'][
                    'dpa_named_group_data_prod']['pk']
            }

    class BasicReducedObsRawObsAssoc(DataProdAssoc):
        __tablename__ = 'dpa_basic_reduced_obs_raw_obs'
        __mapper_args__ = {
            'polymorphic_identity':
                dispatch_labels['data_prod_assoc_type'][
                    'dpa_basic_reduced_obs_raw_obs']['pk']
            }

    class RawObsSweepObsAssoc(DataProdAssoc):
        __tablename__ = 'dpa_raw_obs_sweep_obs'
        __mapper_args__ = {
            'polymorphic_identity':
                dispatch_labels['data_prod_assoc_type'][
                    'dpa_raw_obs_sweep_obs']['pk']
            }

    ClientInfo = c.client_info_model(Base)  # noqa: F841

    # https://github.com/sqlalchemy/sqlalchemy/issues/7149
    def to_snake_case(text):
        return ''.join(
            [f"_{i.lower()}" if i.isupper() else i for i in text]
            ).lstrip('_')

    def name_for_scalar_relationship(
            _base, _local_cls, referred_cls, _constraint):
        return to_snake_case(referred_cls.__name__)

    # this is need to resolve the many-to-many relation among the child
    # tables
    def name_for_collection_relationship(
            base, local_cls, referred_cls, constraint):
        desc = '_'.join(
            re.sub(r'(_pk|_fk)', '', col.name)
            for col in constraint.columns
            )
        return f'{to_snake_case(referred_cls.__name__)}_{desc}_collection'

    Base.prepare(
        name_for_scalar_relationship=name_for_scalar_relationship,
        name_for_collection_relationship=name_for_collection_relationship)

    db.models = {
        k: v
        for k, v in locals().items()
        if isinstance(v, type) and issubclass(v, Base)
        }
