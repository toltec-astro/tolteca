#! /usr/bin/env python

"""
This recipe shows how to populate the data prod db with TolTEC dataset.
"""

import yaml
from tollan.utils import rupdate
from tolteca.recipes import get_logger
from sqlalchemy.sql import select
from tollan.utils import odict_from_list
from tollan.utils.fmt import pformat_dict
from tollan.utils.sys import get_hostname
from collections import OrderedDict
from tollan.utils.db import SqlaDB
from tolteca.datamodels.db.toltec import data_prod
from tolteca.datamodels.toltec.enums import KidsDataKind
from tolteca.datamodels.toltec import BasicObsDataset


def collect_data_prods(db, dataset):
    # this add datasets to the db, creating known associations

    logger = get_logger()

    # create engine
    from sqlalchemy.ext.automap import automap_base
    from tollan.utils.db.conventions import client_info_model

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
                    select([_t[table_name]])), key='label')
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

    ClientInfo = client_info_model(Base)

    # this is need to resolve the many-to-many relation among the child
    # tables
    def name_for_collection_relationship(
            base, local_cls, referred_cls, constraint):
        disc = '_'.join(col.name for col in constraint.columns)
        return referred_cls.__name__.lower() + '_' + disc + "_collection"

    Base.prepare(
        # name_for_scalar_relationship=name_for_scalar_relationship,
        name_for_collection_relationship=name_for_collection_relationship)

    _hostname = get_hostname()
    client_info = session.query(ClientInfo).filter(
            ClientInfo.hostname == _hostname).first()
    if client_info is None:
        client_info = ClientInfo(hostname=_hostname)
        session.add(client_info)
        session.commit()

    # group all data files by the obs -subobs -scan
    tbl = dataset.index_table
    # the kidsmodel files does not have master and repeat
    # we manually add them for now
    tbl['master'] = 1
    tbl['repeat'] = 1
    print('\n'.join(tbl[['obsnum', 'subobsnum', 'scannum', 'master', 'repeat', 'data_kind']].pformat(max_lines=-1)))

    grouped = tbl.group_by(
            ['obsnum', 'subobsnum', 'scannum', 'master', 'repeat'])
    for key, group in zip(grouped.groups.keys, grouped.groups):
        print('****** {0} *******'.format(key['obsnum']))
        print(group)
        print('')

    raw_obs_items = dict()
    basic_reduced_obs_items = dict()
    assocs = []

    def make_item_key(item):
        return (
                item.source['obsnum'],
                item.source['subobsnum'],
                item.source['scannum'])

    for tbl in grouped.groups:
        # for each group, we collate all per-interface entry to
        # a single raw obs or basic reduced obs data product.
        ds = BasicObsDataset(index_table=tbl)
        # common meta data for this data product
        meta = ds.bod_list[0].meta
        common = {
                'master': int(meta.get('master', 1)),
                'obsnum': int(meta['obsnum']),
                'subobsnum': int(meta['subobsnum']),
                'scannum': int(meta['scannum']),
                'repeat': int(meta.get('repeat', 1)),
                'source_key': 'interface',
                'source_urlbase': None,
                }

        # these hold the bods of the relevant the data kind
        raw_obs_data_items = []
        basic_reduced_obs_data_items = []

        for bod in ds.bod_list:
            kind = bod.meta['data_kind']
            if isinstance(kind, KidsDataKind):
                if bod.meta['data_kind'] & KidsDataKind.RawKidsData:
                    raw_obs_data_items.append(bod)
                else:
                    basic_reduced_obs_data_items.append(bod)
            else:
                basic_reduced_obs_data_items.append(bod)

        print(f"collected {len(raw_obs_data_items)} raw obs data items and {len(basic_reduced_obs_data_items)} basic reduced obs data items")
        raw_obs = None
        basic_reduced_obs = None

        if len(raw_obs_data_items) > 0:
            raw_obs_source = OrderedDict(
                **common,
                **{
                    'sources': [
                        {
                            'key': d.meta['interface'],
                            'url': d.meta['file_loc'].uri,
                            'meta': {
                                k: d.meta[k]
                                for k in [
                                    'interface',
                                    'roachid',
                                    'n_tones',
                                    'n_tones_design',
                                    ]
                                }
                            }
                        for d in raw_obs_data_items
                        ],
                    'data_kind': meta['data_kind'].name,
                    'obs_type': int(meta['obs_type']),
                    'cal_obsnum': int(meta['cal_obsnum']),
                    'cal_subobsnum': int(meta['cal_subobsnum']),
                    'cal_scannum': int(meta['cal_scannum']),
                    'meta': {
                        'data_prod_type': 'raw_obs'
                        },
                    },
                )
            # raw_obs_type = dispatch_labels['dp_raw_obs_type'][
            #         raw_obs_source['type']]
            # master = dispatch_labels['dp_raw_obs_master'][
            #         raw_obs_source['master'].upper()]
            raw_obs = RawObs(
                    # dp_raw_obs_type_pk=raw_obs_type['pk'],
                    # dp_raw_obs_master_pk=master['pk'],
                    dp_raw_obs_type_pk=raw_obs_source['obs_type'],
                    dp_raw_obs_master_pk=raw_obs_source['master'],
                    obsnum=raw_obs_source['obsnum'],
                    subobsnum=raw_obs_source['subobsnum'],
                    scannum=raw_obs_source['scannum'],
                    repeat=raw_obs_source['repeat'],
                    source=raw_obs_source,
                    clientinfo=client_info,
                    )
            raw_obs_items[make_item_key(raw_obs)] = raw_obs

        if len(basic_reduced_obs_data_items) > 0:
            basic_reduced_obs_source = OrderedDict(
                **common,
                **{
                    'sources': [
                        {
                            'key': d.meta['interface'],
                            'url': d.meta['file_loc'].uri
                            }
                        for d in basic_reduced_obs_data_items
                        ],
                    'meta': {
                        'data_prod_type': 'basic_reduced_obs'
                        },
                    },
                )
            basic_reduced_obs = BasicReducedObs(
                    source=basic_reduced_obs_source,
                    clientinfo=client_info,
                    )
            basic_reduced_obs_items[
                    make_item_key(basic_reduced_obs)] = basic_reduced_obs

        if basic_reduced_obs is not None and raw_obs is not None:
            assoc_info = DataProdAssocInfo(
                    context='null',
                    clientinfo=client_info,
                    )
            basic_reduced_obs_raw_obs_assoc = BasicReducedObsRawObsAssoc(
                    dataprodassocinfo=assoc_info,
                    rawobs=raw_obs,
                    basicreducedobs=basic_reduced_obs,
                    module='tolteca.web.tasks.kidsreduce'
                    )
            assocs.append(
                    basic_reduced_obs_raw_obs_assoc)

    # this is to let the pk available on the items
    print(f"collected {len(raw_obs_items)} raw obs items and {len(basic_reduced_obs_items)} basic reduced obs items")
    for item in raw_obs_items.values():
        session.add(item)
    for item in basic_reduced_obs_items.values():
        session.add(item)
    session.flush()

    # search in the raw obs items to build raw_obs_sweep_obs assoc
    for item in raw_obs_items.values():
        cal_key = (
                item.source['cal_obsnum'],
                item.source['cal_subobsnum'],
                item.source['cal_scannum'])
        if cal_key in raw_obs_items:
            assoc_info = DataProdAssocInfo(
                context='null',
                clientinfo=client_info,
                )
            raw_obs_sweep_obs_assoc = RawObsSweepObsAssoc(
                    dp_sweep_obs_pk=raw_obs_items[cal_key].pk,
                    dp_raw_obs_pk=item.pk,
                    dataprodassocinfo=assoc_info,
                    )
            assocs.append(raw_obs_sweep_obs_assoc)

    # add this as a named group
    if 'file_loc' in dataset.meta:
        source_url = dataset.meta['file_loc'].uri
        assoc_info = DataProdAssocInfo(
                context={
                    'loaders': [
                        {
                            'func': (
                                'tolteca.datamodels.toltec'
                                ':BasicObsDataset.from_index_table'),
                            'args': [f'{source_url}', ]
                            },
                        ]
                    },
                clientinfo=client_info,
                )

        named_group = NamedGroup(
                source_url=source_url,
                name=dataset.meta['file_loc'].path.stem,
                clientinfo=client_info,
                )
        # assocs
        data_keys = [
                (d['obsnum'], d['subobsnum'], d['scannum'])
                for d in dataset]
        for ro, bro in zip(raw_obs_items, basic_reduced_obs_items):
            if (ro.obsnum, ro.subobsnum, ro.scannum) in data_keys:
                assocs.extend([
                    NamedGroupDataProdAssoc(
                        namedgroup=named_group,
                        dataprod=ro,
                        dataprodassocinfo=assoc_info,
                        ),
                    NamedGroupDataProdAssoc(
                        namedgroup=named_group,
                        dataprod=bro,
                        dataprodassocinfo=assoc_info,
                        ),
                    ])
    for item in assocs:
        session.add(item)
    session.commit()

    return len(assocs) + len(raw_obs_items) + len(basic_reduced_obs_items)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
            description='Populate data prod db with TolTEC dataset.')
    parser.add_argument(
            "-c", "--config",
            nargs='+',
            help="The path to the TolTECA config file(s). "
                 "Multiple config files are merged in order.",
            metavar='FILE',
            )
    parser.add_argument(
            "paths",
            nargs='+',
            help='Path(s) of data files.'
            )
    parser.add_argument(
            "-s", "--select",
            metavar="COND",
            help='A selection predicate, e.g.,:'
            '"(obsnum>8900) & (nwid==3) & (fileext=="nc")"',
            )
    option = parser.parse_args()
    # load config
    _config = None
    for c in option.config:
        with open(c, 'r') as fo:
            if _config is None:
                _config = yaml.safe_load(fo)
            else:
                rupdate(_config, yaml.safe_load(fo))
    option.config = _config

    logger = get_logger()

    dpdb_uri = option.config['db']['tolteca']['uri']

    logger.debug(f"populate database: {dpdb_uri}")

    db = SqlaDB.from_uri(dpdb_uri, engine_options={'echo': False})
    data_prod.init_db(db, create_tables=False)
    dataset = BasicObsDataset.from_files(option.paths)
    collect_data_prods(db, dataset)
