#! /usr/bin/env python

"""
This recipe shows how to populate the data prod db with TolTEC dataset.
"""

import yaml
from astropy.table import Table
from tolteca.fs.toltec import ToltecDataset
from tollan.utils import rupdate
from tolteca.recipes import get_logger
from sqlalchemy.sql import select
from tollan.utils import odict_from_list
from tollan.utils.fmt import pformat_dict
from tollan.utils.sys import get_hostname
from pathlib import Path
from collections import OrderedDict
from tollan.utils.db import SqlaDB
from tolteca.db.toltec import dataprod


def load_dataset(path):
    logger = get_logger()
    try:
        # try load the dataset if it is a pickle
        dataset = ToltecDataset.load(path)
    except Exception:
        # now try load the dataset as an index table
        dataset = ToltecDataset(
                Table.read(path, format='ascii'))
    except Exception:
        raise RuntimeError(f"cannot load dataset from {path}")
    logger.debug(f'loaded dataset: {dataset}')
    dataset.source = path
    return dataset


def collect_data_prods(db, datasets):

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
    grouped = ToltecDataset.vstack(datasets).split(
            'obsid', 'subobsid', 'scanid')

    raw_obs_items = []
    basic_reduced_obs_items = []
    assocs = []

    for dataset in grouped:
        # build sources
        data = dataset[0]
        common = {
                'obsid': int(data['obsid']),
                'subobsid': int(data['subobsid']),
                'scanid': int(data['scanid']),
                'repeat': 1 if data['master'] == 'repeated' else 0,
                'source_key': 'interface',
                'source_urlbase': None,
                }

        raw_obs_data_items = []
        basic_reduced_obs_data_items = []

        for data in dataset:
            if data['kindstr'] == 'ancillary':
                raw_obs_data_items.append(data)
            else:
                basic_reduced_obs_data_items.append(data)
        raw_obs_source = OrderedDict(
                **common,
                **{
                    'master': data['master'],
                    'sources': [
                        {
                            'key': data['interface'],
                            'url': Path(data['source']).resolve().as_uri()
                            }
                        for data in raw_obs_data_items
                        ],
                    'type': data['kindstr'],
                    'meta': {
                        'data_prod_type': 'raw_obs'
                        },
                    },
                )
        basic_reduced_obs_source = OrderedDict(
                **common,
                **{
                    'master': data['master'],
                    'sources': [
                        {
                            'key': data['interface'],
                            'url': Path(data['source']).resolve().as_uri()
                            }
                        for data in basic_reduced_obs_data_items
                        ],
                    'type': data['kindstr'],
                    'meta': {
                        'data_prod_type': 'basic_reduced_obs'
                        },
                    },
                )
        # create objs
        raw_obs_type = dispatch_labels['dp_raw_obs_type'][
                raw_obs_source['type']]
        master = dispatch_labels['dp_raw_obs_master'][
                raw_obs_source['master'].upper()]
        raw_obs = RawObs(
                dp_raw_obs_type_pk=raw_obs_type['pk'],
                dp_raw_obs_master_pk=master['pk'],
                obsid=raw_obs_source['obsid'],
                subobsid=raw_obs_source['subobsid'],
                scanid=raw_obs_source['scanid'],
                repeat=raw_obs_source['repeat'],
                source=raw_obs_source,
                clientinfo=client_info,
                )
        raw_obs_items.append(raw_obs)

        basic_reduced_obs = BasicReducedObs(
                source=basic_reduced_obs_source,
                clientinfo=client_info,
                )
        basic_reduced_obs_items.append(basic_reduced_obs)

    assoc_info = DataProdAssocInfo(
            context='null',
            clientinfo=client_info,
            )

    for ro, bro in zip(raw_obs_items, basic_reduced_obs_items):
        basic_reduced_obs_raw_obs_assoc = BasicReducedObsRawObsAssoc(
                dataprodassocinfo=assoc_info,
                rawobs=ro,
                basicreducedobs=bro,
                module='tolteca.web.tasks.kidsreduce'
                )
        assocs.append(basic_reduced_obs_raw_obs_assoc)

    for dataset in datasets:
        path = dataset.source
        source_url = Path(path).resolve().as_uri()
        assoc_info = DataProdAssocInfo(
                context={
                    'loaders': [
                        {
                            'func': 'tolteca.fs.toltec:ToltecDataset',
                            'args': [f'{source_url}', ]
                            },
                        ]
                    },
                clientinfo=client_info,
                )

        named_group = NamedGroup(
                source_url=source_url,
                name=Path(path).stem,
                clientinfo=client_info,
                )
        data_keys = [(d['obsid'], d['subobsid'], d['scanid']) for d in dataset]
        for ro, bro in zip(raw_obs_items, basic_reduced_obs_items):
            if (ro.obsid, ro.subobsid, ro.scanid) in data_keys:
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
            help='Path(s) of dataset index files.'
            )
    parser.add_argument(
            "-s", "--select",
            metavar="COND",
            help='A selection predicate, e.g.,:'
            '"(obsid>8900) & (nwid==3) & (fileext=="nc")"',
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
    dataprod.init_db(db, create_tables=False)
    datasets = list(map(load_dataset, option.paths))

    collect_data_prods(db, datasets)
