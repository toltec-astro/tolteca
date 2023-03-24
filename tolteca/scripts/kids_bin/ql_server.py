import socketserver
import json
import dispatch_reduction as dr
from pathlib import Path
from json import JSONEncoder
import yaml

from tolteca.datamodels.toltec import BasicObsDataset
from tolteca.datamodels.toltec.data_prod import ToltecDataProd, DataItemKind, ScienceDataProd


QL_SEARCH_PATHS = [
        '/home/toltec/work_toltec/220708/redu/focus/focus3',
        'toltec/reduced_manual',
        'toltec/reduced'
        ]


def collect_from_citlali_index(index_file):
    with open(index_file, 'r') as fo:
        index = yaml.safe_load(fo)
    # need to do some translation
    data_items = list()
    rootdir = Path(index_file).parent
    for p in index['files']:
        for a in ['a1100', 'a1400', 'a2000']:
            if a in p:
                array_name = a
                break
        else:
            array_name = None
        p = rootdir.joinpath(p).resolve()
        if p.suffix == '.fits' and array_name is not None:
            data_items.append({
               'array_name': array_name,
               'kind': DataItemKind.CalibratedImage,
               'filepath': p,
               })
        elif p.suffix == '.nc' and 'timestream' in p.name:
            data_items.append({
                'array_name': array_name,
                'kind': DataItemKind.CalibratedTimeOrderedData,
                'filepath': p
                })
    index = {
        'data_items': data_items,
        'meta': {
            'name': f'o{int(rootdir.name)}',
            'id': 0,
            }
        }
    return ToltecDataProd(source={'data_items': [ScienceDataProd(source=index)], 'meta': {'id': 0}})


def get_dp_for_dataset(rootdir, dataset, reduced_dir='toltec/reduced'):
    reduced_dir = rootdir.joinpath(reduced_dir)
    obsnum = dataset[0]['obsnum']
    dpdir = reduced_dir.joinpath(f'{obsnum}')
    if not dpdir.exists():
        dpdir = None
    if dataset[0]['master_name'] == 'ics':
        # toltec kids data
        data_files = list(reduced_dir.glob(f'toltec[0-9]?_{obsnum:06d}_*'))
        if not data_files:
            return None
        bods = BasicObsDataset.from_files(data_files)
        bods.meta['dpdir'] = dpdir
        return bods
    if dataset[0]['master_name'] == 'tcs':
        # reduction is on per obsnum directory basis
        if dpdir is None:
            return
        citlali_index = dpdir.joinpath("index.yaml")
        if citlali_index.exists():
            # translate the index to dp index
            dps = collect_from_citlali_index(citlali_index)
        else:
            dps = ToltecDataProd.collect_from_dir(dpdir)
        if not dps:
            return None
        # get the one with largest id
        dp = sorted(dps.index['data_items'], key=lambda d: d.meta['id'])[-1]
        return dp


def get_quicklook_data(rootdir, bods, search_paths=None):
    if search_paths is None:
        search_paths = ['toltec/reduced', ]
    for rd in search_paths:
        dp = get_dp_for_dataset(rootdir, bods, reduced_dir=rd)
        if dp is not None:
            break
    # extract quick look data from dp
    if dp is None:
        return None, 'No reduced files found.'
    if isinstance(dp, BasicObsDataset):
        # just report the number of reduced files
        return {
                'meta': {
                    'name': f'{dp!r}',
                    },
                # 'n_data_items': len(dp),
                'data_items': [
                    {
                        'filepath': entry['source']
                        }
                    for entry in dp.index_table
                    ],
                'kids_ql_data': get_kids_ql_data(dp),
                }, None
    if isinstance(dp, ToltecDataProd):
        # handle pointing reader outputs
        obs_goal = dr.get_obs_goal(bods)
        index = dp.index.copy()
        print(f'obs goal was {obs_goal}')
        if obs_goal in ['test', 'pointing'] or 'pointing' in index['data_items'][0]['filepath'].name:
            pointing_reader_data = get_pointing_reader_data(dp)
            index['pointing_reader_data'] = pointing_reader_data
        elif obs_goal in ['beammap', ] or 'beammap' in index['data_items'][0]['filepath'].name:
            beammap_reader_data = get_beammap_reader_data(dp)
            index['beammap_reader_data'] = beammap_reader_data
        print(index)
        return index, None
    return None, "Unkown type of data product."


def get_kids_ql_data(dp):
    print(f'collecting kids ql data for {dp}')
    dpdir = dp.meta['dpdir']
    quicklook_files = []
    if dpdir is not None:
        quicklook_files += list(dpdir.glob("ql_*.png"))
    result = {
            'quicklook_files': quicklook_files,
            }
    return result


def get_pointing_reader_data(dp):
    print(f'collecting pointing reader data for {dp}')
    # get the data rootpath
    datadir = Path(dp.index['data_items'][0]['filepath']).parent
    result = {
            'meta': dp.meta,
            'data_items': list()
            }
    for array_name in ['a1100', 'a1400', 'a2000']:
        param_file = list(datadir.glob(f'toltec_{array_name}_pointing_*_params.txt'))
        if not param_file:
            continue
        param_file = param_file[-1]
        quicklook_files = [param_file]
        image_file = list(datadir.glob(f'toltec_{array_name}_pointing_*_image.png'))
        if image_file:
            quicklook_files.append(image_file[-1])
        with open(param_file, 'r') as fo:
            params = json.load(fo)
        result['data_items'].append({
            'array_name': array_name,
            'params': params,
            'quicklook_files': quicklook_files
            })
    return result


def get_beammap_reader_data(dp):
    print(f'collecting beammap reader data for {dp}')
    # get the data rootpath
    datadir = Path(dp.index['data_items'][0]['filepath']).parent
    result = {
            'meta': dp.meta,
            'data_items': list()
            }
    result['data_items'].append({
        'quicklook_files': list(datadir.glob(f'toltec_beammap_*_image.png'))
        })
    return result



class MyEncoder(JSONEncoder):
    def default(self, o):
        return str(o)


class MyTCPHandler(socketserver.BaseRequestHandler):
    """
    The request handler class for our server.

    It is instantiated once per connection to the server, and must
    override the handle() method to implement communication to the
    client.
    """

    def handle(self):
        # self.request is the TCP socket connected to the client
        data = self.data = self.request.recv(1024).strip()
        print("{} requested reduction:".format(self.client_address[0]))
        try:
            obsnum = int(data.decode())
        except:
            print(f'invalid data: {data}')
            self.send_json(None)
            return
        # handle obsnum
        print(f'obsnum={obsnum}')
        # just send back the same data, but upper-cased
        data = self.handle_obsnum(obsnum)
        self.send_json(data)

    def send_json(self, data):
        s = json.dumps(data, cls=MyEncoder).encode()
        self.request.sendall(s)

    @classmethod
    def handle_obsnum(cls, obsnum):
        rootdir = Path('/data/data_lmt')
        bods = dr.get_dataset(rootdir, obsnum)
        if bods is None:
            print('no data found for obsnum')
            return {
                    'exit_code': -1,
                    'message': f'No data found for obsnum={obsnum}'
                    }
        # print(bods)
        ql_data, message = get_quicklook_data(rootdir, bods, search_paths=QL_SEARCH_PATHS)
        if ql_data is None:
            return {
                    'exit_code': 1,
                    'message': message
                    }
        return {
                'exit_code': 0,
                'data': ql_data,
                'message': message,
                }


def run_server():
    HOST, PORT = "0.0.0.0", 9876

    print(f'serve at {HOST=} {PORT=}')

    with socketserver.TCPServer((HOST, PORT), MyTCPHandler) as server:
        server.serve_forever()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        resp = MyTCPHandler.handle_obsnum(int(sys.argv[1]))
        print('response:\n--------------------\n')
        print(json.dumps(resp, cls=MyEncoder, indent=2))
    else:
        run_server()
