import socketserver
import json
import dispatch_reduction as dr
from pathlib import Path

from tolteca.datamodels.toltec import BasicObsDataset
from tolteca.datamodels.toltec.data_prod import ToltecDataProd

def get_dp_for_dataset(rootdir, dataset):
    reduced_dir = rootdir.joinpath('toltec/reduced')
    obsnum = dataset[0]['obsnum']
    if dataset[0]['master_name'] == 'ics':
        # toltec kids data
        data_files = list(reduced_dir.glob(f'toltec[0-9]?_{obsnum:06d}_*'))
        if not data_files:
            return None
        bods = BasicObsDataset.from_files(data_files)
        return bods
    if dataset[0]['master_name'] == 'tcs':
        # reduction is on per obsnum directory basis
        dpdir = rootdir.joinpath(f'{obsnum}')
        if not dpdir.exists():
            return None
        return ToltecDataProd.collect_from_dir(dpdir)


def get_quicklook_data(rootdir, bods):
    dp = get_dp_for_dataset(rootdir, bods)
    print(dp)
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
                    ]
                }, None
    return dp.index, None


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
        s = json.dumps(data).encode()
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
        ql_data, message = get_quicklook_data(rootdir, bods)
        if ql_data is None:
            return {
                    'exit_code': 1,
                    'message': message
                    }
        return {
                'exit_code': 0,
                'data': get_quicklook_data(rootdir, bods)
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
        print(json.dumps(resp))
    else:
        run_server()
