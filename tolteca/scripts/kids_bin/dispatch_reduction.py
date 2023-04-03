#!/usr/bin/env python

from pathlib import Path
from tolteca.datamodels.toltec import BasicObsDataset
from tollan.utils import call_subprocess_with_live_output
import pty
import shlex

scriptdir = Path(__file__).parent


def get_dataset_of_latest_obsnum():
    links = (
            list(Path("/data/data_toltec").glob('toltec[0-9].nc'))
            + list(Path("/data/data_toltec").glob('toltec[0-9][0-9].nc')))
    if not links:
        return None
    dataset = BasicObsDataset.from_files([link.resolve() for link in links])
    dataset.sort(['ut'])
    obsnum = dataset[-1]['obsnum']
    print(f'latest obsnum: {obsnum}')
    return dataset[dataset['obsnum'] == obsnum]


def get_dataset(rootdir, obsnum):
    links = (
        list(rootdir.glob(f"toltec/tcs/toltec[0-9]*/toltec[0-9]*_{obsnum:06d}_*.nc"))
        + list(rootdir.glob(f"toltec/ics/toltec[0-9]*/toltec[0-9]*_{obsnum:06d}_*.nc"))
        + list(rootdir.glob(f"tel/tel_toltec_*_{obsnum:06d}_*.nc"))
        + list(rootdir.glob(f"toltec/ics/wyatt/wyatt_*_{obsnum:06d}_*.nc"))
        )
    print(links)
    if not links:
        return None
    dataset = BasicObsDataset.from_files([link.resolve() for link in links])
    dataset.sort(['interface'])
    return dataset


def shell_run(cmd):
    import shlex
    import subprocess
    from io import TextIOWrapper

    def _handle_ln(ln):
        sys.stdout.write(ln)

    with subprocess.Popen(
            shlex.split(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
            shell=True,
            ) as proc:
        reader = TextIOWrapper(proc.stdout, newline='')
        for ln in iter(
                reader.readline, b''):
            _handle_ln(ln)
            if proc.poll() is not None:
                sys.stderr.write('\n')
                break
        retcode = proc.returncode
        if retcode:
            # get any remaining message
            ln, _ = proc.communicate()
            _handle_ln(ln.decode())
            _handle_ln(
                f"The process exited with error code: {retcode}\n")
            return False
        return True


def get_obs_goal(dataset):
    if dataset[0]['master_name'] != 'tcs':
        return None
    try:
        bod_tel = dataset[dataset['interface'] == 'lmt'].bod_list[0].open()
        obs_goal = bod_tel.meta['obs_goal']
        return obs_goal
    except Exception:
        return None


if __name__ == "__main__":
    from tollan.utils.log import init_log
    # init_log(level='DEBUG')
    import sys
    if len(sys.argv) < 2:
        dataset = get_dataset_of_latest_obsnum()
        if dataset is None:
            print('no file found, abort')
            sys.exit(1)
        obsnum = dataset[-1]['obsnum']
    else:
        obsnum = int(sys.argv[1])
        dataset = get_dataset(Path('/data/data_lmt'), obsnum)
        if dataset is None:
            print(f'no file found for obsnum={obsnum}, abort')
            sys.exit(1)
    print(dataset)

    # diaptch based on type
    if dataset[0]['master_name'] == 'ics':
        print('run {reduce_all.sh}')
        # shell_run(f'./reduce_all_seq.sh {obsnum}')
        pty.spawn(shlex.split(f'{scriptdir}/reduce_all_seq_new.sh {obsnum}'))
    elif dataset[0]['master_name'] == 'tcs':
        # cehck the tel obsgola for the obsgaol
        obs_goal = get_obs_goal(dataset)
        print(obs_goal)
        if obs_goal is None:
            pass
            # print('run general reduction')
            # pty.spawn(shlex.split(f'{scriptdir}/reduce_all_seq_new.sh {obsnum}'))
        else:
            # need to get all tune files reduced
            for filepath in dataset['source']:
                if filepath.endswith('tune.nc'):
                    pass
                    # pty.spawn(shlex.split(f'{scriptdir}/reduce.sh {filepath}'))
        if obs_goal == 'beammap' or obs_goal == 'azscan' or obs_goal == 'elscan':
            print('run beammap')
            pty.spawn(shlex.split(f'bash {scriptdir}/reduce_beammap.sh {obsnum}'))
        elif obs_goal == 'pointing' or obs_goal == 'focus' or obs_goal == 'astigmatism' or obs_goal == 'm3offset':
            print('run pointing')
            pty.spawn(shlex.split(f'{scriptdir}/reduce_pointing.sh {obsnum}'))
        elif obs_goal == 'science':
            print('run science')
            pty.spawn(shlex.split(f'{scriptdir}/reduce_science.sh {obsnum}'))
        else:
            print(f'unknown obs goal: {obs_goal}, no action.')
            # print('run pointing for test')
            # pty.spawn(shlex.split(f'./reduce_pointing.sh {obsnum}'))
    else:
        print('unknown data, ignore')
