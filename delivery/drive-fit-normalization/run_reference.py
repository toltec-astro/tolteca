"""Run both public workflows on all network-0 channels of the compact reference."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import yaml
from tolteca_datamodels.data_prod import DataProdIndex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-lmt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plots', action='store_true', help='Generate summaries and per-channel fit plots.')
    args = parser.parse_args()
    raw = args.data_lmt.resolve() / 'toltec/tcs/toltec0'
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('--output must be a new or empty directory')
    paths = sorted(raw.glob('toltec0_145484_*_targsweep.nc'))
    targets = sorted(raw.glob('toltec0_145485_*_targsweep.nc'))
    if len(paths) != 7 or len(targets) != 1:
        parser.error('Expected seven 145484 target sweeps and one 145485 target sweep; see reference-network0.json')
    out.mkdir(parents=True, exist_ok=True)
    products = out / 'dataprod_toltec'

    def discovery(obsnum):
        return {'input_discovery': {'inputs': [{'sources': [
            {'glob': str(raw / f'toltec0_{obsnum}_*_targsweep.nc')}
        ]}]}}

    def execute(name, config):
        config_path = out / f'{name}.yaml'
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        cmd = [sys.executable, '-m', f'tolteca_kids.{name}', '--config_paths', str(config_path)]
        resolved = subprocess.run([*cmd, '--show_config'], check=True, capture_output=True, text=True)
        (out / f'{name}.resolved.json').write_text(resolved.stdout)
        print(f'Running {name}; log: {out / (name + ".log")}', flush=True)
        with (out / f'{name}.log').open('w') as log:
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
        kind = 'dp_drivefit' if name == 'drive_fit' else 'dp_drive_fit_commit'
        matches = [p for p in products.rglob('index.yaml')
                   if DataProdIndex.read(p).meta.get('data_prod_type') == kind]
        [product] = matches
        return product

    def output(name):
        return {'enabled': True, 'path': str(products), 'namespace': 'scratch',
                'run': {'name': name}, 'registration': False}

    fit = discovery(145484)
    fit['drive_fit'] = {
        'select': None,
        'fit': {'channels': None, 'n_adrvs_min': 5, 'weight_window_Qr': 12000.0, 'a_ref': 0.2},
        'plot': {'enabled': args.plots},
        'output': output('delivery-drive-fit-145484'),
    }
    fit_path = execute('drive_fit', fit)
    commit = discovery(145485)
    commit['drive_fit_commit'] = {
        'reference': str(fit_path),
        'commit': {'adrv_max': 30.0, 'adrv_fill_perc': 50.0, 'adrv_ref_perc': 10.0, 'adrv_ref_step': 0.25},
        'plot': {'enabled': args.plots},
        'output': output('delivery-drive-fit-commit-145485'),
        'tlaloc_output': {'enabled': False},
    }
    commit_path = execute('drive_fit_commit', commit)
    manifest = {'fit': str(fit_path), 'commit': str(commit_path)}
    (out / 'products.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
