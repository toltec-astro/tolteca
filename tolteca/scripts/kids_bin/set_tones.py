from astropy.table import Table, unique
import shutil
from pathlib import Path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_params_files', nargs='+')
    parser.add_argument('--dp', required=True)
    parser.add_argument('--filter_by_apt')


    option = parser.parse_args()
    dp = float(option.dp)
    apt = option.filter_by_apt
    if apt is not None:
        apt = Table.read(apt, format='ascii.ecsv')
        apt['idx'] = range(len(apt))
    print(option)
    for f in option.model_params_files:
        mp = Table.read(f, format='ascii.ecsv')
        print(mp.meta)
        nw = mp.meta['Header.Toltec.RoachIndex']
        ampcor_file = f"/home/toltec/tlaloc/etc/toltec{nw}/default_targ_amps.dat"
        ampcor_orig_file = ampcor_file + '.backup_set_tones_orig'
        try:
            if not Path(ampcor_orig_file).exists():
                print(f"make copy of {ampcor_file} {ampcor_orig_file}")
                shutil.copy(ampcor_file, ampcor_orig_file)
            ampcor = Table.read(ampcor_orig_file, format='ascii.no_header')
        except Exception:
            print(f"skip ampcor for nw {nw}")
            ampcor = None
        if nw <= 6:
            rsp = -5.794e-5
        elif nw >= 7 and nw <= 10:
            rsp = -1.1e-4
        elif nw >= 11:
            rsp = -1.1e-4
        else:
            raise ValueError('invalid nw')
        dx = dp * rsp
        mp['f_out_orig'] = mp['f_out']
        mp['f_out'] = mp['f_out'] * (1 + dx)
        if apt is not None:
            m_good = (apt['nw'] == nw) & (apt['flag'] == 1)
            print(f"select {m_good.sum()} of {len(mp)} detectors based on APT {option.filter_by_apt}")
            mp = mp[apt['kids_tone'][m_good]]
            mp['apt_idx'] = apt['idx'][m_good]
            mp['kids_tone'] = apt['kids_tone'][m_good]
            # remove duplicated tones
            mp_u = unique(unique(mp, keys='f_in'), keys='f_out',)
            print(f"select {len(mp_u)} of {len(mp)} detectors for unique tone")
            mp = mp_u
            if ampcor is not None:
                ampcor_u = ampcor[mp['kids_tone']]
            ampcor_u.write(f.replace("tune.txt", "ampcor_filtered.txt"), format='ascii.no_header', overwrite=True)
        else:
            ampcor.write(f.replace("tune.txt", "ampcor_filtered.txt"), format='ascii.no_header', overwrite=True)
        mp.write(f.replace("tune.txt", 'tune_shifted.txt'), format='ascii.ecsv', overwrite=True)
