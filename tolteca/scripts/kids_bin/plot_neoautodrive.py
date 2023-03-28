#!/bin/bash
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
from get_ampcor_from_adrv import load_ampcor


def get_adrv(scratch_dir, obsnum, nw):
    f = scratch_dir.joinpath(
        f"drive_atten_toltec{nw}_{obsnum:06d}_adrv.csv")
    if f.exists():
        return load_ampcor(f)
    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('obsnums', nargs='+')
    parser.add_argument('--scratch_dir')

    option = parser.parse_args()

    scratch_dir = option.scratch_dir or "/data_lmt/toltec/reduced"
    scratch_dir = Path(scratch_dir)

    obsnums = list(map(int, option.obsnums))

    data = {}
    for nw in range(13):
        for obsnum in obsnums:
            adrv = get_adrv(scratch_dir, obsnum, nw)
            if adrv is None:
                continue
            if nw not in data:
                data[nw] = []
            data[nw].append(adrv)

    n_rows = len(obsnums)
    n_cols = len(data.keys())

    panel_size = 4 if n_cols <= 6 else 2

    fig, axes = plt.subplots(
        n_rows, n_cols,
        constrained_layout=True,
        figsize=(panel_size * n_rows, panel_size * n_cols),
        squeeze=False,
        sharex='col')

    for j, (nw, adrvs) in enumerate(data.items()):
        for i, obsnum in enumerate(range(n_rows)):
            ax = axes[i, j]
            adrv = adrvs[i]
            if adrv is not None:
                bins = np.arange(-5, 35, 0.5)
                print(adrv)
                ax.hist(adrv, bins=bins)
                vs = []
                for p, c in zip(
                    [3, 10, 50, 90, 97],
                    ['green', 'yellow', 'black', 'yellow', 'green'],
                        ):
                    v = np.nanpercentile(adrv, p)
                    ax.axvline(v, label=f"p{p}={v:.1f}", color=c)
                ax.legend()

    axes[-1, 0].set_xlabel("Best A_drv (dB)")
    axes[-1, 0].set_ylabel("Counts")

    plt.show()

