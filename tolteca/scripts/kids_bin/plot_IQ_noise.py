#!/bin/env python

import numpy as np
from tolteca.datamodels.toltec import BasicObsData


if __name__ == "__main__":

    import sys

    ds = BasicObsData.open(sys.argv[1])
    print(ds)

    kd = ds.read()
    print(kd)
    print(dir(kd))
    print(kd.r)
    print(list(kd.meta.keys()))

    import matplotlib.pyplot as plt


    i = 0

    fig, (ax, bx, cx, dx, ex) = plt.subplots(5, 1)

    ax.plot(kd.meta['f_psd'], kd.meta['I_psd'][i])
    ax.set_xlabel('f_psd')
    ax.set_ylabel('I_psd')
    ax.set_yscale('log')
    ax.set_title('obsnum={obsnum} nwid={nwid}'.format(**kd.meta))

    bx.plot(kd.meta['f_psd'], kd.meta['Q_psd'][i])
    bx.set_xlabel('f_psd')
    bx.set_ylabel('Q_psd')
    bx.set_yscale('log')

    m_f_psd = (kd.meta['f_psd'] > 10.) & (kd.meta['f_psd'] < 40.)
    I_psd_mean = np.mean(kd.meta['I_psd'][:, m_f_psd], axis=1)
    Q_psd_mean = np.mean(kd.meta['Q_psd'][:, m_f_psd], axis=1)
    phi_psd_mean = np.mean(kd.meta['phi_psd'][:, m_f_psd], axis=1)


    cx.plot(kd.meta['tone_axis_data']['f_center'], I_psd_mean, marker='.', linestyle='none')
    cx.set_xlabel('f_probe')
    cx.set_ylabel('I_psd_mean')
    cx.set_yscale('log')

    dx.plot(kd.meta['tone_axis_data']['f_center'], Q_psd_mean, marker='.', linestyle='none')
    dx.set_xlabel('f_probe')
    dx.set_ylabel('Q_psd_mean')
    dx.set_yscale('log')

    ex.plot(kd.meta['tone_axis_data']['f_center'], phi_psd_mean, marker='.', linestyle='none')
    ex.set_xlabel('f_probe')
    ex.set_ylabel('phi_psd_mean')
    ex.set_yscale('log')

    plt.show()
