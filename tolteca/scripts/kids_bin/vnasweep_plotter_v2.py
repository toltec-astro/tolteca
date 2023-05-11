# code to make rms plots for vna sweep analysis
# by obs, includes all networks where possible

from pathlib import Path
from tollan.utils.fmt import pformat_yaml
from tollan.utils.log import get_logger, init_log
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from tqdm import tqdm
import astropy.units as u
from kidsproc.kidsdata.sweep import MultiSweep
from scipy.ndimage import median_filter
from astropy.table import Table
from tolteca.datamodels.toltec import BasicObsDataset

# from Zhiyuan's code
def calc_chan_rms(vnasweep):
    # an example routine to work with the vna sweep data
    # vnasweep is an instance of VnaSweepData defined in
    # /work/toltec/toltec_astro/toltec-data-product-utilities/toltec_dp_utils/BasicObsData.py
    # Because we want to analyze all channels, we use the n_chan x n_sweepsteps
    # raw object rather than the getSweep(di) function:
    # swp is MultiSweep of size (1000, 492)
    swp = vnasweep.data_io.read()
    fs = swp.frequency.to_value(u.MHz)
    S21_adu = swp.S21.to_value(u.adu)

    # we collect 10 data points at each step, and the stddev is stored
    # in the uncertainty object of S21:
    S21_unc_adu = swp.uncertainty.quantity.to_value(u.adu)

    # here we may want to work in dB scale? 
    # TO DO -- convert to dBm (depends on slice)
    # the unc in db scale is noise / sig
    S21_unc = 20 * np.log10(np.e) * np.abs(S21_unc_adu / S21_adu)

    # per-channel mean and std value of S21_unc
    S21_unc_mean = np.mean(S21_unc, axis=1)
    S21_unc_std = np.std(S21_unc, axis=1)

    # this returns all variable values as a dict
    return locals()


def grab_loopback(date):
    # data all in dB
    # data below taken from S21 data with an external VNA
    # could just make a table with this data
    if date == 2303:
        # measured at site 3/2023
        medLoopback = np.array([-10.1, -10.6, -11.6, -11.32, -12.2, -7, -11.4,
                       -8.8, -13.7, -8.8, np.nan,
                       -13.2, -8.7])
    elif date == 2002:
        # measured at UMass 2/20
        medLoopback = np.array([-4.9, -3.6, -6, -6.2, -6.6, -5.9, -4.3,
                       -4, -3.5, -3, -21.4,
                       -20.4, -5.4]) - 6

    elif date == 2011:
        # measured at UMass 11/20
        medLoopback = np.array([-5.4, -27.2, -7.1, -5.3, -6.4, -5.7, -4.9,
                                -2.8, -6.6, -2.9, -22.,
                                -21.5, -5.5]) - 6.

    elif date == 2101:
        # measured at UMass 1/21
         medLoopback = np.array([-6.2,-23.6, -7.9, -5.7, -6.7, -5.8, -5.,
                       -2.8, -3.7, -2.7, -21.9,
                       -23., -5.5]) - 6
    else:
        print('no matching data found, defaulting to march 2023 lmt data')
        # measured at site 3/2023
        medLoopback = np.array([-10.1, -10.6, -11.6, -11.32, -12.2, -7, -11.4,
                       -8.8, -13.7, -8.8, np.nan,
                       -13.2, -8.7])

    return medLoopback

def compare_rms_vs_gain(data0, data1):
    # grab loopback data -- could do this in a more smart way
    # but easy to do this. only compare two dates!
    medlp0 = data0.med_lb
    medlp1 = data1.med_lb

    # set up data for plot
    dx = medlp1 - medlp0
    dy = data1.med_rms - data0.med_rms

    # network id for the colorbar
    c = np.arange(0, 13)

    # distinct colormap for networks
    cmap = mpl.cm.rainbow
    norm = mpl.colors.BoundaryNorm(np.arange(0, 14), cmap.N)
    norm_n = [norm.__call__(n) for n in c] # need this for the scatter plot edgecolors
    colors = cmap(norm_n) # edgecolors in rgba

    # make figure
    fig, ax = plt.subplots()

    # scatter plots with open circles
    s1 = ax.scatter(medlp0, data0.med_rms, alpha=0.75, facecolors='none', edgecolors=colors)
    s2 = ax.scatter(medlp1, data1.med_rms, alpha=0.75, facecolors='none', edgecolors=colors)

    # make quiver plot going from first value to second
    im = ax.quiver(medlp0, data0.med_rms, dx, dy, c,
                    scale=1, scale_units='xy', angles='xy',
                    cmap=cmap, norm=norm)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)

    # set background color to be more readable
    ax.set_facecolor('whitesmoke')

    # labels, etc.
    plt.xlabel('Median loopback gain [dB]')
    plt.ylabel('Median of mean channel uncertainties [dB]')
    plt.title('compare window closed vs open vna sweep noise')
    plt.grid()#color='whitesmoke')

    # will exclude some low gain networks
    plt.xlim(-14, -8.)
    plt.ylim(0, 0.2)

    return

class NS:
    pass

class vna_rms:

    def __init__(self,
                 bods,
                 year=2303):
        obsnum = bods.index_table['obsnum'][0]
        subobsnum = bods.index_table['subobsnum'][0]
        scannum = bods.index_table['scannum'][0]
        name = f"{obsnum}-{subobsnum}-{scannum}"

        # features
        nets = np.arange(0,13)
        n_chans = 1000

        self.obsnum = obsnum
        self.name = name
        self.nets = nets
        self.year = year

        # make our empty arrays to fill
        self.rms_data = np.full((len(nets), n_chans), np.nan)
        self.rms_data_adu = np.full((len(nets), n_chans), np.nan)
        self.med_rms = np.full(len(nets), np.nan)
        self.los = np.full(len(nets), np.nan)
        self.check = np.full(len(nets), 0.)

        # set up to grab data
        self.bods = bods
        self.obs_list = np.empty(13)
        self.file_list = np.empty(13, dtype=np.dtype('U100'))

        # update and get data
        self._update()

        return

    def _update(self):
        # need this case if obsnums are different for diff networks
        self.t = self.bods.index_table
        #print(t)
        # load data, rewrites the vna dict which might be the slowest part

        def load_data(bod):
            r = NS()
            r.data_io = bod
            r.meta = bod.meta
            return r

        vna = [load_data(bod) for bod in self.bods]

        # loop through nets
        for v in vna:
            # if network has no data, do not include
            # otherwise grab the rms values
            n = v.meta['nwid']
            self.obs_list[n] = v.meta['obsnum'] # save obsnum for if they are different
            self.file_list[n] = v.meta['filename_orig'].split('/')[-1]

            # add a dictionary value with the rms
            # can edit to be another rms method
            chan_rms_result = calc_chan_rms(v)

            # get frequencies -- I'm assuming all use the same roach freqs
            freqs = chan_rms_result['fs']
            freq_cent = freqs[:, freqs.shape[1] // 2] - v.meta['flo_center'] / 1e6

            # sort the data
            isort = np.argsort(freq_cent)
            freq_cent_sort = freq_cent[isort]

            # save the sorted rms data
            all_chans = chan_rms_result['S21_unc_mean'][isort]
            self.rms_data[n] = all_chans
            self.med_rms[n] = np.median(all_chans)

            # and saving the ADU data
            all_chans_adu = np.mean(np.abs(chan_rms_result['S21_unc_adu']), axis=1)[isort]
            self.rms_data_adu[n] = all_chans_adu

            # check that LO issue hasn't popped in
            all_chans_dB = 20. * np.log10(np.abs(chan_rms_result['S21_adu']))
            abs_diffs = np.diff(all_chans_dB)
            abs_num = 0.

            # check to see if there are differences > 0.5 dB
            # if so, if there are between 2 and 19 spikes, accept it
            # if there are less, most likely not LO
            # if there are more, most likely atten or other issue
            for a in abs_diffs:
                w = np.where(a > 0.5)[0]
                if ((len(w) > 1) and (len(w) < 20)):
                    abs_num+=1

            self.check[n] = abs_num / 1000 # what number of channels have diffs > 0.5 dB?

            # if you want to correct the vna plots to be by network not slice
            # do something like this
            #if ((17502 in self.obsnum) & (n == 5)):
            #    self.rms_data[2] = all_chans
            #    self.rms_data[5] = np.full(len(all_chans), np.nan)

            #    self.rms_data_adu[2] = all_chans_adu
            #    self.rms_data_adu[5] = np.full(len(all_chans), np.nan)

            #    self.med_rms[2] = self.med_rms[5]
            #    self.med_rms[5] = np.nan

        # save the frequencies
        self.freqs = freq_cent_sort

        # get the loopback data
        self.med_lb = grab_loopback(self.year)
        return

    def medS21_plot(self, units='dB'):
        # what units?
        if units == 'dB':
            med_rms = self.med_rms
            y_low = 0
            y_high = 0.2

        elif units == 'dBm':
            med_rms = self.med_rms - 162.4 # conversion factor to dBm

            y_low = -162.3
            y_high = -162.9

        # network id for the colorbar
        c = np.arange(0, 13)

        # distinct colormap for networks
        cmap = mpl.cm.rainbow
        norm = mpl.colors.BoundaryNorm(np.arange(0, 14), cmap.N)

        # make figure
        fig, ax = plt.subplots()
        im = plt.scatter(self.med_lb, med_rms, c=c, cmap=cmap, norm=norm)
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)

        # labels, etc.
        plt.xlabel('Median loopback gain [dB]')
        plt.ylabel(f'Median rms value [{units}]')
        plt.title(f'{self.name}')
        plt.grid()

        # will exclude some low gain networks
        plt.xlim(-15, -5)
        plt.ylim(y_low, y_high)

        return locals()

    def noise_plot(self, units='dB'):
        # what units?
        if units == 'dB':
            rms_data = self.rms_data

            # might want to hard code these for consistency between obs
            vmin = 0 #np.quantile(rms_data, 0.01)
            vmax = 0.1 #np.quantile(rms_data, 0.99)

        elif units == 'dBm':
            rms_data = self.rms_data - 162.4 # conversion factor to dBm

            # might want to hard code these for consistency between obs
            w = np.isnan(rms_data)
            vmin = np.quantile(rms_data[~w], 0.01)
            vmax = np.quantile(rms_data[~w], 0.99)

        # have spot for each network
        nets=np.arange(0,13)

        # looking only at vna sweeps, get 1000 channels
        n_chans=np.shape(rms_data)[1]

        # make figure
        fig, ax = plt.subplots(figsize=(16,7))

        # make map with colorbar
        im = ax.imshow(rms_data, vmin=vmin, vmax=vmax, 
                       interpolation='none', aspect='auto',
                       cmap = 'RdYlGn_r')#mpl.colormaps['RdYlGn_r'])
        cbar = plt.colorbar(im)

        # labels and ticks and more
        ax.set_yticks(nets)
        ax.set_yticklabels(nets)
        ax.set_xticks(list(range(n_chans))[::50])

        # xaxis is in downconverted frequencies [Hz]
        sort_labels = [f'{freq:.2f}' for freq in self.freqs[::50]]
        ax.set_xticklabels(sort_labels)

        ax.set_title(f'{self.name}')
        ax.set_xlabel('channel center frequency [Hz]')
        ax.set_ylabel('network')

        # once set conversion to dBm change this unit
        cbar.set_label(f'channel rms [{units}]')

        # check that data is not spikey
        w = np.where(self.check > 0.15)[0]
        if len(w) > 0:
            ax.text(0.2, 0.5, 'SPIKES FOUND IN SWEEP, CHECK SETTINGS',
                    style='normal', transform=ax.transAxes, fontsize=20,
                    bbox={'facecolor': 'red', 'pad': 10})

        return locals()

    # plot data in ADUs
    # can overplot with other networks if you save the axis to local
    def adu_freq_plot(self, nw=0, ax=None):
        if ax == None:
            fig, ax = plt.subplots(figsize=(10,10))

            # labels
            ax.set_title(f'{self.name}')
            ax.set_xlabel('channel center frequency [Hz]')
            ax.set_ylabel('channel rms [ADU]')

            # make easier to see
            ax.grid()

        p = ax.plot(self.freqs, self.rms_data_adu[nw][:], linestyle='none', marker='.',
                    label=f'sl{nw}')

        ax.legend(loc='upper right')
        ax.set_yscale('log')

        return ax

    # saves all the data for each network to a separate ecsv file
    # with the same filename as its original
    def save_data(self, path='./'):
        # loop through slices
        for n in np.arange(0, 13):
            # open a table and make the columns
            data = Table()
            data['ch_cent_freq_hz'] = self.freqs
            data['chan_rms_ADU'] = self.rms_data_adu[n][:]
            data['chan_rms_db'] = self.rms_data[n][:]
            data['chan_rms_dbm'] = self.rms_data[n][:] - 162.4
            data['chan_flag'] = np.zeros_like(self.rms_data[n][:])

            # assuming only looking at nc files
            # can split original name at .nc and change extension
            # but don't save a file if there is no data
            if self.file_list[n] == '':
                print(f'slice {n} has no data')
            else:
                filename = self.file_list[n].split('.')[0]
                data.write(path + f'{filename}_chan_rms.ecsv', overwrite=True)
        return


def collect_data(data_rootpath, obsnums):
    logger = get_logger()
    data_files = []
    for obsnum in obsnums:
        try:
            obsnum = int(obsnum)
        except ValueError:
            pass
        if isinstance(obsnum, int):
            obsnum = f'{obsnum:06d}'
        for p in [
            f'toltec/?cs/toltec*/toltec*_{obsnum}_*sweep.nc',
            f'toltec/?cs/toltec*/toltec*_{obsnum}_*tune.nc',
            ]:
            logger.debug(f"collect file from pattern {p} in {data_rootpath}")
            data_files.extend(data_rootpath.glob(p))

    logger.debug(f"collected files:\n{data_files}")
    logger.info(f"collected {len(data_files)} files")
    return BasicObsDataset.from_files(data_files)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("obsnum")
    parser.add_argument("--data_rootpath", '-d', type=Path)
    parser.add_argument("--log_level", default='INFO')
    parser.add_argument("--save_plot", action="store_true")

    option = parser.parse_args()

    logger = get_logger()
    init_log(level=option.log_level)
    import logging
    for n in ['NcFileIO', '_KidsDataAxisSlicer', 'ncopen', 'open']:
        logging.getLogger(n).disabled = True


    data_rootpath = option.data_rootpath or Path('data_lmt')

    obsnum = int(option.obsnum)

    logger.info(f"run for {obsnum=}")

    bods = collect_data(data_rootpath, [obsnum])
    year = 2303 #2011 #2101
    # testing out the class
    v = vna_rms(bods=bods, year=year)

    # plotting
    rms_plot = 1
    wf_plot = 1

    # make save path
    obsnum = bods.index_table['obsnum'][0]
    subobsnum = bods.index_table['subobsnum'][0]
    scannum = bods.index_table['scannum'][0]
    outdir = data_rootpath.joinpath(f"toltec/reduced/{obsnum}")
    if not outdir.exists():
        outdir.makedirs(exist_ok=True)
    outname = f"sweepcheck_{obsnum:06d}_{subobsnum:03d}_{scannum:04d}"
    # rms plot
    if rms_plot:
        ctx = v.medS21_plot()
        if option.save_plot:
            save_filepath=outdir.joinpath(outname + "_rms.png")
            ctx['fig'].savefig(save_filepath)
        else:
            plt.show()

    # noise plot at each frequency
    if wf_plot:
        # make plot
        ctx = v.noise_plot()
        if option.save_plot:
            save_filepath=outdir.joinpath(outname + "_noise.png")
            ctx['fig'].savefig(save_filepath)
        else:
            plt.show()
