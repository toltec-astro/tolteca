from tolteca.datamodels.toltec import BasicObsData
from tollan.utils.log import timeit, get_logger
from matplotlib import pyplot as plt
import plotly.graph_objs as go
from astropy import units as u
from glob import glob
import numpy as np
import netCDF4
import os
plt.ion()

# This class supports the viewing of Tune data through either
# matplotlib or dasha hooks.  The data is stored in netcdf files
# and stored in a directory referenced by
# the input filepath below.


class Tune():
    def __init__(self, filepath):
        # check that the path exists and that Efficiency files are present
        if not os.path.isdir(filepath):
            raise Exception("No such directory {}".format(filepath))
        flist = glob(os.path.join(filepath, 'Efficiency_N*.nc'))
        print('Found {0:} Efficiency files in {1:}'.format(
            len(flist), filepath))
        if(len(flist) == 0):
            raise Exception("No Efficiency files found in {}".format(filepath))
        flist.sort()
        r10 = [i for i in flist if 'N10' in i]
        r11 = [i for i in flist if 'N11' in i]
        r12 = [i for i in flist if 'N12' in i]
        if(len(r10) > 0):
            flist.sort(key=r10[0].__eq__)
        if(len(r11) > 0):
            flist.sort(key=r11[0].__eq__)
        if(len(r12) > 0):
            flist.sort(key=r12[0].__eq__)

        # generate dictionaries of metadata
        self.nets = {'exists': [0]*13, }
        self.goodNets = []
        for f in flist:
            meta = {}
            try:
                nc = netCDF4.Dataset(f)
                meta['network'] = int(nc.network)
                nc.close()
            except:
                return None
            meta['file'] = f
            if(meta['network'] <= 6):
                meta['array'] = 'a1100'
            elif(meta['network'] >= 11):
                meta['array'] = 'a2000'
            else:
                meta['array'] = 'a1400'
            self.nets['N{}'.format(meta['network'])] = meta
            self.nets['exists'][meta['network']] = 1
            self.goodNets.append('N{}'.format(meta['network']))

    def getArrayData(self):
        # collect fres and efficiency for all available resonances
        for n in self.goodNets:
            nc = netCDF4.Dataset(self.nets[n]['file'])
            self.nets[n]['fres'] = nc.variables['resonantFrequency'][:].data.tolist()
            # gotta remove outliers
            s = nc.variables['efficiency'][:].data
            w = np.where((s < 0) | (s > 0.5))
            s[w] = 0
            self.nets[n]['signal'] = s.tolist()
            self.nets[n]['signal_name'] = 'Efficiency'
            self.nets[n]['signal_units'] = 'unitless'
            nc.close()
        return

    def getNetworkAverageValues(self):
        for n in self.goodNets:
            nc = netCDF4.Dataset(self.nets[n]['file'])
            fr = nc.variables['resonantFrequency'][:].data
            self.nets[n]['fr'] = fr.tolist()
            self.nets[n]['efficiency'] = nc.variables['efficiency'][:].data.tolist()
            nc.close()
        return

    def getArrayAverageValues(self):
        def avgArray(array):
            ap = []
            for n in self.goodNets:
                if(self.nets[n]['array'] == array):
                    ap.append(n)
            eff = []
            for a in ap:
                eff.append(self.nets[n]['efficiency'])
            eff = np.array(eff).mean()
            r = {'meanEff': eff}
            return r
        self.a1100 = avgArray('a1100')
        self.a1400 = avgArray('a1400')
        self.a2000 = avgArray('a2000')
        return

    def matPlotNetworkAvg(self, network, fhigh=500):
        if((network < 0) | (network > 12)):
            print("No such network: {}".format(network))
            return
        # check that the network has Efficiency data
        if(self.nets['exists'][network] == 0):
            print("No Efficiency data for Network {}".format(network))
            return
        plt.ion()
        plt.clf()
        plt.xlim(0, fhigh)
        plt.title('Efficiency for Network {}'.format(network))
        plt.xlabel('Resonant Frequency [MHz]')
        plt.ylabel('Efficiency')
        n = 'N{}'.format(network)
        plt.plot(self.nets[n]['fr'], self.nets[n]['efficiency'])
        return

    def getPlotlyNetworkAvg(self, data, ns):
        fig = go.Figure()
        xaxis, yaxis = getXYAxisLayouts()
        xaxis['title'] = 'Resonnat Frequency [MHz]'
        yaxis['title'] = 'Integral Normalized Spectrum'
        for n in ns:
            if(n in data['goodNets']):
                fr = data['nets'][n]['fr']
                sc = np.array(data['nets'][n]['efficiency'])
                # remove outliers
                w = np.where((sc < 0.5) & (sc > 0.))[0]
                sc = sc[w].tolist()
                fig.add_trace(
                    go.Scattergl(x=fr, y=sc, mode='markers',
                                 name="Network {}".format(n)))
        fig.update_layout(
            uirevision=True,
            showlegend=True,
            width=1200,
            height=400,
            xaxis=xaxis,
            yaxis=yaxis,
            autosize=True,
            margin=dict(
                autoexpand=True,
                l=10,
                r=10,
                t=30,
            ),
            plot_bgcolor='white'
        )
        return fig


# common figure axis definitions
def getXYAxisLayouts():
    xaxis = dict(
        titlefont=dict(size=20),
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor='black',
        linewidth=4,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=18,
            color='rgb(82, 82, 82)',
        ),
    )

    yaxis = dict(
        titlefont=dict(size=20),
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor='black',
        linewidth=4,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=18,
            color='rgb(82, 82, 82)',
        ),
    )
    return xaxis, yaxis


def getEmptyFig(width, height):
    xaxis, yaxis = getXYAxisLayouts()
    fig = go.Figure()
    fig.update_layout(
        uirevision=True,
        showlegend=False,
        width=width,
        height=height,
        xaxis=xaxis,
        yaxis=yaxis,
        autosize=True,
        margin=dict(
            autoexpand=True,
            l=10,
            r=10,
            t=30,
        ),
        plot_bgcolor='white'
    )
    return fig


def _fetchTuneData(filepath):

    with timeit(f"read in processed data from {filepath}"):
        tune = BasicObsData.read(filepath)

    with timeit(f"read in model data from {filepath}"):
        model = BasicObsData.read(filepath.replace('_processed.nc', '.txt'))
        model = model.model

    # generate a set of frequencies to evaluate the model
    f = tune.frequency
    nkids = f.shape[0]
    nsamps = f.shape[1]
    fm = f.copy()
    bw = 40000.*u.Hz
    for i in np.arange(nkids):
        mdf = f[i, :].mean()
        fm[i, :] = np.linspace(mdf-bw, mdf+bw, nsamps)

    # this is idiotic but I'm working on getting answers
    f0 = f.copy()
    fr = np.empty((nkids, 3), dtype='float') * u.Hz
    for i in np.arange(nkids):
        f0[i, :] = np.array([model.f0[i]]*nsamps) * u.Hz
        isort = np.argsort(np.abs(f[i, :].value-model.fr[i]))
        fr[i, :] = np.array([model.fr[i],
                             f[i, isort[0]].value,
                             f[i, isort[1]].value]) * u.Hz

    # derotate both the data and the model sweeps
    S21_orig = tune.S21.value.copy()
    S21 = model.derotate(tune.S21.value, tune.frequency)
    S21_model = model.derotate(model(fm).value, fm)
    S21_resid = S21_orig - model(f).value

    # find derotated S21 values at the resonance and tone frequencies
    S210 = model.derotate(model(f0).value, f0)
    S21r = model.derotate(model(fr).value, fr)
    S21_f0 = S210[:, 0]
    S21_fr = S21r[:, 0]

    # get angle that sweeps two points on either side of fr
    r = S21r[:, 0].real/2.
    phi_fr = (np.arctan2(np.abs(S21r[:, 1].imag), S21r[:, 1].real-r) +
              np.arctan2(np.abs(S21r[:, 2].imag), S21r[:, 2].real-r))

    phi_f0 = np.arctan2(S21_f0.imag, S21_f0.real-r)

    return {
        'network': tune.meta['roachid'],
        'obsnum': tune.meta['obsnum'],
        'subobsnum': tune.meta['subobsnum'],
        'scannum': tune.meta['scannum'],
        'LoCenterFreq': tune.meta['flo_center'],
        'SenseAtten': tune.meta['atten_sense'],
        'DriveAtten': tune.meta['atten_drive'],
        'S21_orig': S21_orig,
        'S21_resid': S21_resid,
        'S21': S21,
        'f': tune.frequency,
        'S21_model': S21_model,
        'f_model': fm,
        'fr': model.fr,
        'f0': model.f0,
        'Qr': model.Qr,
        'S21_f0': S21_f0,
        'S21_fr': S21_fr,
        'phi_fr': phi_fr,
        'phi_f0': phi_f0,
         }
