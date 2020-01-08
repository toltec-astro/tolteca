#! /usr/bin/env python
from tolteca.utils.log import get_logger
from dasha.web.frontend.templates.ncscope import NcScope
from cached_property import cached_property
import numpy as np


__all__ = [
        'label',
        'template',
        'title_text',
        'title_icon',
        'fig_layout',
        'update_interval',
        'source',
        ]

logger = get_logger()

label = 'thermometry'
template = 'live_ncfile_view'
title_text = 'Thermometry'
title_icon = 'fas fa-thermometer-half'

fig_layout = dict(
    yaxis={
        'type': 'log',
        'autorange': True,
        'title': 'Temperature (K)'
        },
    )

update_interval = 30 * 1000  # ms


class Thermetry(NcScope):

    logger = get_logger()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def v_time(self, i):
        return self.nc.variables[f'Data.ToltecThermetry.Time{i + 1}']

    def v_temp(self, i):
        return self.nc.variables[f'Data.ToltecThermetry.Temperature{i + 1}']

    def v_resis(self, i):
        return self.nc.variables[f'Data.ToltecThermetry.Resistance{i + 1}']

    def n_times(self):
        return self.nc.dimensions['times'].size

    @cached_property
    def n_channels(self):
        return self.nc.dimensions[
                'Header.ToltecThermetry.ChanLabel_xlen'].size

    @cached_property
    def channel_labels(self):
        strlen = self.nc.dimensions[
                'Header.ToltecThermetry.ChanLabel_slen'].size
        return list(map(
            lambda x: x.decode().strip(), self.nc.variables[
                'Header.ToltecThermetry.ChanLabel'][:].view(
                f'S{strlen}').ravel()))


def get_traces(src):
    tm = Thermetry.from_link(src['runtime_link'])
    tm.sync()
    n_times = 100
    result = []
    for i in range(tm.n_channels):
        result.append({
            'x': np.asarray(tm.v_time(i)[-n_times:], dtype='datetime64[s]'),
            'y': tm.v_temp(i)[-n_times:],
            'name': tm.channel_labels[i],
            'mode': 'lines+markers',
            'type': 'scatter'
        })
    try:
        time_latest = np.max([t['x'][-1] for t in result if len(t['x']) > 0])
    except RuntimeError:
        logger.warning(f"data file {tm} is empty")
        return list()
    else:
        for t in result:
            mask = np.where(
                    (t['x'] >= (time_latest - np.timedelta64(24, 'h'))) &
                    (t['y'] > 0.))[0]
            t['x'] = t['x'][mask]
            t['y'] = t['y'][mask]
    return result


source = {
    'label': 'all_thermometers',
    # 'runtime_link': '/data_toltec/thermetry/thermetry.nc',
    'runtime_link': '/Users/ma/Codes/toltec/kids/test_data/thermetry.nc',
    'local_tz': 'EST',
    'traces': get_traces,
    'controls': ['toggle-collate', 'toggle-ut'],
    }


# source = {
#     'label': 'tel',
#     # 'runtime_link': '/data_toltec/thermetry/thermetry.nc',
#     'runtime_link': '/Users/ma/Codes/toltec/kids/test_data/thermetry.nc',
#     'local_tz': 'EST',
#     'n_points': 100
#     'traces': [
#             {
#                 "x": 'Data.Lmt.RA',
#                 'y': 'Data.Lmt.Dec',
#                 'x_label': 'RA',
#                 'y_label': 'Dec',
#                 'name': 'Telescope Position (2D)',
#                 'mode': 'lines+markers',
#                 'type': 'scatter',
#                 },

#             {
#                 "x": 'Data.TempTime'%i,
#                 'x_type': 'Time',
#                 # 'x_range': slice(None, None),
#                 'x_slice': lambda v: v[::100]
#                 'y': 'Data.TempValue{i}',
#                 'y_trans': lambda x: x * 100.,
#                 'y_label': 'RA (arcsec)'
#                 'name': 'RA',
#                 'mode': 'lines+markers',
#                 'type': 'scatter',
#                 'data_mask': lambda x, y: x > x[-1] - 24.
#                 'x_label': 'Header...ChanLabel'
#                 ''
#                 # x = v[slice]   # reading
#                 # x = x[mask]    # 
#                 },
#             {
#                 "x": 'Data.Lmt.Time',
#                 'x_type': 'Time',
#                 # 'x_range': slice(None, None),
#                 'x_slice': lambda v: v[::100]
#                 'y': 'Data.Lmt.RA',
#                 'y_trans': lambda x: x * 100.,
#                 'y_label': 'RA (arcsec)'
#                 'name': 'RA',
#                 'mode': 'lines+markers',
#                 'type': 'scatter',
#                 'data_mask': lambda x, y: x > x[-1] - 24.
#                 'x_label': 'Header...ChanLabel'
#                 ''
#                 # x = v[slice]   # reading
#                 # x = x[mask]    # 
#                 },
#             {
#                 "x": 'Data.Lmt.Time',
#                 'y': 'Data.Lmt.Dec',
#                 'name': 'RA',
#                 'mode': 'lines+markers',
#                 'type': 'scatter',
#                 },
#         ],
#     }
