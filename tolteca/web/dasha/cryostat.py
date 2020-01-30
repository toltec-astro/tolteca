#! /usr/bin/env python
import numpy as np


__all__ = [
        'label',
        'template',
        'title_text',
        'title_icon',
        'update_interval',
        'sources',
        ]


label = 'cryostats'
template = 'live_ncfile_view'
title_text = 'Cryostat'
title_icon = 'fas fa-object-group'

update_interval = 30 * 1000  # ms


def _F2K(t):
    return (t - 32.) * 5. / 9. + 273.15


def _C2K(t):
    return t + 273.15


def _make_trace(interface, type_, var_name):
    dispatch_name = {
            'cryocmp': "Compressor",
            'dltfrg': "Dilution Fridge",
            }
    dispatch_runtime_link = {
            "cryocmp": "/data_toltec/cryocmp/cryocmp.nc",
            "dltfrg": "/data_toltec/dilutionFridge/dilutionFridge.nc",
            }
    dispatch_x = {
            'cryocmp': "Data.ToltecCryocmp.Time",
            'dltfrg': 'Data.ToltecDilutionFridge.SampleTime',
            }
    dispatch_ylabel = {
            'temp': "Temperature (K)",
            'pres': "Pressure (kPa?)",
            'flag': 'Flag',
            }
    dispatch_trans = {
            "cryocmp": lambda x, y: (x, _F2K(y)),
            "dltfrg": lambda x, y: (x, _C2K(y)),
            }
    return {
                 'name': dispatch_name[interface] + " " + var_name.rsplit('.', 1)[-1],
                 'runtime_link': dispatch_runtime_link[interface],
                 'x': dispatch_x[interface],
                 'y': var_name,
                 'x_dtype': 'posixtime',
                 'trans': dispatch_trans[interface],
                 'x_label': 'Time',
                 'y_label': dispatch_ylabel.get(type_, type_),
                 'mode': 'lines+markers',
                 'type': 'scattergl',
            }


def _make_source(group, traces):
    dispatch_title = {
            "temp": "Temperatures",
            "pres": "Pressures",
            "stat": "Status",
            }
    dispatch_controls = {
            "temp": ['toggle-collate', ],
            "pres": ['toggle-collate', ],
            }
    dispatch_fig_layout = {
            "temp": {
                'yaxis': {
                    'type': 'log',
                    },
                },
            }
    return {
            "label": f"{group}",
            "title": dispatch_title[group],
            'local_tz': "EST",
            'controls': ['toggle-tz', ] + dispatch_controls.get(group, list()),
            'fig_layout': dict(
                legend_orientation="h",
                legend=dict(y=-0.2),
                **dispatch_fig_layout.get(group, dict())),
            'traces': [_make_trace(*trace) for trace in traces] 
            }

sources = [_make_source(*source) for source in [
        ("temp", (
                   ("dltfrg", "temp", "Data.ToltecDilutionFridge.StsDevC1PtcSigWit"),
                   ("dltfrg", "temp", "Data.ToltecDilutionFridge.StsDevC1PtcSigWot"),
                   ("dltfrg", "temp", "Data.ToltecDilutionFridge.StsDevC1PtcSigOilt"),
                   ("dltfrg", "temp", "Data.ToltecDilutionFridge.StsDevC1PtcSigHt"),
                   ("cryocmp", "temp", "Data.ToltecCryocmp.CoolInTemp"),
                   ("cryocmp", "temp", "Data.ToltecCryocmp.CoolOutTemp"),
                   ("cryocmp", "temp", "Data.ToltecCryocmp.OilTemp"),
                   ("cryocmp", "temp", "Data.ToltecCryocmp.HeliumTemp"),
                )
        ),
        ("pres", (
                   ("dltfrg", "pres", "Data.ToltecDilutionFridge.StsDevC1PtcSigHlp"),
                   ("dltfrg", "pres", "Data.ToltecDilutionFridge.StsDevC1PtcSigHhp"),
                   ("cryocmp", "pres", "Data.ToltecCryocmp.LoPressure"),   
                   ("cryocmp", "pres", "Data.ToltecCryocmp.LoPressureAvg"),
                   ("cryocmp", "pres", "Data.ToltecCryocmp.HiPressure"),
                   ("cryocmp", "pres", "Data.ToltecCryocmp.HiPressureAvg"),
                   ("cryocmp", "pres", "Data.ToltecCryocmp.DeltaPressureAvg"),
                 )
        ),
        ("stat", (
                   ("dltfrg", "Value", "Data.ToltecDilutionFridge.Status"),
                   ("dltfrg", "Value", "Data.ToltecDilutionFridge.Heartbeat"),
                   ("dltfrg", "Power (W?)", "Data.ToltecDilutionFridge.StsDevTurb1PumpSigPowr"),
                   ("dltfrg", "Speed (mL/s?)", "Data.ToltecDilutionFridge.StsDevTurb1PumpSigSpd"),
                   ("cryocmp", "Value", "Data.ToltecCryocmp.Heartbeat"),
                   ("cryocmp", "Value", "Data.ToltecCryocmp.Enabled"),
                   ("cryocmp", "Value", "Data.ToltecCryocmp.OperState"),
                   ("cryocmp", "Value", "Data.ToltecCryocmp.Energized"),
                   ("cryocmp", "Value", "Data.ToltecCryocmp.Warnings"),
                   ("cryocmp", "Value", "Data.ToltecCryocmp.Alarms"),
                   ("cryocmp", "Current (A)", "Data.ToltecCryocmp.MotorCurrent"),
                   ("cryocmp", "Hours of Operation", "Data.ToltecCryocmp.OpHours"),
                 )
        ),
        ]
    ]
