#! /usr/bin/env python

from dash.dependencies import Output
import dash_html_components as html

from dasha.web.templates import ComponentTemplate
from dasha.web.templates.common import LiveUpdateSection

from tollan.utils.log import get_logger
from tollan.utils import fileloc

import numpy as np
import subprocess
import cachetools.func

from ...datamodels.fs.rsync import RsyncAccessor
from ...datamodels.toltec import BasicObsDataset


TOLTECA_CUSTOM_DISK_STATS_PATHS = [
        'clipa:/data1/data_toltec/',
        'clipa:/data2/data_toltec/',
        'clipo:/data1/data_toltec/',
        'taco:/mnt/data_toltec_2/',
        'taco:/mnt/data_toltec_3/',
        'taco:/mnt/data_toltec_4/',
        'taco:/mnt/data_toltec_5/',
        'taco:/mnt/data_toltec_50/toltec_data_archive/cdl/clipa/',
        'taco:/mnt/data_toltec_50/toltec_data_archive/cdl/clipo/',
        'taco:/mnt/data_toltec_90/toltec_data_archive/cdl/clipa/',
        'taco:/mnt/data_toltec_90/toltec_data_archive/cdl/clipo/',
        ]


accessor = RsyncAccessor()


@cachetools.func.ttl_cache(maxsize=None, ttl=60)
def get_bods_from_rootpath(rootpath):
    logger = get_logger()
    try:
        files = accessor.glob(rootpath)
        # only extract nc files
        files = filter(lambda f: f.endswith('.nc'), files)
        dataset = BasicObsDataset.from_files(list(files))
        # dataset.sort(['ut'])
        logger.debug(f'{len(dataset)} BODs in {rootpath}')
        return dataset
    except Exception as e:
        logger.debug(f"no BOD found in {rootpath}: {e}")
    return None


@cachetools.func.ttl_cache(maxsize=None, ttl=60)
def get_df_of_path(path):
    f = fileloc(path)
    try:
        result = subprocess.check_output([
            'ssh', f'{f.netloc}',
            'df', '-h', f'{f.path}']).decode()
    except Exception as e:
        result = f'Error: {e}'
    return result


class DiskStats(ComponentTemplate):
    _component_cls = html.Div

    logger = get_logger()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # This section of code only runs once when the server is started.
    def setup_layout(self, app):

        container = self
        header_container, body = container.grid(2, 1)
        header = header_container.child(
                LiveUpdateSection(
                    title_component=html.H3("Disk Stats"),
                    interval_options=[60000, 120000],
                    interval_option_value=120000
                    ))

        disk_stats_container = body.grid(1, 1)

        super().setup_layout(app)

        @app.callback(
            [
                Output(disk_stats_container.id, 'children'),
                Output(header.loading.id, 'children'),
                Output(header.banner.id, 'children'),
                ],
            header.timer.inputs
            )
        def update(n_calls):
            result = []
            for p in TOLTECA_CUSTOM_DISK_STATS_PATHS:
                bods = get_bods_from_rootpath(p)
                stats = [html.H5(f'{p}'), ]
                stats.append(html.Pre(get_df_of_path(p)))
                if bods is None:
                    stats.append(
                        html.Pre('No files found')
                        )
                else:
                    print(bods.index_table)
                    i_oldest = np.argmin(bods['ut'])
                    i_latest = np.argmax(bods['ut'])
                    obsnum_range = (
                        f"{bods['obsnum'][i_oldest]}"
                        f"-{bods['obsnum'][i_latest]}")
                    stats.append(
                        html.Pre(
                            f'{len(bods)} BODs, obsnum range {obsnum_range}')
                        )
                result.append(html.Div(stats))
            return result, "", ""
