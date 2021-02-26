#!/bin/bash

pyexec="${HOME}/toltec_astro/venvs/toltec/bin/python3"
scriptdir=$(dirname "$(readlink -f "$0")")
nese_rw_mount=${HOME}/nese_rw
nese_data_lmt_dir=${nese_rw_mount}/test_rsync_data_lmt

toltec_hk_data_ics=/data/data_toltec/./ics/toltec_hk
toltec_hk_data_df=/data/data_toltec/./dilutionFridge
toltec_hk_data_therm=/data/data_toltec/./thermetry
toltec_hk_data_cryo=/data/data_toltec/./cryocmp

# make sure nese_rw is mounted
if ! findmnt -M ${nese_rw_mount} --noheadings |grep nese.rc.umass.edu > /dev/null; then
    echo NESE rw drive is not properly mounted, abort!
    exit 1
fi

rsync -avhPR --append-verify ${toltec_hk_data_ics}  ${nese_data_lmt_dir}
rsync -avhPR --append-verify ${toltec_hk_data_ics} ${toltec_hk_data_df} ${toltec_hk_data_therm} ${toltec_hk_data_cryo} ${nese_data_lmt_dir}
