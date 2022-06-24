#!/bin/bash

pyexec="${HOME}/toltec_astro/venvs/toltec/bin/python3"
scriptdir=$(dirname "$(readlink -f "$0")")
nese_rw_mount=${HOME}/nese_rw
# nese_data_lmt_dir=${nese_rw_mount}/test_rsync_data_lmt
nese_data_lmt_dir=${nese_rw_mount}/02_data_lmt

if [[ $(hostname) == "clipa" ]]; then
    echo "rsync to nese from clipa"
elif [[ $(hostname) == "clipo" ]]; then
    echo "rsync to nese from clipo"
elif [[ $(hostname) == "clipy" ]]; then
    hosts="clipa clipo"
    this=$(basename $0)
    parallel -v ssh -t {} ${scriptdir}/${this} $@ ::: $hosts
    echo 'hosts all done'
    # exit 0
else
    echo "invalid host"
    exit 1
fi

if [[ ! $1 ]]; then
    echo "Usage: rsync_to_nese.sh <obsnum>"
    exit 1
fi
# make sure nese_rw is mounted
if ! findmnt -M ${nese_rw_mount} --noheadings |grep nese.rc.umass.edu > /dev/null; then
    echo NESE rw drive is not properly mounted, abort!
    exit 1
fi
${pyexec} /home/toltec/toltec_astro/tolteca/tolteca/recipes/dataset_rsync2.py -s "obsnum>=$1 & master_name==\"$2\"" -m lmt_archive -d ${nese_data_lmt_dir} \
    -fo nese_rsync_$(date +"%Y%m%dT%H%M%S").index \
    '/data_lmt/toltec/ics' \
    '/data_lmt/toltec/tcs' \
    '/data_lmt/tel'
    # '/data_lmt/toltec/ics/toltec*/*.nc' \
    # '/data_lmt/toltec/ics/wyatt*/*.nc' \
    # '/data_lmt/toltec/tcs/toltec*/*.nc' \
    # '/data_lmt/tel/*.nc'
