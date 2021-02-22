#!/bin/bash

pyexec="${HOME}/toltec_astro/venvs/toltec/bin/python3"
scriptdir=$(dirname "$(readlink -f "$0")")
nese_data_lmt_dir=astrotolteclab@nese.rc.umass.edu:/mount/test_rsync_data_lmt

if [[ $(hostname) == "clipa" ]]; then
    echo "rsync to nese from clipa"
elif [[ $(hostname) == "clipo" ]]; then
    echo "rsync to nese from clipo"
elif [[ $(hostname) == "clipy" ]]; then
    hosts="clipa clipo"
    this=$(basename $0)
    parallel -v ssh -t {} ${scriptdir}/${this} $@ ::: $hosts
    echo 'hosts all done'
    exit 0
else
    echo "invalid host"
    exit 1
fi

if [[ ! $1 ]]; then
    echo "Usage: rsync_to_nese.sh <obsnum>"
    exit 1
fi

${pyexec} /home/toltec/toltec_astro/tolteca/tolteca/recipes/dataset_rsync2.py -s "obsnum>=$1" -m lmt_archive -d ${nese_data_lmt_dir} \
    -fo nese_rsync_$(date +"%Y%m%dT%H%M%S").index /data/data_toltec/ics/toltec*/*.nc
