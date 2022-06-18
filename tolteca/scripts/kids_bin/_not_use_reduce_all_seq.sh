#!/bin/bash

pyexec="${HOME}/toltec_astro/venvs/toltec/bin/python3"
scriptdir=$(dirname "$(readlink -f "$0")")
scratchdir=/data/data_toltec/reduced
bin=$HOME/toltec_astro/tolteca/tolteca/recipes/autodrive.py

if [[ ! $1 ]]; then
	echo "reduce the latest"
    obsnum=$(${pyexec} ${scriptdir}/get_latest_obsnum.py)
	name_pattern="*toltec[0-9]*_*${obsnum}*.nc"
else
	name_pattern="*toltec[0-9]*_*${1}*.nc"
fi
echo $name_pattern
echo $(find /data/data_toltec/ics/ -name "${name_pattern}")
# find /data/data_toltec/ics/ -name "${name_pattern}" | parallel "$HOME/kids_bin/reduce.sh {} -r --output dummy_output"
# for i in $(find /data/data_toltec/ics/ -name "${name_pattern}"); do
for i in /data/data_lmt/toltec/{ics,tcs}/toltec*/${name_pattern}; do
	echo $HOME/kids_bin/reduce.sh $i -r --output dummy_output
	$HOME/kids_bin/reduce.sh $i -r --output dummy_output
done
