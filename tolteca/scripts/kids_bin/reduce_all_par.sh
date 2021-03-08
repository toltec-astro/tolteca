#!/bin/bash

if [[ ! $1 ]]; then
	echo "reduce the latest"
	name_pattern='*toltec*[0-9].nc'
else
	name_pattern="*toltec*[0-9]${1}*.nc"
fi
echo $name_pattern
echo $(find /data/data_toltec/ics/ -name "${name_pattern}")
find /data/data_toltec/ics/ -name "${name_pattern}" | parallel "$HOME/kids_bin/reduce.sh {} -r --output dummy_output"
# for i in $(find /data/data_toltec/ics/ -name "${name_pattern}"); do
# 	echo $HOME/kids_bin/reduce.sh $i -r --output dummy_output
# 	$HOME/kids_bin/reduce.sh $i -r --output dummy_output
# done
