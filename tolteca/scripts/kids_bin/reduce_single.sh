#!/bin/bash

if [[ ! $2 ]]; then
    echo "usage: $0 <nwid> <obsnum>"
    exit 1
fi
nwid=${1}
obsid=$(printf "%06d" ${2})
${HOME}/kids_bin/reduce.sh /data/data_toltec/ics/toltec${nwid}/toltec${nwid}_${obsid}_*.nc $3
