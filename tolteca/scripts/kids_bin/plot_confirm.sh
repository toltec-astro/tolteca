#!/bin/bash

before=$1
after=$2
for i in $before $after; do
    obsid=$(printf "%06d" ${i})
    python ${HOME}/toltec_astro/tolteca/tolteca/recipes/autodrive.py collect \
        /data/data_toltec/reduced/toltec*_${obsid}_autodrive.a_drv -fo /dev/null --plot &
done
