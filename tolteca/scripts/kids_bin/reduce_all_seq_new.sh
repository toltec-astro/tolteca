#!/bin/bash

pybindir="${HOME}/toltec_astro/extern/pyenv/versions/tolteca_v1/bin"
toltecaexec=${pybindir}/tolteca
scriptdir=$(dirname "$(readlink -f "$0")")
dataroot=/data/data_lmt
scratchdir=${dataroot}/toltec/reduced
rcdir=$HOME/toltec_astro/run/tolteca/beammap

if [[ -e ${SCRATCHDIR} ]]; then
    scratchdir=${SCRATCHDIR}
fi
echo "use scratch ${scratchdir}"
echo "additional output to: ${scratchdir}"

if [[ ! $1 ]]; then
    obsnum=$(${pybindir}/python3 ${scriptdir}/get_latest_obsnum.py)
    echo found latest obsnum ${obsnum}
else
    obsnum=$1
fi

echo "reduce kids obsnum=${obsnum}"

obsnum_str=$(printf "%06d" ${obsnum})

for i in ${dataroot}/toltec/{ics,tcs}/toltec*/toltec*_${obsnum_str}_*.nc; do
    echo ${scriptdir}/reduce.sh $i -r --output dummy_output
    ${scriptdir}/reduce.sh $i -r --output dummy_output
done
