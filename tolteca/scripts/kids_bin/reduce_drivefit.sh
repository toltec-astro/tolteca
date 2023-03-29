#!/bin/bash

pybindir="${HOME}/toltec_astro/extern/pyenv/versions/tolteca_v1/bin"
toltecaexec=${pybindir}/tolteca
scriptdir=$(dirname "$(readlink -f "$0")")
dataroot=/data/data_lmt
scratchdir=${dataroot}/toltec/reduced

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

echo "reduce DriveFit obsnum=${obsnum}"

# obsnum_str=$(printf "%06d" ${obsnum})

config_file=${scriptdir}/kid_phase_fit/config_20230321.yaml
for nw in $(seq 0 13); do
    ${pybindir}/python ${scriptdir}/kid_phase_fit/kid_phase_fit.py $config_file --network ${nw} --obsnum ${obsnum}
done

# generate ampcor files

if [[ $? != 0 ]]; then
    exit 0
fi
