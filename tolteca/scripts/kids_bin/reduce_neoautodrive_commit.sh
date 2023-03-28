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

echo "commit autodrive results obsnum=${obsnum}"

bin=${scriptdir}/get_ampcor_from_adrv.py
bin_lut=${scriptdir}/add_lut.py
bin_targ_amps=${scriptdir}/get_amps_for_freqs.py
perc=3

obsnum_str=$(printf "%06d" ${obsnum})

for i in $(seq 0 12); do
    # ${pyexec} ${bin} -p ${perc} -- ${scratchdir}/toltec${i}_${obsnum_str}_autodrive.a_drv > ${scratchdir}/toltec${i}_${obsnum_str}_autodrive.log
    ${pybindir}/python ${bin} -p ${perc} -- ${scratchdir}/drive_atten_toltec${i}_${obsnum_str}_adrv.csv > ${scratchdir}/drive_atten_toltec${i}_${obsnum_str}_adrv.log
    ${pybindir}/python ${bin_lut} ${dataroot}/toltec/ics/toltec${i}/toltec${i}_${obsnum_str}_000*_targsweep.nc ${scratchdir}/drive_atten_toltec${i}_${obsnum_str}_autodrive.p${perc}.txt
    cp ${scratchdir}/toltec${i}_${obsnum_str}_autodrive.p${perc}.lut.txt ${scratchdir}/toltec${i}/drive_atten_toltec${i}_${obsnum_str}_default_targ_amps.dat
done
