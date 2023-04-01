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
    adrv_file=${scratchdir}/drive_atten_toltec${i}_${obsnum_str}_adrv.csv
    adrv_log=${scratchdir}/drive_atten_toltec${i}_${obsnum_str}_adrv.log
    if ! [ -f ${adrv_file} ]; then
        echo "skip nw=${i}, no adrv.csv file found."
        continue
    fi

    set -x
    ${pybindir}/python ${bin} -p ${perc} -- ${adrv_file} > ${adrv_log}
    ${pybindir}/python ${bin_lut} \
        ${dataroot}/toltec/*/toltec${i}/toltec${i}_${obsnum_str}_001*_targsweep.nc  ${scratchdir}/drive_atten_toltec${i}_${obsnum_str}_adrv.p${perc}.txt
    cp ${scratchdir}/drive_atten_toltec${i}_${obsnum_str}_adrv.p${perc}.lut.txt ${scratchdir}/drive_atten_toltec${i}_${obsnum_str}_default_targ_amps.dat
    set +x
    if (( i <= 6 )); then
        dest=clipa
    else
        dest=clipo
    fi
    scp ${scratchdir}/drive_atten_toltec${i}_${obsnum_str}_default_targ_amps.dat clipa:/home/toltec/tlaloc/etc/toltec${i}/
    echo "~~~~~~~ drivefit result commited to dest=${dest} nw=${i}"
done
