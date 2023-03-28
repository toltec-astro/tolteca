#!/bin/bash

pyexec="${HOME}/toltec_astro/venvs/toltec/bin/python3"
scriptdir=$(dirname "$(readlink -f "$0")")
scratchdir=/data/data_toltec/reduced

if [[ -e ${SCRATCHDIR} ]]; then
    scratchdir=${SCRATCHDIR}
fi
echo "use scratch ${scratchdir}"
echo "additional output to: ${scratchdir}"


if [[ ! $1 ]]; then
    obsnum=$(${pyexec} ${scriptdir}/get_latest_obsnum.py)
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
    ${pyexec} ${bin} -p ${perc} -- ${scratchdir}/drive_atten_toltec${i}_${obsnum_str}_adrv.csv > ${scratchdir}/drive_atten_toltec${i}_${obsnum_str}_adrv.log
    ${pyexec} ${bin_lut} /data_lmt/toltec/ics/toltec${i}/toltec${i}_${obsnum_str}_000*_targsweep.nc ${scratchdir}/drive_atten_toltec${i}_${obsnum_str}_autodrive.p${perc}.txt
    # cp ${scratchdir}/toltec${i}_${obsnum_str}_autodrive.p${perc}.lut.txt /home/toltec/tlaloc/etc/toltec${i}/default_targ_amps.dat
done
