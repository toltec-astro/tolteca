#!/bin/bash

pyexec="${HOME}/toltec_astro/venvs/toltec/bin/python3"
scriptdir=$(dirname "$(readlink -f "$0")")
scratchdir=/data/data_toltec/reduced

if [[ -e ${SCRATCHDIR} ]]; then
    scratchdir=${SCRATCHDIR}
fi
echo "use scratch ${scratchdir}"
echo "additional output to: ${scratchdir}"


if [[ $(hostname) == "clipa" ]]; then
    nws=$(seq 0 6)
elif [[ $(hostname) == "clipo" ]]; then
    nws=$(seq 7 12)
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
    obsnum=$(${pyexec} ${scriptdir}/get_latest_obsnum.py)
    echo found latest obsnum ${obsnum}
else
    obsnum=$1
fi

echo "commit autodrive results obsnum=${obsnum}"

bin=${scriptdir}/get_ampcor_from_adrv.py
perc=3

obsnum_str=$(printf "%06d" ${obsnum})
for i in $nws; do
    echo "+++++++++++++ $(hostname) +++ toltec$i ++++++++++++++"
    echo cp ${scratchdir}/toltec${i}_${obsnum_str}_autodrive.txt /home/toltec/roach/etc/toltec${i}/default_targ_amps.dat
    ${pyexec} ${bin} -p ${perc} -- ${scratchdir}/toltec${i}_${obsnum_str}_autodrive.a_drv > ${scratchdir}/toltec${i}_${obsnum_str}_autodrive.log
    cp ${scratchdir}/toltec${i}_${obsnum_str}_autodrive.p${perc}.txt /home/toltec/roach/etc/toltec${i}/default_targ_amps.dat
done
