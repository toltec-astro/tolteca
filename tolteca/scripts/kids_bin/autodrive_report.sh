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

obsnum_str=$(printf "%06d" ${obsnum})

if [[ $(hostname) == "clipa" ]]; then
    cat ${scratchdir}/toltec*_${obsnum_str}_autodrive.log | grep a_drv_ref | awk '{print NR-1, $2}'
elif [[ $(hostname) == "clipo" ]]; then
    cat ${scratchdir}/toltec*_${obsnum_str}_autodrive.log | grep a_drv_ref | awk '{print NR+6, $2}'
elif [[ $(hostname) == "clipy" ]]; then
    hosts="clipa clipo"
    this=$(basename $0)
    for host in ${hosts}; do
        ssh -t $host ${scriptdir}/${this} $@ ::: $hosts
    done
    exit 0
else
    echo "invalid host"
    exit 1
fi
