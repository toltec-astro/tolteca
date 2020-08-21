#!/bin/bash

pyexec="${HOME}/toltec_astro/venvs/toltec/bin/python3"
scriptdir=$(dirname "$(readlink -f "$0")")
scratchdir=/data/data_toltec/reduced

if [[ -e ${SCRATCHDIR} ]]; then
    scratchdir=${SCRATCHDIR}
fi
echo "use scratch ${scratchdir}"
echo "additional output to: ${scratchdir}"

if [[ $(hostname) == "clipy" ]]; then
    hosts="clipa clipo"
    this=$(basename $0)
    parallel -v ssh -t {} ${scriptdir}/${this} $@ ::: $hosts
    echo 'hosts all done'
    exit 0
fi

if [[ ! $1 ]]; then
    obsnum=$(${pyexec} ${scriptdir}/get_latest_obsnum.py)
    echo found latest obsnum ${obsnum}
else
    obsnum=$1
fi

echo "set adrv of obsnum=${obsnum}"
obsnum_str=$(printf "%06d" ${obsnum})
echo cmd: $(cat ${scratchdir}/toltec_${obsnum_str}_autodrive.cmd)
# nc -q 0 clipy 61559 < ${scratchdir}/toltec_${obsnum_str}_autodrive.cmd
