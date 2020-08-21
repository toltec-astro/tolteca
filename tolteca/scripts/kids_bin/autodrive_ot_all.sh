#!/bin/bash

pyexec="${HOME}/toltec_astro/venvs/toltec/bin/python3"
scriptdir=$(dirname "$(readlink -f "$0")")
scratchdir=/data/data_toltec/reduced
bin=$HOME/toltec_astro/tolteca/tolteca/recipes/autodrive.py

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

echo "autodrive all obsnum=${obsnum}"


parallel ${scriptdir}/autodrive_ot.sh {} ${obsnum} ::: $nws

# collect
obsnum_str=$(printf "%06d" ${obsnum})
# mk links
ampcorfiles=()
adrvfiles=()
for i in $nws; do
    ampcorfile="${scratchdir}/toltec${i}_${obsnum_str}_autodrive.txt"
    ampcorfiles+=(${ampcorfile})
    if [ -f ${ampcorfile} ]; then
        echo ln -rsf ${ampcorfile} ${scratchdir}/toltec${i}_autodrive.txt
        ln -rsf ${ampcorfile} ${scratchdir}/toltec${i}_autodrive.txt
    fi
    adrvfile="${scratchdir}/toltec${i}_${obsnum_str}_autodrive.a_drv"
    adrvfiles+=(${adrvfile})
    if [ -f ${adrvfile} ]; then
        echo ln -rsf ${adrvfile} ${scratchdir}/toltec${i}_autodrive.a_drv
        ln -rsf ${adrvfile} ${scratchdir}/toltec${i}_autodrive.a_drv
    fi
done
outfile=${scratchdir}/toltec_${obsnum_str}_autodrive.txt
echo "super collect result from ${adrvfiles[@]}"
${pyexec} ${bin} collect ${adrvfiles[@]} -fo ${outfile}
