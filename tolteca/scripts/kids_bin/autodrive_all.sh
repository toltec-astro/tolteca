#!/bin/bash

if [[ ! $1 ]]; then
	echo -e "Compute autodrive ampcor file for all networks.\n Usage: $0 obsid_start [obsid_stop]"
	exit 1
fi

scriptdir=$(dirname "$(readlink -f "$0")")
datadir=/data_toltec/repeat
scratchdir=/data_toltec/reduced
if [[ -e ${SCRATCHDIR} ]]; then
    scratchdir=${SCRATCHDIR}
fi
echo "use scratch ${scratchdir}"
echo "additional output to: ${scratchdir}"

pyexec="${HOME}/zma/venvs/toltec/bin/python3"
bin=$HOME/zma/tolteca/tolteca/recipes/autodrive.py

runid=$1
obsid0=$1
obsid1=$2

if [[ ! $obsid1 ]]; then
	obsid1=10000000
fi

echo "autodrive all obsid=${obsid0}:${obsid1}"


seq 0 12 | parallel ${scriptdir}/autodrive.sh {} $@ 

# collect
obsid=$(printf "%06d" ${obsid0})
files=(${scratchdir}/toltec*_${obsid}_autodrive.a_drv)
outfile=${scratchdir}/toltec_${obsid}_autodrive.txt
echo "collect results from ${files[@]} to ${outfile}"
${pyexec} ${bin} collect ${files[@]} -fo ${outfile} 
scp ${outfile} ${files[@]} clipa:/data/data_toltec/reduced/
scp ${outfile} ${files[@]} clipo:/data/data_toltec/reduced/
