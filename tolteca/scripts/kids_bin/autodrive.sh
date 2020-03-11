#!/bin/bash

if [[ ! $1 ]]; then
	echo -e "Compute autodrive ampcor for a single network.\nUsage: $0 nwid obsid_start [obsid_stop]"
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

nwid=$1
runid=$2
obsid0=$2
obsid1=$3

if [[ ! $obsid1 ]]; then
	obsid1=10000000
fi

echo "autodrive nw=${nwid} obsid=${obsid0}:${obsid1}"


files=(${scratchdir}/toltec${nwid}*targsweep.txt ${datadir}/*/toltec${nwid}*targsweep.nc)

echo "# of input files ${#files[@]}"

# build index
# echo ${pyexec} ${bin} index ${files[@]} -s "(obsid>${obsid0}) & (obsid<${obsid1})" -fo ${scratchdir}/toltec${nwid}_autodrive.index
obsid=$(printf "%06d" ${obsid0})
indexfile=${scratchdir}/toltec${nwid}_${obsid}_autodrive.index
echo "build index ${indexfile}"
${pyexec} ${bin} index ${files[@]} -s "(obsid>${obsid0}) & (obsid<${obsid1})" -fo ${indexfile}
# run autodrive
ampcorfile=${scratchdir}/toltec${nwid}_${obsid}_autodrive.txt
echo "run autodrive ${ampcorfile}"
${pyexec} ${bin} run -i ${indexfile} -t : -o ${ampcorfile} 
