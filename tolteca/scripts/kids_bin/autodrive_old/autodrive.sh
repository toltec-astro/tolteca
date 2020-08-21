#!/bin/bash

if [[ ! $1 ]]; then
	echo -e "Compute autodrive ampcor for a single network.\nUsage: $0 nwid obsid_start [obsid_stop]"
	exit 1
fi

scriptdir=$(dirname "$(readlink -f "$0")")
datadir=/data/data_toltec/ics
scratchdir=/data/data_toltec/reduced
if [[ -e ${SCRATCHDIR} ]]; then
    scratchdir=${SCRATCHDIR}
fi
echo "use scratch ${scratchdir}"
if [[ -e ${DATADIR} ]]; then
    datadir=${DATADIR}
fi
echo "use data dir ${datadir}"

echo "additional output to: ${scratchdir}"

pyexec="${HOME}/toltec_astro/venvs/toltec/bin/python3"
bin=$HOME/toltec_astro/tolteca/tolteca/recipes/autodrive.py

nwid=$1
runid=$2
obsid0=$2
obsid1=$3

if [[ ! $obsid1 ]]; then
	obsid1=10000000
fi

echo "autodrive nw=${nwid} obsid=${obsid0}:${obsid1}"


files=(${scratchdir}/toltec${nwid}_*targsweep.txt ${datadir}/*/toltec${nwid}_*targsweep.nc)

echo "# of input files ${#files[@]}"

# build index
obsid=$(printf "%06d" ${obsid0})
indexfile=${scratchdir}/toltec${nwid}_${obsid}_autodrive.index
mv ${indexfile} ${indexfile}.bak
echo "build index ${indexfile}"
${pyexec} ${bin} index ${files[@]} -s "(nwid==${nwid}) & (obsid>${obsid0}) & (obsid<${obsid1})" -fo ${indexfile}
# run autodrive
ampcorfile=${scratchdir}/toltec${nwid}_${obsid}_autodrive.txt
echo "run autodrive ${ampcorfile}"
${pyexec} ${bin} run -i ${indexfile} -t : -o ${ampcorfile}
