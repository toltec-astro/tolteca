#!/bin/bash

if [[ ! $2 ]]; then
	echo -e "Compute autodrive ampcor for a single network.\nUsage: $0 roachid obsnum"
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

echo "scratchdir: ${scratchdir}"

pyexec="${HOME}/toltec_astro/venvs/toltec/bin/python3"
bin=$HOME/toltec_astro/tolteca/tolteca/recipes/autodrive.py

roachid=$1
obsnum=$2
obsnum_str=$(printf "%06d" ${obsnum})

raw_obs_files=${datadir}/toltec${roachid}/toltec${roachid}_${obsnum_str}_*targsweep.nc
for i in $raw_obs_files; do
    reportfile=$(basename ${i})
    reportfile=${reportfile%.*}.txt
    echo "reportfile: ${reportfile}"
    reportfile="${scratchdir}/${reportfile}"
    if [[ ! -f ${reportfile} ]]; then
        ${scriptdir}/reduce.sh ${i} -r
    fi
done
reduced_files=${scratchdir}/toltec${roachid}_${obsnum_str}_*targsweep.txt

n_subobs=$(ls ${raw_obs_files} | wc -l)
n_reduced=$(ls ${reduced_files} | wc -l)

echo "autodrive nw=${roachid} obsnum=${obsnum} n_subobs=${n_subobs} n_reduced=${n_reduced}"
files=($reduced_files $raw_obs_files)
echo "# of input files ${#files[@]}"

# build index
obsid=$(printf "%06d" ${obsid0})
indexfile=${scratchdir}/toltec${nwid}_${obsid}_autodrive.index
mv ${indexfile} ${indexfile}.bak
echo "build index ${indexfile}"
${pyexec} ${bin} index ${files[@]} -s "roachid==${roachid} & obsnum == ${obsnum}" -fo ${indexfile}
# run autodrive
ampcorfile=${scratchdir}/toltec${nwid}_${obsid}_autodrive.txt
echo "run autodrive ${ampcorfile}"
${pyexec} ${bin} run -i ${indexfile} -t : -o ${ampcorfile}
