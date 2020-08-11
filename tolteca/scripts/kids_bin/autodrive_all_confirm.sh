#!/bin/bash

if [[ ! $1 ]]; then
	echo -e "Compute autodrive ampcor file for all networks.\n Usage: $0 obsid_start [obsid_stop]"
	exit 1
fi

scriptdir=$(dirname "$(readlink -f "$0")")
datadir=/data/data_toltec/ics
scratchdir=/data/data_toltec/reduced
if [[ -e ${SCRATCHDIR} ]]; then
    scratchdir=${SCRATCHDIR}
fi
echo "use scratch ${scratchdir}"
echo "additional output to: ${scratchdir}"

pyexec="${HOME}/toltec_astro/venvs/toltec/bin/python3"
bin=$HOME/toltec_astro/tolteca/tolteca/recipes/autodrive.py

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
corfiles=(${scratchdir}/toltec*_${obsid}_autodrive.txt)
files=(${scratchdir}/toltec*_${obsid}_autodrive.a_drv)
outfile=${scratchdir}/toltec_${obsid}_autodrive.txt
echo "collect results from ${files[@]} to ${outfile}"
${pyexec} ${bin} collect ${files[@]} -fo ${outfile} 
rsync -avhP ${corfiles[@]} ${outfile} ${files[@]} clipa:/data/data_toltec/reduced/
rsync -avhP ${corfiles[@]} ${outfile} ${files[@]} clipo:/data/data_toltec/reduced/
# mk links
for i in $(seq 0 12); do
	ampcorfile=${scratchdir}/toltec${i}_${obsid}_autodrive.txt
	if [ -f ${ampcorfile} ]; then
		ln -rsf ${ampcorfile} ${scratchdir}/toltec${i}_autodrive.txt
	fi
	adrvfile=${scratchdir}/toltec${i}_${obsid}_autodrive.a_drv
	if [ -f ${adrvfile} ]; then
		ln -rsf ${adrvfile} ${scratchdir}/toltec${i}_autodrive.a_drv
	fi
done
corfiles=(${scratchdir}/toltec[0-9]_autodrive.txt ${scratchdir}/toltec[0-9][0-9]_autodrive.txt)
files=(${scratchdir}/toltec[0-9]_autodrive.a_drv ${scratchdir}/toltec[0-9][0-9]_autodrive.a_drv)
outfile=${scratchdir}/toltec_autodrive.txt
echo "super collect result from ${files[@]}"
${pyexec} ${bin} collect ${files[@]} -fo ${outfile}
