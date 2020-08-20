#!/bin/bash

if [[ ! $1 ]]; then
	echo -e "Compute autodrive ampcor file for all networks.\n Usage: $0 obsnum"
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

obsnum=$1

echo "autodrive all obsnum=${obsnum}"


if [[ $(hostname) == "clipa" ]]; then
    nws=$(seq 0 6)
elif [[ $(hostname) == "clipo" ]]; then
    nws=$(seq 7 12)
else
    echo "invalid host"
    exit 1
fi

# echo $nws | parallel ${scriptdir}/autodrive_ot.sh {} $@
parallel ${scriptdir}/autodrive_ot.sh {} $@ ::: $nws
# for nw in $nws; do
#    ${scriptdir}/autodrive_ot.sh ${nw} $@
# done
# collect
obsnum_str=$(printf "%06d" ${obsnum})
# mk links
for i in $nws; do
	ampcorfile=${scratchdir}/toltec${i}_${obsnum_str}_autodrive.txt
	if [ -f ${ampcorfile} ]; then
		ln -rsf ${ampcorfile} ${scratchdir}/toltec${i}_autodrive.txt
	fi
	adrvfile=${scratchdir}/toltec${i}_${obsnum_str}_autodrive.a_drv
	if [ -f ${adrvfile} ]; then
		ln -rsf ${adrvfile} ${scratchdir}/toltec${i}_autodrive.a_drv
	fi
done
corfiles=(${scratchdir}/toltec[0-9]_autodrive.txt ${scratchdir}/toltec[0-9][0-9]_autodrive.txt)
files=(${scratchdir}/toltec[0-9]_autodrive.a_drv ${scratchdir}/toltec[0-9][0-9]_autodrive.a_drv)
outfile=${scratchdir}/toltec_autodrive.txt
echo "super collect result from ${files[@]}"
${pyexec} ${bin} collect ${files[@]} -fo ${outfile}

for i in $nws; do
    echo "+++++++++++++ $(hostname) +++ toltec$i ++++++++++++++"
    echo cp ${scratchdir}/toltec${i}_autodrive.txt /home/toltec/roach/etc/toltec$i/default_targ_amps.dat
    cp ${scratchdir}/toltec${i}_autodrive.txt /home/toltec/roach/etc/toltec$i/default_targ_amps.dat
done
