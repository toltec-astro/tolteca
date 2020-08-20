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

obsnum=$1

echo "autodrive all obsnum=${obsnum}"


# seq 0 12 | parallel ${scriptdir}/autodrive.sh {} $@

seq 0 12 | parallel ${scriptdir}/autodrive_ot.sh {} $@
# for nw in 2 5 6; do
#    ${scriptdir}/autodrive.sh ${nw} $@
# done
# collect
obsnum_str=$(printf "%06d" ${obsid0})
# mk links
for i in $(seq 0 6); do
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

if [[ $(hostname) ~= "clipa" ]]; then
  for i in 0 1 2 3 4 5 6; do
  	echo "+++++++++++++ clipa +++ toltec$i ++++++++++++++"
  	echo cp ${scratchdir}/toltec${i}_autodrive.txt /home/toltec/roach/etc/toltec$i/default_targ_amps.dat
  	cp ${scratchdir}/toltec${i}_autodrive.txt /home/toltec/roach/etc/toltec$i/default_targ_amps.dat
  done
elif [[ $(hostname) ~= "clipa" ]]; then
  for i in 7 8 9 10 11 12; do
  	echo "+++++++++++++ clipo +++ toltec$i ++++++++++++++"
  	scp ${scratchdir}/toltec${i}_autodrive.txt clipo:/home/toltec/roach/etc/toltec$i/default_targ_amps.dat
  done
else
    echo "invalid host"
fi
