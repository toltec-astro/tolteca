#!/bin/bash

if [[ ! $1 ]]; then
	echo -e "Compute autodrive ampcor file for all networks, using the last <num> of autodrive data.\n Usage: $0 <num>"
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

obsid1=$(${pyexec} ${scriptdir}/get_latest_obsnum.py)
# runid=obsid1
obsid0="$(($obsid1 - $1))"

echo "autodrive all obsid=${obsid0}:${obsid1}"
echo "seq 0 12 | parallel ${scriptdir}/autodrive.sh {} $obsid0 $obsid1"
# seq 0 12 | parallel ${scriptdir}/autodrive.sh {} $obsid0 $obsid1
exit 1

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
rsync -avhP ${outfile} ${corfiles[@]} ${files[@]} clipa:/data/data_toltec/reduced/
rsync -avhP ${outfile} ${corfiles[@]} ${files[@]} clipo:/data/data_toltec/reduced/

for i in 0 1 2 3 4 5 6; do
	echo "+++++++++++++ clipa +++ toltec$i ++++++++++++++"
	scp ${scratchdir}/toltec${i}_autodrive.txt clipa:/home/toltec/roach/etc/toltec$i/default_targ_amps.dat
done

for i in 7 8 9 10 11 12; do
	echo "+++++++++++++ clipo +++ toltec$i ++++++++++++++"
	scp ${scratchdir}/toltec${i}_autodrive.txt clipo:/home/toltec/roach/etc/toltec$i/default_targ_amps.dat
done
