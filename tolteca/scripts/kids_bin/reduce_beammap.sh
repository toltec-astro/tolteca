#/bin/bash
echo "reduce beammap recipe"
pybindir="${HOME}/toltec_astro/extern/pyenv/versions/tolteca_v1/bin"
toltecaexec=${pybindir}/tolteca
scriptdir=$(dirname "$(readlink -f "$0")")
dataroot=/data/data_lmt
scratchdir=${dataroot}/toltec/reduced
rcdir=$HOME/toltec_astro/run/tolteca/beammap
prcdir=$HOME/toltec_astro/run/tolteca/pointing
srcdir=$HOME/toltec_astro/run/tolteca/science

if [[ -e ${SCRATCHDIR} ]]; then
    scratchdir=${SCRATCHDIR}
fi
echo "use scratch ${scratchdir}"
echo "additional output to: ${scratchdir}"

if [[ ! $1 ]]; then
    obsnum=$(${pybindir}/python3 ${scriptdir}/get_latest_obsnum.py)
    echo found latest obsnum ${obsnum}
else
    obsnum=$1
fi

scannum=$(${pybindir}/python3 ${scriptdir}/get_latest_scannum.py ${obsnum})
echo found latest scannum ${scannum}

echo "reduce beammap obsnum=${obsnum}"

obsnum_str=$(printf "%06d" ${obsnum})

# link files to input folder
tel_file=${dataroot}/tel/tel_toltec*_${obsnum_str}_*.nc
# tel_filename=$(basename ${tel_file})
# echo ${tel_filename} '->' ${tel_filename/tel_toltec_/tel_}
# ln -sf ${tel_file} ${rcdir}/data/${tel_filename/tel_toltec_/tel_}
ln -sf $tel_file ${rcdir}/data/
ln -sf ${dataroot}/toltec/tcs/toltec*/toltec[0-9]*_${obsnum_str}_*.nc ${rcdir}/data/
# ${pybindir}/python3 ${scriptdir}/make_beammap_input_apt.py obsnum

# run tolteca reduce
$toltecaexec -g -d ${rcdir} -- reduce --jobkey reduced/${obsnum} --inputs.0.select "(obsnum == ${obsnum}) & (scannum == ${scannum})"

# run the pointing script
resultdir=${rcdir}/reduced/${obsnum}
redudir=$(${pybindir}/python3 ${scriptdir}/get_largest_redu_dir_for_obsnum.py $resultdir $obsnum)
if [[ $? != 0 ]]; then
    exit 0
fi
# prepare the apt table and put it in the apt folder
aptdir=${HOME}/toltec_astro/run/apt
apt_output_file=${redudir}/${obsnum}/raw/apt_*.ecsv
${pybindir}/python3 ${scriptdir}/cleanup_beammap_apt.py --obsnum ${obsnum} ${apt_output_file} --output_dir ${aptdir}
echo "run beammap reader in ${redudir}"
ln -sf ${aptdir}/apt_${obsnum}_cleaned.ecsv ${rcdir}/apt.ecsv
ln -sf ${aptdir}/apt_${obsnum}_cleaned.ecsv ${prcdir}/apt.ecsv
ln -sf ${aptdir}/apt_${obsnum}_cleaned.ecsv ${srcdir}/apt.ecsv
${pybindir}/python3 $scriptdir/beammap_reader_v1.py -p ${redudir}/${obsnum}/raw --obsnum ${obsnum} -s -o ${redudir}/${obsnum}/raw
