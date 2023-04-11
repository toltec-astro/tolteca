#!/bin/bash

pybindir="${HOME}/toltec_astro/extern/pyenv/versions/tolteca_v1/bin"
toltecaexec=${pybindir}/tolteca
scriptdir=$(dirname "$(readlink -f "$0")")
dataroot=/data/data_lmt
scratchdir=${dataroot}/toltec/reduced
rcdir=$HOME/toltec_astro/run/tolteca/pointing

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

apt_filename=$(${pybindir}/python3 ${scriptdir}/get_apt_filename.py ${obsnum})
echo use apt filename ${apt_filename}
echo "reduce pointing obsnum=${obsnum}"

obsnum_str=$(printf "%06d" ${obsnum})

# link files to input folder
tel_file=${dataroot}/tel/tel_toltec*_${obsnum_str}_*.nc
apt_file=${rcdir}/apt.ecsv
apt_in_file=${rcdir}/apt_GW_v6_2.ecsv
# run match apt with current obs
set -x
${pybindir}/python3 ${scriptdir}/make_matched_apt.py \
    --data_rootpath /data_lmt \
    --apt_in_file ${apt_in_file} \
    --output_dir ${rcdir}/data \
    -- ${obsnum}
apt_matched_file=${rcdir}/data/apt_${obsnum}_matched.ecsv
set +x
# tel_filename=$(basename ${tel_file})
# echo ${tel_filename} '->' ${tel_filename/tel_toltec_/tel_}
# ln -sf ${tel_file} ${rcdir}/data/${tel_filename/tel_toltec_/tel_}
ln -sf ${tel_file} ${rcdir}/data/
# ln -sf ${apt_file} ${rcdir}/data/${apt_filename}
ln -sf ${apt_matched_file} ${rcdir}/data/${apt_filename}
ln -sf ${dataroot}/toltec/tcs/toltec*/toltec*_${obsnum_str}_*.nc ${rcdir}/data/

# run tolteca reduce
$toltecaexec -d ${rcdir} -- reduce --jobkey reduced/${obsnum} --inputs.0.select "obsnum == ${obsnum} & (scannum == ${scannum})"
# $toltecaexec -d ${rcdir} -- reduce --jobkey reduced/${obsnum} --inputs.0.select "obsnum == ${obsnum} & (scannum == ${scannum}) & (interface != \"toltec6\")"
# $toltecaexec -g -d ${rcdir} -- reduce --jobkey reduced/${obsnum} --inputs.0.select "(obsnum == ${obsnum}) & (scannum == ${scannum}) & (interface != \"toltec0\") & (interface != \"toltec7\")
# & (interface != \"toltec9\") & (interface != \"toltec8\") & (interface != \"toltec2\") & (interface != \"toltec3\") " #& (interface != \"toltec1\") & (interface != \"toltec4\") & (interface != \"toltec6\")"
# run the pointing script
resultdir=${rcdir}/reduced/${obsnum}
redudir=$(${pybindir}/python3 ${scriptdir}/get_largest_redu_dir_for_obsnum.py $resultdir $obsnum)
if [[ $? != 0 ]]; then
    exit 0
fi
echo "run pointing reader in ${redudir}"
${pybindir}/python3 $scriptdir/pointing_reader_v1_3.py -c crpix -p ${redudir}/${obsnum}/raw --obsnum ${obsnum} -s -o ${redudir}/${obsnum}/raw
