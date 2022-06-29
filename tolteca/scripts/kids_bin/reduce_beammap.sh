#!/bin/bash

pybindir="${HOME}/toltec_astro/extern/pyenv/versions/tolteca_v1/bin"
toltecaexec=${pybindir}/tolteca
scriptdir=$(dirname "$(readlink -f "$0")")
dataroot=/data/data_lmt
scratchdir=${dataroot}/toltec/reduced
rcdir=$HOME/toltec_astro/run/tolteca/beammap

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
$toltecaexec -g -d ${rcdir} -- reduce --jobkey reduced/${obsnum} --inputs.0.select "(obsnum == ${obsnum}) & (scannum == scannum.max())"
