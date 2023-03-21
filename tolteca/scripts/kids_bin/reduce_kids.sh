#!/bin/bash

pybindir="${HOME}/toltec_astro/extern/pyenv/versions/tolteca_v1/bin"
toltecaexec=${pybindir}/tolteca
scriptdir=$(dirname "$(readlink -f "$0")")
dataroot=/data_lmt
scratchdir=${dataroot}/toltec/reduced

if [[ -e ${SCRATCHDIR} ]]; then
    scratchdir=${SCRATCHDIR}
fi
echo "use scratch ${scratchdir}"
echo "additional output to: ${scratchdir}"

if [[ ! $1 ]]; then
    echo "Usage: reduce_kids.sh filepath"
fi
filepath=$1
filename=$(basename ${filepath})
output_filepath=${scratchdir}/${filename%.*}_targ_freqs.dat
log_filepath=${scratchdir}/${filename%.*}_reduce.log
echo "kids reduce for ${filepath} ${output_filepath}"

${scriptdir}/reduce.sh ${filepath} -r --output ${output_filepath} > ${log_filepath}
