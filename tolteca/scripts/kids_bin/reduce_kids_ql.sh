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
    echo "Usage: reduce_kids_ql.sh filepaths"
fi
filepaths="$@"
output_dir=${scratchdir}
echo "kids ql reduce for ${filepaths} ${output_dir}"

${pybindir}/python3 ${scriptdir}/reduce_kids_ql.py ${filepaths} --output_dir ${output_dir}
