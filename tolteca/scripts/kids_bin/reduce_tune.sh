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
    echo "Usage: reduce_tune.sh filepath"
fi
filepath=$1
filename=$(basename ${filepath})
echo ${filename}
if [[ "${filename}" =~ ^toltec([0-9][0-9]?)_.+ ]]; then
  nw=${BASH_REMATCH[1]}
else
  nw=0
fi
output_filepath=${scratchdir}/${filename%.*}_targ_freqs.dat
log_filepath=${scratchdir}/${filename%.*}_reduce.log
echo "tune reduce for ${filepath} ${output_filepath}"

# locate ref_data
ref_file=${scratchdir}/toltec${nw}_vnasweep.refdata
${pybindir}/python ${scriptdir}/reduce_tune_v20230321.py -r ${ref_file} ${filepath} # > ${log_filepath}
# ${scriptdir}/reduce.sh ${filepath} -r --output ${output_filepath} > ${log_filepath}
