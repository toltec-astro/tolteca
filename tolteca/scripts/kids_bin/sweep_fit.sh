#!/bin/bash
## these are common post processing for sweep files VNA or TUNE

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
    echo "Usage: reduce_sweep.sh filepath"
fi
filepath=$(readlink -f $1)
filename=$(basename ${filepath})
reportfile=$(basename ${filepath})
reportfile=${reportfile%.*}.txt
reportfile=${scratchdir}/${reportfile}

# echo ${filename}
if [[ "${filename}" =~ ^toltec([0-9][0-9]?)_.+_(tune|vnasweep|targsweeep).nc ]]; then
  nw=${BASH_REMATCH[1]}
  kind_str=${BASH_REMATCH[2]}
else
  nw=0
  kind_str='tune'
fi
echo "sweep fit for nw=${nw} kind_str=${kind_str} ${filepath} ${reportfilepath}"
# locate ref_data
ref_file=${scratchdir}/toltec${nw}_vnasweep.refdata
shift
${pybindir}/python ${scriptdir}/reduce_tune_v20230321.py -r ${ref_file} ${filepath} --save_fit_only "$@"
