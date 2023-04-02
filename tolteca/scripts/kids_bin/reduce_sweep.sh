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
echo "sweep reduce for nw=${nw} kind_str=${kind_str} ${filepath} ${reportfilepath}"
# output_filepath=${scratchdir}/${filename%.*}_targ_freqs.dat
# log_filepath=${scratchdir}/${filename%.*}_reduce.log
# echo "vna reduce for ${filepath} ${output_filepath}"

if [[ ${kind_str} == "vnasweep" ]]; then
    ref_file=${reportfile%.*}.refdata
    echo "generating refdata using reportfile ${reportfile} to ${ref_file}"
    ${pybindir}/python ${scriptdir}/make_ref_data.py ${filepath} ${reportfile}
    ref_link=${scratchdir}/toltec${nw}_vnasweep.refdata
    if ! [ -f ${ref_file} ]; then
        echo "failed to create ref_file, abort"
        exit 1
    fi
    ln -sf ${ref_file} ${ref_link}
    echo "link report file ${ref_link}"
fi
# locate ref_data
ref_file=${scratchdir}/toltec${nw}_vnasweep.refdata
set -x
${pybindir}/python ${scriptdir}/reduce_tune_v20230321.py -r ${ref_file} ${filepath} --no_fit --tune_mode # > ${log_filepath}
set +x
# ${scriptdir}/reduce.sh ${filepath} -r --output ${output_filepath} > ${log_filepath}
