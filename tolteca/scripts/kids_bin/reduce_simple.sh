#!/bin/bash
# source /home/toltec/toltec_astro/dotbashrc

file=${@: 1:1}
args_all=${@: 2:$#-1}
link=$(readlink ${file})
if [[ ! ${link} ]]; then
    link=${file}
fi

# filter out --output arg
# filter out -r/-p
args=()
runmode="reduce"
shift # start with args after ${file} 
while [[ "$#" > 0 ]] ; do
    if [[ ${1} == "--output" ]]; then
        outfile=${2}
    shift
    # args+=( ${@: 3:$#-2} )
    # break
    elif [[ ${1} == "-r" ]]; then
        runmode="reduce"
    elif [[ ${1} == "-p" ]]; then
        runmode="plot"
    elif [[ ${1} == "-f" ]]; then
        runmode="fg"
    else
        args+=( ${1} )
    fi
    shift
done
args=$(echo ${args[@]})
echo "file: ${file}"
echo "link: ${link}"
echo "outfile: ${outfile}"
echo "args: ${args}"


if [[ ${link} == *"vnasweep"* ]]; then
    type="vna"
elif [[ ${link} == *"targsweep"* ]]; then
    type="targ"
elif [[ ${link} == *"tune"* ]]; then
    type="targ"
elif [[ ${link} == *"timestream"* ]]; then
    type="timestream"
else
    type="timestream"
    echo "unrecognized type ${file}, exit."
    # exit 1
fi

scriptdir=$(dirname "$(readlink -f "$0")")
scratchdir=/data/data_toltec/reduced
if [[ -e ${SCRATCHDIR} ]]; then
    scratchdir=${SCRATCHDIR}
fi
echo "use scratch ${scratchdir}"
echo "process ${type}: ${link}"
echo "additional output to: ${scratchdir}"

kidscppdir="${HOME}/toltec_astro_v1/kidscpp"
kidspydir="${HOME}/zma_deprecated/kids_master/scripts"
pyexec="${HOME}/toltec_astro/venvs/toltec/bin/python3"
# finder_thresh=10  # TODO need a better way to handle this
# fitter_Qr=13000  # TODO need a better way to handle this

filename=$(basename ${link})
echo ${filename}
if [[ "${filename}" =~ ^toltec([0-9][0-9]?)_.+ ]]; then
  nw=${BASH_REMATCH[1]}
else
  nw=0
fi
echo "nw: ${nw}"
if (( ${nw} >= 7 )); then
    # 1.4 and 2.0mm
    finder_args=( \
        --fitter_weight_window_Qr 5000 \
        --finder_use_savgol_deriv \
        --finder_smooth_size 15 \
        --finder_threshold 3 --finder_stats_clip_sigma 2 --fitter_lim_gain_min 0)
    fitter_args=( \
        --fitter_weight_window_Qr 5000 \
        --finder_use_savgol_deriv \
        --finder_smooth_size 15 \
        --fitter_auto_global_shift)
elif (( ${nw} <= 6 )); then
    finder_args=( \
        --fitter_weight_window_Qr 7500 \
        --finder_use_savgol_deriv \
        --finder_smooth_size 15 \
        --finder_threshold 3 --finder_stats_clip_sigma 2 --fitter_lim_gain_min 0)
    fitter_args=( \
        --fitter_weight_window_Qr 7500 \
        --finder_use_savgol_deriv \
        --finder_smooth_size 15 \
        --fitter_auto_global_shift)
fi
echo finder_args: ${finder_args[@]}

if [[ ${type} == "vna" ]]; then
    echo "do ${type} ${runmode}"
    reportfile=$(basename ${link})
    reportfile=${reportfile%.*}.txt
    echo "reportfile: ${reportfile}"
    reportfile="${scratchdir}/${reportfile}"
    if [[ ${runmode} == "reduce" ]]; then
        ${kidscppdir}/build/bin/kids \
            ${finder_args[@]} \
            --output_d21 ${scratchdir}/'{stem}_d21.nc' \
            --output_processed ${scratchdir}/'{stem}_processed.nc' \
            --output "${reportfile}" \
                "${file}" ${args}
        if [[ $outfile ]]; then
            ${pyexec} ${scriptdir}/fix_lo.py ${file} "${reportfile}" "${outfile}"
        fi
    elif [[ ${runmode} == "plot" ]]; then
        ${pyexec} ${kidspydir}/kidsdetect.py ${file} --plot_d21 ${scratchdir}/'{stem}_d21.nc' ${args} &
    elif [[ ${runmode} == "fg" ]]; then
        ${pyexec} ${kidspydir}/kidsdetect.py ${file} --plot_d21 ${scratchdir}/'{stem}_d21.nc' ${args} 
    fi
elif [[ ${type} == "targ" ]]; then
    echo "do ${type} ${runmode}"
    reportfile=$(basename ${link})
    reportfile=${reportfile%.*}.txt
    echo "reportfile: ${reportfile}"
    reportfile="${scratchdir}/${reportfile}"
    if [[ ${runmode} == "reduce" ]]; then
        ${kidscppdir}/build/bin/kids \
            ${fitter_args[@]} \
            --output_processed ${scratchdir}/'{stem}_processed.nc' \
            --output "${reportfile}"  "${file}" ${args}
    if [[ $outfile ]]; then
        ${pyexec} ${scriptdir}/fix_lo.py ${file} "${reportfile}" "${outfile}"
    fi
    elif [[ ${runmode} == "plot" ]]; then
        ${pyexec} ${kidspydir}/kidsvis.py ${file} --fitreport "${reportfile}" --use_derotate ${args} --grid 8 8 &
    elif [[ ${runmode} == "fg" ]]; then
        ${pyexec} ${kidspydir}/kidsvis.py ${file} --fitreport "${reportfile}" --use_derotate ${args} --grid 8 8 
    fi
elif [[ ${type} == "timestream" ]]; then
    echo "do ${type} ${runmode}"
    if [[ ${runmode} == "reduce" ]]; then
        o=$(basename ${file})
        o=${scratchdir}/${o%.*}_processed.nc
        if [ -f "${o}" ] ; then
            echo rm ${o}
            rm "${o}"
        fi
        ${kidscppdir}/build/bin/kids \
            --solver_fitreportdir ${scratchdir} \
            --output "${scratchdir}/{stem}_processed.nc" --solver_chunk_size 500000 --solver_extra_output ${file} ${args}
    elif [[ ${runmode} == "plot" ]]; then
        ${pyexec} ${kidspydir}/timestream.py --fitreportdir ${scratchdir} ${file} ${args} &
    elif [[ ${runmode} == "fg" ]]; then
        ${pyexec} ${kidspydir}/timestream.py --fitreportdir ${scratchdir} ${file} ${args} --grid 1 1
    fi
fi