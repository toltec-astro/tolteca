#!/bin/bash
source /home/toltec/toltec_astro/dotbashrc

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

kidscppdir="${HOME}/toltec_astro/kidscpp"
kidspydir="${HOME}/zma_deprecated/kids_master/scripts"
pyexec="${HOME}/zma_deprecated/venvs/kids_master/bin/python3"
finder_thresh=10  # TODO need a better way to handle this
fitter_Qr=13000  # TODO need a better way to handle this
if [[ ${type} == "vna" ]]; then
    echo "do ${type} ${runmode}"
    reportfile=$(basename ${link})
    reportfile=${reportfile%.*}.txt
    echo "reportfile: ${reportfile}"
    reportfile="${scratchdir}/${reportfile}"
    if [[ ${runmode} == "reduce" ]]; then
        ${kidscppdir}/build/bin/kids --finder_threshold ${finder_thresh} \
            --fitter_weight_window_Qr ${fitter_Qr} \
		    --output_d21 ${scratchdir}/'{stem}_d21.nc' \
		    --output_processed ${scratchdir}/'{stem}_processed.nc' \
		    --output "${reportfile}" \
	       	    "${file}" ${args}
	if [[ $outfile ]]; then
		${pyexec} ${scriptdir}/fix_lo.py ${file} "${reportfile}" "${outfile}"
	fi
	#    ${pyexec} ${kidspydir}/kidsdetect.py --smooth 5 --resample 1 --exclude_edge 40 \
	#	    --noplot --output ${outfile} \
	#	    --output_d21 ${scratchdir}/'{stem}_d21.nc' \
	#	    ${file} ${args}
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
    # debug="_debug"
    echo "exec: ${kidscppdir}/build/bin/kids --output "${reportfile}"  "${file}" ${args}"
    if [[ ${runmode} == "reduce" ]]; then
        echo ${kidscppdir}/build${debug}/bin/kids \
            --fitter_weight_window_Qr ${fitter_Qr} \
		    --output_processed ${scratchdir}/'{stem}_processed.nc' \
            --output "${reportfile}"  "${file}" ${args}
        ${kidscppdir}/build${debug}/bin/kids \
		    --output_processed ${scratchdir}/'{stem}_processed.nc' \
            --output "${reportfile}"  "${file}" ${args}
       	# >> ${file}.reduce.log 2>&1
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
        # ${kidscppdir}/build/bin/kids \
	#		--solver_fitreportdir ${scratchdir} \
	#	--output "${scratchdir}/{stem}_processed.nc" ${file} ${args}
        # echo ${pyexec} ${kidspydir}/timestream.py ${file} --fitreportdir ${scratchdir} --output "${scratchdir}/{stem}_processed.nc" ${args} --noplot
        # ${pyexec} ${kidspydir}/timestream.py ${file} --fitreportdir ${scratchdir} --output "${scratchdir}/{stem}_processed.nc" ${args} --noplot
        echo ${kidscppdir}/build/bin/kids \
		--solver_fitreportdir ${scratchdir} \
		--output "${scratchdir}/{stem}_processed.nc" --solver_chunk_size 500000 --solver_extra_output ${file} ${args}
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
