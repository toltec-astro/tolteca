
if [[ ! $2 ]]; then
    echo "invalid args"
    exit 1
fi
obsnum=$1
obsnum_str=$(printf "%06d" ${obsnum})
scriptdir=$(dirname "$(readlink -f "$0")")
scratchdir=/home/toltec/tlaloc/bin/reduced_zma
pyexec="${HOME}/toltec_astro/venvs/toltec/bin/python3"


if [[ $(hostname) == "clipa" ]]; then
    nws=$(seq 0 6)
elif [[ $(hostname) == "clipo" ]]; then
    nws=$(seq 7 12)
elif [[ $(hostname) == "taco" ]]; then
    if [[ $3 ]]; then
        ${pyexec} ${scriptdir}/filter_apt.py $3
        # copy apt to clipa and clipo for remove flagged detectors
        scp $3 clipa:${scriptdir}
        scp $3 clipo:${scriptdir}
    fi
    hosts="clipa clipo"
    this=$(basename $0)
    parallel -v ssh -t {} bash ${scriptdir}/${this} $1 $2 ${scriptdir}/$(basename $3) ::: $hosts
    echo 'hosts all done'
    exit 0
else
    echo "invalid host"
    exit 1
fi

# do the reduction

# for i in /data_lmt/toltec/*/toltec*/toltec*_${obsnum_str}_*tune.nc ; do SCRATCHDIR=${scratchdir} ${scriptdir}/reduce.sh $i -r --output dummy_output ; done
if [[ $3 ]]; then
    echo "filter with apt"
    args=(--filter_by_apt $3)
else
    args=()
fi
${pyexec} ${scriptdir}/set_tones.py ${scratchdir}/toltec*_${obsnum_str}_*tune.txt --dp $2 "${args[@]}"
if [ $? -eq 0 ]; then
    echo "tones successfully created"
else
    echo "failed to set tones"
    exit 1
fi
for nw in $nws; do
    file=/data_lmt/toltec/*/toltec${nw}/toltec${nw}_${obsnum_str}_*tune.nc
    echo ${file}
    if [ -f ${file} ]; then
        reportfile=${scratchdir}/toltec${nw}_${obsnum_str}_*tune_shifted.txt
        reportfile=$(echo ${reportfile})
        ampcorfile=${scratchdir}/toltec${nw}_${obsnum_str}_*ampcor_filtered.txt
        ampcorfile=$(echo ${ampcorfile})
        outfile=${reportfile%_tune_shifted.txt}_targ_freqs.dat
        outfile_commit=${scratchdir}/toltec${nw}_targ_freqs.dat
        ampcorfile_commit=${scratchdir}/toltec${nw}_ampcor.dat
        ${pyexec} ${scriptdir}/fix_lo.py ${file} ${reportfile} ${outfile}
        ln -sf ${outfile} ${outfile_commit}
        ln -sf ${ampcorfile} ${ampcorfile_commit}
    fi
done
