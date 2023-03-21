
paths=( \
    /data_lmt/toltec/ \
    /data_lmt/toltec/tcs \
    /data_lmt/toltec/ics \
    /data_lmt/toltec/reduced \
    /data_lmt/tel \
)
for p in ${paths[@]}; do
    lp=$(readlink -f $p)
    echo "******** ${p} *******"
    echo "link: ${lp}"
    df -h ${lp}
    echo ""
done
remote_paths=( \
    clipa:/data/toltec \
    clipo:/data/toltec \
   #  clipy:/data_lmt \
)

for p in ${remote_paths[@]}; do
    hn=${p%%:*}
    pp=${p##*:}
    lp=$(ssh ${hn} readlink -f $pp)
    echo "******** ${p} *******"
    echo "link: ${lp}"
    ssh ${hn} df -h ${lp}
    echo ""
done
