for j in $(seq 0 13);
do
    ./reduce_simple.sh /data_lmt/toltec/tcs/toltec${j}/toltec${j}_*${1}*tune.nc -r --output dummy_output
done
