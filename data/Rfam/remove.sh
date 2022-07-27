for i in `seq 4222`
do
    idx=$(printf "%05d" $i)
    if [ ! -e RF$idx.fa ]; then
        rm acc$idx.csv
    fi
done