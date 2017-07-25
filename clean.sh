path=/home/meng/下载/SteadyState9
dpath=/media/meng/9079-7B0D/clean_data/clean_m7
for filename in `ls $path`
do
    dname=$(basename $filename .lvm).csv
    echo $path/$filename
    echo $dpath/$dname
    awk 'NR>22' $path/$filename > $dpath/$dname
done