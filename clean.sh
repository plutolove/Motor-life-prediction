path=/media/meng/9079-7B0D/motor3
dpath=/media/meng/9079-7B0D/clean_m3
for filename in `ls $path`
do
    dname=$(basename $filename .lvm).csv
    echo $path/$filename
    echo $dpath/$dname
    awk 'NR>22' $path/$filename > $dpath/$dname
done