sample_size=$1;

head -n 1 ./data/train.csv > ./data/train_subset.csv;
tail -n +2 ./data/train.csv | shuf -n $sample_size >> ./data/train_subset.csv;
