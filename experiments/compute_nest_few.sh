#!/bin/bash

skgraph_home="$HOME/tesi/code/scikit-learn-graph/"
experiments_home="$HOME/tesi/repo/experiments/"

for dataset in "CAS" "NCI1"; do
for radius in "3"; do
for lambda in "1.1" "1.2" "1.3" "1.4" "1.5"; do
for C in "0.1" "1.0" "10.0" "100.0"; do

echo $dataset $radius $lambda
outname=$experiments_home/$dataset.$radius.$lambda.oddstpc_tanh_v0.mtx.svmlight

python -m cross_validation_from_matrix $outname $C cv_res_$dataset.r$radius.l$lambda

done
done
done
done
