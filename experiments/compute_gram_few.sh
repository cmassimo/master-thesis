#!/bin/bash

skgraph_home="$HOME/tesi/code/scikit-learn-graph/"
experiments_home="$HOME/tesi/repo/experiments/"

for dataset in "CAS" "NCI1"; do
#for dataset in "NCI1"; do
    for radius in "3" "2"; do
        for lambda in "1.4" "1.5"; do

            echo $dataset $radius $lambda
            outname=$experiments_home/$dataset.$radius.$lambda.$1

            python -m scripts/calculate_matrix_allkernels $dataset $radius $lambda $outname $2

        done
    done
done
