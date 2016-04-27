#!/bin/bash

#blades[0]="01"
#blades[1]="16"
#for dset_length in "AIDS",1503 "CPDB",684 "GDD",1178 "CAS",4337 "NCI1",4110; do
#for dset_length in "CPDB",684; do # "GDD",1178; do # "CAS",4337 "NCI1",4110; do
for dset_length in "CAS",4337 "NCI1",4110; do
#for dset_length in "CAS",4337; do
#for dset_length in "NCI1",4110; do
dataset=${dset_length%,*}
size=${dset_length#*,}
for lambda in "0.0" "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7"; do # "0.8" "0.9" "1.0"; do
for seed in "42" "43" "44" "45" "46" "47" "48" "49" "50" "51"; do

lines=`qstat | grep cmass | wc -l`

while [ "$lines" -gt 79 ]
do
	echo "$lines jobs, waiting..."
	sleep 60
	lines=`qstat | grep cmass | wc -l`
done

#i=$RANDOM
#let "i %= 2"
#blade=${blades[i]}

ram=16
#if [[ $dataset == "CAS" ]]; then
#    if [[ $2 == "1" ]]; then
#        ram=18
#    else
#        ram=34
#    fi
#elif [[ $dataset == "NCI1" ]]; then
#    if [[ $2 == "1" ]]; then
#        ram=18
#    else
#        ram=29
#    fi
#elif [[ $dataset == "AIDS" ]]; then
#    if [[ $2 == "1" ]]; then
#        ram=3
#    else
#        ram=5
#    fi
#elif [[ $dataset == "CPDB" ]]; then
#    if [[ $2 == "1" ]]; then
#        ram=2
#    else
#        ram=3
#    fi
#elif [[ $dataset == "GDD" ]]; then
#    if [[ $2 == "1" ]]; then
#        ram=1
#    else
#        ram=3
#    fi
#fi
#multikernel.seed48.L0.0
if ! [ -e "$HOME/cluster_bundle/scikit-learn-graph/cvres/$dataset/group3/mkl$2/multikernel.seed$seed.L$lambda" ]; then

echo "#!/bin/sh
### Set the job name
#PBS -N nb.$1.$seed.$dataset.l$lambda

### Declare myprogram non-rerunable
#PBS -r n

### Optionally specifiy destinations for your myprogram output
### Specify localhost and an NFS filesystem to prevent file copy errors.
#PBS -e localhost:${HOME}/tesi/logs/gramsmkl/err/${dataset}.$1.L$lambda.nested.err
#PBS -o localhost:${HOME}/tesi/logs/gramsmkl/${dataset}.$1.L$lambda.nested.out

### Set the queue to batch, the only available queue. 
#PBS -q cluster_long

### You MUST specify some number of nodes or Torque will fail to load balance.
### nodes=number of distinct host
### ppn=processes per node  :cache6mb
###PBS -l nodes=1:ppn=1:hpblade$blade
#PBS -l nodes=1:ppn=1:infiniband:nocuda

### You should tell PBS how much memory you expect your job will use.  mem=1g or mem=1024
#PBS -l mem=${ram}g

### You can override the default 1 hour real-world time limit.  -l walltime=HH:MM:SS
### Jobs on the public clusters are currently limited to 10 days walltime.
#PBS -l walltime=999:00:00

### Switch to the working directory; by default Torque launches processes from your home directory.
### Jobs should only be run from /home, /project, or /work; Torque returns results via NFS.

cd $HOME/cluster_bundle/scikit-learn-graph/

python -u $HOME/cluster_bundle/master-thesis/experiments/wl_kernels_cv_mkl_single.py $lambda cvres/${dataset}/group3/mkl$2/multikernel $seed $1 $dataset" > $HOME/tesi/jobs/${dataset}.$1.$seed.$lambda.nested.job

qsub $HOME/tesi/jobs/${dataset}.$1.$seed.$lambda.nested.job
rm $HOME/tesi/jobs/${dataset}.$1.$seed.$lambda.nested.job
fi

done
done
done
