#!/bin/bash

blades[0]="07"
blades[1]="08"
#rams[0]=60
#rams[1]=92
for dset_length in "AIDS",1503 "CPDB",684 "GDD",1178; do # "CAS",4337 "NCI1",4110; do
#for dset_length in "CPDB",684 "GDD",1178 "CAS",4337 "NCI1",4110; do
dataset=${dset_length%,*}
size=${dset_length#*,}
#kODDSTC.r1.l1.3.mtx.svmlight
for n in 32; do #2 4 8; do # 16 32 64 110; do
#for n in 1 2 4 8 16; do
#for n in 32 64 110; do

lines=`qstat | grep inc | wc -l`

while [ "$lines" -gt 30 ]
do
	echo "$lines jobs, waiting..."
	sleep 60
	lines=`qstat | grep inc | wc -l`
done

i=$RANDOM
let "i %= 2"
blade=${blades[i]}
#ram=${rams[i]}

ram=8
#if (( n > 16 )); then
#    ram=36
#else
#    ram=10
#fi

#if [[ $dataset == "CAS" ]]; then
#    if [[ $2 == "1" ]]; then
#        ram=32
#    else
#        ram=56
#    fi
#elif [[ $dataset == "NCI1" ]]; then
#    if [[ $2 == "1" ]]; then
#        ram=24
#    else
#        ram=48
#    fi
#elif [[ $dataset == "AIDS" ]]; then
#    if [[ $2 == "1" ]]; then
#        ram=5
#    else
#        ram=10
#    fi
#elif [[ $dataset == "CPDB" ]]; then
#    if [[ $2 == "1" ]]; then
#        ram=2
#    else
#        ram=5
#    fi
#elif [[ $dataset == "GDD" ]]; then
#    if [[ $2 == "1" ]]; then
#        ram=1
#    else
#        ram=3
#    fi
#fi

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

echo "#!/bin/sh
### Set the job name
#PBS -N times.$dataset.inc.test1

### Declare myprogram non-rerunable
#PBS -r n

### Optionally specifiy destinations for your myprogram output
### Specify localhost and an NFS filesystem to prevent file copy errors.
#PBS -e localhost:${HOME}/tesi/logs/err/${dataset}.mkl.times.err
###PBS -o localhost:${HOME}/tesi/logs/${dataset}.mkl.times.out

### Set the queue to batch, the only available queue. 
#PBS -q cluster_long

### You MUST specify some number of nodes or Torque will fail to load balance.
### nodes=number of distinct host
### ppn=processes per node  :cache6mb
#PBS -l nodes=1:ppn=1:hpblade$blade

### You should tell PBS how much memory you expect your job will use.  mem=1g or mem=1024
#PBS -l mem=${ram}g

### You can override the default 1 hour real-world time limit.  -l walltime=HH:MM:SS
### Jobs on the public clusters are currently limited to 10 days walltime.
#PBS -l walltime=999:00:00

### Switch to the working directory; by default Torque launches processes from your home directory.
### Jobs should only be run from /home, /project, or /work; Torque returns results via NFS.

cd $HOME/cluster_bundle/scikit-learn-graph/

python -u $HOME/cluster_bundle/master-thesis/experiments/times/easymkl_incremental.py times/$dataset/$1 $size $n grams/$dataset/mkl_all2 "> $HOME/tesi/jobs/${dataset}.times.test.mkl.job
#python -u $HOME/cluster_bundle/master-thesis/experiments/times/easymkl_compute_times.py times/$dataset/$1 $dataset

qsub $HOME/tesi/jobs/${dataset}.times.test.mkl.job

done
done

