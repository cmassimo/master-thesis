#!/bin/bash

for dset_length in "AIDS",1503 "CPDB",684 "CAS",4337 "NCI1",4110 "GDD",1178; do
dataset=${dset_length%,*}
size=${dset_length#*,}

echo "#!/bin/sh
### Set the job name
#PBS -N times.$dataset.svm

### Declare myprogram non-rerunable
#PBS -r n

### Optionally specifiy destinations for your myprogram output
### Specify localhost and an NFS filesystem to prevent file copy errors.
###PBS -e localhost:${HOME}/tesi/logs/err/${dataset}.times.err
###PBS -o localhost:${HOME}/tesi/logs/${dataset}.times.out

### Set the queue to batch, the only available queue. 
#PBS -q cluster_long

### You MUST specify some number of nodes or Torque will fail to load balance.
### nodes=number of distinct host
### ppn=processes per node  :cache6mb
#PBS -l nodes=1:ppn=1:hpblade08

### You should tell PBS how much memory you expect your job will use.  mem=1g or mem=1024
#PBS -l mem=4g

### You can override the default 1 hour real-world time limit.  -l walltime=HH:MM:SS
### Jobs on the public clusters are currently limited to 10 days walltime.
#PBS -l walltime=100:00:00

### Switch to the working directory; by default Torque launches processes from your home directory.
### Jobs should only be run from /home, /project, or /work; Torque returns results via NFS.

cd $HOME/cluster_bundle/scikit-learn-graph/

python $HOME/cluster_bundle/master-thesis/experiments/cross_validation_from_matrix.py grams/${dataset}/$1/$2 $size times/$dataset/$3"> $HOME/tesi/jobs/${dataset}.$1.times.job

qsub $HOME/tesi/jobs/${dataset}.$1.times.job

done

