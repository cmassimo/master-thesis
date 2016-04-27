#!/bin/bash

#for dataset in "AIDS" "CPDB" "GDD" "CAS" "NCI1"; do
#for dataset in "AIDS" "CPDB" "GDD"; do
#for dataset in "CAS" "NCI1"; do

echo "#!/bin/sh

### Set the job name
#PBS -N modsel.sign

### Declare myprogram non-rerunable
#PBS -r n

### Optionally specifiy destinations for your myprogram output
### Specify localhost and an NFS filesystem to prevent file copy errors.
#PBS -e localhost:${HOME}/tesi/logs/err/modsel.err
#PBS -o localhost:${HOME}/tesi/logs/results/modsel/$2

### Set the queue to batch, the only available queue. 
#PBS -q cluster_long

### You MUST specify some number of nodes or Torque will fail to load balance.
### nodes=number of distinct host
### ppn=processes per node  :cache6mb
#PBS -l nodes=1:ppn=1:infiniband:nocuda

### You should tell PBS how much memory you expect your job will use.  mem=1g or mem=1024
#PBS -l mem=1g

### You can override the default 1 hour real-world time limit.  -l walltime=HH:MM:SS
### Jobs on the public clusters are currently limited to 10 days walltime.
#PBS -l walltime=100:00:00

### Switch to the working directory; by default Torque launches processes from your home directory.
### Jobs should only be run from /home, /project, or /work; Torque returns results via NFS.

cd $HOME/cluster_bundle/scikit-learn-graph/

#python -u $HOME/cluster_bundle/master-thesis/experiments/new_performance_significance.py $HOME/tesi/cvres/$dataset/group2/$1 $HOME/tesi/logs/results/group2/${dataset}$2
python -u $HOME/cluster_bundle/master-thesis/experiments/svm_model_selection_from_nested.py $1"> $HOME/tesi/jobs/performance_sign.job

qsub $HOME/tesi/jobs/performance_sign.job
rm $HOME/tesi/jobs/performance_sign.job

#done
