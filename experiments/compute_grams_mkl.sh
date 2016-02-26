#!/bin/bash

#for dataset in "CAS" "NCI1" "AIDS" "CPDB" "GDD"; do
for dataset in "GDD"; do
for radius in "8" "7" "6" "5" "4" "3" "2" "1" "0"; do

#lines=`qstat | grep cmass | wc -l`
#
#while [ "$lines" -gt 59 ]
#do
#	echo "$lines jobs, waiting..."
#	sleep 120
#	lines=`qstat | grep cmass | wc -l`
#done

echo "#!/bin/sh

### Set the job name
#PBS -N mkl.$dataset.r$radius.grams

### Declare myprogram non-rerunable
#PBS -r n

### Optionally specifiy destinations for your myprogram output
### Specify localhost and an NFS filesystem to prevent file copy errors.
#PBS -e localhost:${HOME}/tesi/logs/err/${dataset}.mkl.MATRIX.r$radius.grams.err
#PBS -o localhost:${HOME}/tesi/logs/${dataset}.mkl.MATRIX.r$radius.grams.out

### Set the queue to batch, the only available queue. 
#PBS -q cluster_long

### You MUST specify some number of nodes or Torque will fail to load balance.
### nodes=number of distinct host
### ppn=processes per node  :cache6mb
#PBS -l nodes=1:ppn=1:infiniband

### You should tell PBS how much memory you expect your job will use.  mem=1g or mem=1024
#PBS -l mem=4g

### You can override the default 1 hour real-world time limit.  -l walltime=HH:MM:SS
### Jobs on the public clusters are currently limited to 10 days walltime.
#PBS -l walltime=900:00:00


### Switch to the working directory; by default Torque launches processes from your home directory.
### Jobs should only be run from /home, /project, or /work; Torque returns results via NFS.

cd $HOME/cluster_bundle/scikit-learn-graph/

python -u $HOME/cluster_bundle/scikit-learn-graph/scripts/baseline_mkl_grams.py $1 $dataset $radius grams/${dataset}/mkl/ "> $HOME/tesi/jobs/${dataset}.$radius.grams.job

qsub $HOME/tesi/jobs/${dataset}.$radius.grams.job

done
done

