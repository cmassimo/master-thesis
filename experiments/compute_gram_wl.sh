#!/bin/bash

#for dataset in "CAS" "NCI1" "AIDS" "CPDB" "GDD"; do
for dataset in "NCI1"; do
#for radius in "1" "2" "3" "4"; do
for radius in "1" "2"; do
for iteration in "0" "1" "2" "3" "4" "5" "6" "7" "8"; do

lines=`qstat | grep cmass | wc -l`

while [ "$lines" -gt 30 ]
do
	echo "Too many jobs, waiting..."
	sleep 30
	lines=`qstat | grep cmass | wc -l`
done

for lambda in "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1.0" "1.1" "1.2" "1.3" "1.4" "1.5" "1.6" "1.7" "1.8" "1.9" "2.0"; do

echo "#!/bin/sh

### Set the job name
#PBS -N thexp.r$radius.i$iteration.l$lambda.$dataset.gram

### Declare myprogram non-rerunable
#PBS -r n

### Optionally specifiy destinations for your myprogram output
### Specify localhost and an NFS filesystem to prevent file copy errors.
#PBS -e localhost:${HOME}/tesi/logs/err/${dataset}/$1/${dataset}.$1.r$radius.i$iteration.l$lambda.err
###PBS -o localhost:${HOME}/tesi/logs/${dataset}.$1.MATRIX.r$radius.i$iteration.l$lambda.out

### Set the queue to batch, the only available queue. 
#PBS -q cluster_short

### You MUST specify some number of nodes or Torque will fail to load balance.
### nodes=number of distinct host
### ppn=processes per node  :cache6mb
#PBS -l nodes=1:ppn=1:infiniband

### You should tell PBS how much memory you expect your job will use.  mem=1g or mem=1024
#PBS -l mem=24g

### You can override the default 1 hour real-world time limit.  -l walltime=HH:MM:SS
### Jobs on the public clusters are currently limited to 10 days walltime.
#PBS -l walltime=02:59:00


### Switch to the working directory; by default Torque launches processes from your home directory.
### Jobs should only be run from /home, /project, or /work; Torque returns results via NFS.

cd $HOME/cluster_bundle/scikit-learn-graph/

python -m scripts/calculate_matrix_allkernels ${dataset} $radius $lambda grams/$dataset/$1/k$1.r$radius.i$iteration.l$lambda.mtx $1 1 1 0 0 $iteration"> $HOME/tesi/jobs/${dataset}.$radius.$iteration.$lambda.$1.gram.job

qsub $HOME/tesi/jobs/${dataset}.$radius.$iteration.$lambda.$1.gram.job

done
done
done
done
