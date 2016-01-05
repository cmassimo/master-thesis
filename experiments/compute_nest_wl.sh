#!/bin/bash

#for dataset in "CAS" "NCI1" "AIDS" "CPDB" "GDD"; do
for dataset in "NCI1"; do
#for radius in "1" "2" "3" "4"; do
for radius in "3" "4"; do
for iteration in "0" "1" "2" "3" "4" "5" "6" "7" "8"; do
for lambda in "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1.0" "1.1" "1.2" "1.3" "1.4" "1.5" "1.6" "1.7" "1.8" "1.9" "2.0"; do

lines=`qstat | grep cmass | wc -l`

while [ "$lines" -gt 45 ]
do
	echo "$lines jobs, waiting..."
	sleep 120 
	lines=`qstat | grep cmass | wc -l`
done

for C in "0.01" "0.1" "1.0" "10.0" "100.0"; do

echo "#!/bin/sh
### Set the job name
#PBS -N thexp.r$radius.i$iteration.l$lambda.$dataset.$C.nested

### Declare myprogram non-rerunable
#PBS -r n

### Optionally specifiy destinations for your myprogram output
### Specify localhost and an NFS filesystem to prevent file copy errors.
###PBS -e localhost:${HOME}/tesi/logs/err/${dataset}/$1/${dataset}.$1.MATRIX.r$radius.l$lambda.nested.err
###PBS -o localhost:${HOME}/tesi/logs/${dataset}.$1.MATRIX.r$radius.l$lambda.nested.out

### Set the queue to batch, the only available queue. 
#PBS -q cluster_long

### You MUST specify some number of nodes or Torque will fail to load balance.
### nodes=number of distinct host
### ppn=processes per node  :cache6mb
#PBS -l nodes=1:ppn=1:infiniband

### You should tell PBS how much memory you expect your job will use.  mem=1g or mem=1024
#PBS -l mem=8g

### You can override the default 1 hour real-world time limit.  -l walltime=HH:MM:SS
### Jobs on the public clusters are currently limited to 10 days walltime.
#PBS -l walltime=30:00:00


### Switch to the working directory; by default Torque launches processes from your home directory.
### Jobs should only be run from /home, /project, or /work; Torque returns results via NFS.

cd $HOME/cluster_bundle/scikit-learn-graph/

python $HOME/cluster_bundle/master-thesis/experiments/cross_validation_from_matrix.py grams/${dataset}/$1/k$1.r$radius.i$iteration.l$lambda.mtx.svmlight $C cvres/${dataset}/$1/k$1.r$radius.i$iteration.l$lambda"> $HOME/tesi/jobs/${dataset}.$radius.$iteration.$lambda.$1.$C.nested.job

qsub $HOME/tesi/jobs/${dataset}.$radius.$iteration.$lambda.$1.$C.nested.job

done
done
done
done
done

