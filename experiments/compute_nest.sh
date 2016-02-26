#!/bin/bash

for dset_length in "AIDS",1503; do # "CPDB",684 "CAS",4337 "NCI1",4110; do #"GDD",1178; do
dataset=${dset_length%,*}
size=${dset_length#*,}
for radius in "8" "7" "6" "5" "4" "3" "2" "1"; do
#for lambda in "0.1" "0.5" "0.8" "0.9" "1.0" "1.1" "1.2" "1.3" "1.4" "1.5" "1.8"; do
for lambda in "0.5" "0.6" "0.7" "0.8" "0.9" "1.0" "1.1" "1.2" "1.3" "1.4" "1.5" "1.6"; do
for C in "0.001" "0.01" "0.1" "1.0" "10.0" "100.0" "1000.0" "10000.0"; do
for seed in "42" "43" "44" "45" "46" "47" "48" "49" "50" "51"; do

lines=`qstat -n -1 | grep nKFCVv0 | wc -l`

while [ "$lines" -gt 79 ]
do
	echo "$lines jobs, waiting..."
	sleep 30
	lines=`qstat -n -1 | grep nKFCVv0 | wc -l`
done

echo "#!/bin/sh
### Set the job name
#PBS -N nKFCVv0.$dataset.r$radius.l$lambda.$C

### Declare myprogram non-rerunable
#PBS -r n

### Optionally specifiy destinations for your myprogram output
### Specify localhost and an NFS filesystem to prevent file copy errors.
#PBS -e localhost:${HOME}/tesi/logs/err/${dataset}.$1v0.r$radius.l$lambda.s$seed.nested.err
#PBS -o localhost:${HOME}/tesi/logs/${dataset}.$1v0.r$radius.l$lambda.nested.out

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

python $HOME/cluster_bundle/master-thesis/experiments/cross_validation_from_matrix.py grams/${dataset}/$1v0/k$1.r$radius.l$lambda.mtx.svmlight $C $size $seed cvres/${dataset}/$1v0/k$1v0.r$radius.l$lambda"> $HOME/tesi/jobs/${dataset}.$radius.$lambda.$1.$C.nestedv0.job

qsub $HOME/tesi/jobs/${dataset}.$radius.$lambda.$1.$C.nestedv0.job

done
done
done
done
done

