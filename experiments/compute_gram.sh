#!/bin/bash

for dataset in "AIDS" "CPDB" "CAS" "NCI1"; do # "GDD"; do
for radius in "10" "9" "8" "7" "6" "5" "4" "3" "2" "1"; do
for lambda in "0.1" "0.5" "0.8" "0.9" "1.0" "1.1" "1.2" "1.3" "1.4" "1.5" "1.8"; do

lines=`qstat | grep gram | wc -l`

while [ "$lines" -gt 79 ]
do
	echo "$lines jobs, waiting..."
	sleep 30
	lines=`qstat | grep gram | wc -l`
done

echo "#!/bin/sh

### Set the job name
#PBS -N $dataset.r$radius.l$lambda.$1.gram

### Declare myprogram non-rerunable
#PBS -r n

### Optionally specifiy destinations for your myprogram output
### Specify localhost and an NFS filesystem to prevent file copy errors.
#PBS -e localhost:${HOME}/tesi/logs/err/${dataset}.$1.MATRIX.r$radius.l$lambda.err
###PBS -o localhost:${HOME}/tesi/logs/${dataset}.$1.MATRIX.r$radius.l$lambda.out

### Set the queue to batch, the only available queue. 
#PBS -q cluster_long

### You MUST specify some number of nodes or Torque will fail to load balance.
### nodes=number of distinct host
### ppn=processes per node  :cache6mb
#PBS -l nodes=1:ppn=1:infiniband

### You should tell PBS how much memory you expect your job will use.  mem=1g or mem=1024
#PBS -l mem=3g

### You can override the default 1 hour real-world time limit.  -l walltime=HH:MM:SS
### Jobs on the public clusters are currently limited to 10 days walltime.
#PBS -l walltime=999:00:00


### Switch to the working directory; by default Torque launches processes from your home directory.
### Jobs should only be run from /home, /project, or /work; Torque returns results via NFS.

cd $HOME/cluster_bundle/scikit-learn-graph/

python -m scripts/calculate_matrix_allkernels ${dataset} $radius $lambda grams/$dataset/$1v1/k$1.r$radius.l$lambda.mtx $1 True 1"> $HOME/tesi/jobs/${dataset}.$radius.$lambda.$1.gram.job

qsub $HOME/tesi/jobs/${dataset}.$radius.$lambda.$1.gram.job

done
done
done
