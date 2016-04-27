#!/bin/bash

blades[0]="07"
blades[1]="08"
#rams[0]=60
#rams[1]=92
#for dset_length in "AIDS",1503 "CPDB",684 "GDD",1178 "CAS",4337 "NCI1",4110; do
for dset_length in "CAS",4337; do
#for dset_length in "AIDS",1503 "CPDB",684; do
dataset=${dset_length%,*}
size=${dset_length#*,}

#lines=`qstat | grep cmass | wc -l`
#
#while [ "$lines" -gt 59 ]
#do
#	echo "$lines jobs, waiting..."
#	sleep 60
#	lines=`qstat | grep cmass | wc -l`
#done

i=$RANDOM
let "i %= 2"
blade=${blades[i]}
#ram=${rams[i]}

echo "#!/bin/sh
### Set the job name
#PBS -N times.$dataset.mkl.stpc

### Declare myprogram non-rerunable
#PBS -r n

### Optionally specifiy destinations for your myprogram output
### Specify localhost and an NFS filesystem to prevent file copy errors.
#PBS -e localhost:${HOME}/tesi/logs/err/${dataset}.mkl.times.err
#PBS -o localhost:${HOME}/tesi/logs/${dataset}.mkl.times.out

### Set the queue to batch, the only available queue. 
#PBS -q cluster_long

### You MUST specify some number of nodes or Torque will fail to load balance.
### nodes=number of distinct host
### ppn=processes per node  :cache6mb
#PBS -l nodes=1:ppn=1:hpblade$blade

### You should tell PBS how much memory you expect your job will use.  mem=1g or mem=1024
#PBS -l mem=32g

### You can override the default 1 hour real-world time limit.  -l walltime=HH:MM:SS
### Jobs on the public clusters are currently limited to 10 days walltime.
#PBS -l walltime=999:00:00

### Switch to the working directory; by default Torque launches processes from your home directory.
### Jobs should only be run from /home, /project, or /work; Torque returns results via NFS.

cd $HOME/cluster_bundle/scikit-learn-graph/

python -u $HOME/cluster_bundle/master-thesis/experiments/times/easymkl_compute_times.py times/$dataset/$1 $size grams/$dataset/$2*.svmlight"> $HOME/tesi/jobs/${dataset}.times.mkl.job

qsub $HOME/tesi/jobs/${dataset}.times.mkl.job

done

