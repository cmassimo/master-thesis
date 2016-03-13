#!/bin/bash

echo "#!/bin/sh

### Set the job name
#PBS -N mkl.eigenvals.grams

### Declare myprogram non-rerunable
#PBS -r n

### Optionally specifiy destinations for your myprogram output
### Specify localhost and an NFS filesystem to prevent file copy errors.
#PBS -e localhost:${HOME}/tesi/logs/err/eigens.err
#PBS -o localhost:${HOME}/tesi/logs/eigens.out

### Set the queue to batch, the only available queue. 
#PBS -q cluster_short

### You MUST specify some number of nodes or Torque will fail to load balance.
### nodes=number of distinct host
### ppn=processes per node  :cache6mb
#PBS -l nodes=1:ppn=1:infiniband

### You should tell PBS how much memory you expect your job will use.  mem=1g or mem=1024
#PBS -l mem=2g

### You can override the default 1 hour real-world time limit.  -l walltime=HH:MM:SS
### Jobs on the public clusters are currently limited to 10 days walltime.
#PBS -l walltime=02:59:00


### Switch to the working directory; by default Torque launches processes from your home directory.
### Jobs should only be run from /home, /project, or /work; Torque returns results via NFS.

cd $HOME/cluster_bundle/scikit-learn-graph/

python -u $HOME/cluster_bundle/master-thesis/experiments/calculate_matrices_eigenvalues.py $1 $2/*svmlight"> $HOME/tesi/jobs/eigens.job

qsub $HOME/tesi/jobs/eigens.job

