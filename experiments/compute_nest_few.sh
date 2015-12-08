#!/bin/bash

for dataset in "CAS" "NCI1"; do
for radius in "3" do
for lambda in "0.1" "0.5" "0.8" "0.9" "1.0" "1.1" "1.2" "1.3" "1.4" "1.5" "1.8"; do
###for lambda in "1.0"; do
for C in "0.0001" "0.001" "0.01" "0.1" "1.0" "10.0" "100.0" "1000.0"; do

echo "#!/bin/sh
### Set the job name
#PBS -N THCMASS.r$radius.l$lambda.$dataset.$C.nested

### Declare myprogram non-rerunable
#PBS -r n

### Optionally specifiy destinations for your myprogram output
### Specify localhost and an NFS filesystem to prevent file copy errors.
###PBS -e localhost:${HOME}/prova.err
###PBS -o localhost:/scratch/nnavarin/ODDCLSTNEW/AUTO/LOGS/${dataset}.$kernel.MATRIX.r$r.l$l.out

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

cd $HOME/tesi/scikit-learn-graph/

python cross_validation_from_matrix.py ../gram/${dataset}/${dataset}.r$radius.l$lambda.n$1.v$2.e$3.a$4.libsvm $C ../nested/${dataset}/${dataset}.r$radius.l$lambda.n$1.v$2.e$3.a$4"> $HOME/tesi/jobs/${dataset}.$radius.$lambda.$1.$2.$3.$4.$C.nested.job

qsub $HOME/tesi/jobs/${dataset}.$radius.$lambda.$1.$2.$3.$4.$C.nested.job

done
done
done
done
