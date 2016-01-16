#!/bin/bash

#for dataset in "CAS" "NCI1" "AIDS" "CPDB" "GDD"; do
#for lambda in "0" "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1.0"; do
for dataset in "CAS"; do
for lambda in "0"; do
for seed in "42"; do

#lines=`qstat | grep cmass | wc -l`
#
#while [ "$lines" -gt 30 ]
#do
#	echo "$lines jobs, waiting..."
#	sleep 30
#	lines=`qstat | grep cmass | wc -l`
#done
#
#echo "#!/bin/sh
#### Set the job name
##PBS -N thexp.r$radius.l$lambda.$dataset.$C.nested
#
#### Declare myprogram non-rerunable
##PBS -r n
#
#### Optionally specifiy destinations for your myprogram output
#### Specify localhost and an NFS filesystem to prevent file copy errors.
##PBS -e localhost:${HOME}/tesi/logs/err/${dataset}.mkl.L$lambda.nested.err
##PBS -o localhost:${HOME}/tesi/logs/${dataset}.mkl.L$lambda.nested.out
#
#### Set the queue to batch, the only available queue. 
##PBS -q cluster_long
#
#### You MUST specify some number of nodes or Torque will fail to load balance.
#### nodes=number of distinct host
#### ppn=processes per node  :cache6mb
##PBS -l nodes=1:ppn=1:infiniband
#
#### You should tell PBS how much memory you expect your job will use.  mem=1g or mem=1024
##PBS -l mem=24g
#
#### You can override the default 1 hour real-world time limit.  -l walltime=HH:MM:SS
#### Jobs on the public clusters are currently limited to 10 days walltime.
##PBS -l walltime=30:00:00
#
#
#### Switch to the working directory; by default Torque launches processes from your home directory.
#### Jobs should only be run from /home, /project, or /work; Torque returns results via NFS.
#
#cd $HOME/cluster_bundle/scikit-learn-graph/

python -u cv_all_matrices_mkl.py $lambda cvres/${dataset}/mkl/multikernel $seed 4337 grams/${dataset}/mkl/*.svmlight #"> $HOME/tesi/jobs/${dataset}.$lambda.nested.job

#qsub $HOME/tesi/jobs/${dataset}.$lambda.nested.job

done
done
done

