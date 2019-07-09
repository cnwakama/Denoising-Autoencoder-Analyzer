#!/bin/bash

#####Set Scheduler Configuration Directives#####
#Set the name of the job. This will be the first part of the error/output filename.
#$ -N ${name}

#Set the current working directory as the location for the error and output files.
#(Will show up as .e and .o files)
#$ -cwd

#Send e-mail at beginning/end/suspension of job
#$ -m bes

#E-mail address to send to
#$ -M chibuzo-nwakama@uiowa.edu
#####End Set Scheduler Configuration Directives#####


#####Resource Selection Directives#####
#See the HPC wiki for complete resource information: https://wiki.uiowa.edu/display/hpcdocs/Argon+Cluster
#Select the queue to run in
#$ -q CG,COE,UI,all.q

#Select the number of slots the job will use
#$ -pe smp 56

#Indicate that the job requires a GPU
#$ -l gpu=true

#Sets the number of requested GPUs to 1
#$ -l ngpus=1

#Indicate that the job requires a mid-memory (currently 256GB node)
#$ -l mem_256G=true

#Indicate the CPU architecture the job requires
##$ -l cpu_arch=broadwell

#Specify a data center for to run the job in
##$ -l datacenter=LC

#Specify the high speed network fabric required to run the job
##$ -l fabric=omnipath
#####End Resource Selection Directives#####

module load python/2.7.15

# default parameters for non argon users
dataset='rna_data_v2.npy'
directory=/Volumes/Files\ Backups/Document_12-12-18/New\ Folder\ With\ Items/yadlt/models/
output='InputsV2/'

python2.7 compress_data.py --dataset "${dataset}" --directory "${directory}" --output "${output}"
