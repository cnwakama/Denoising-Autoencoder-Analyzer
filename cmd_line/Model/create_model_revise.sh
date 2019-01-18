#!/bin/bash

#####Set Scheduler Configuration Directives#####
#Set the name of the job. This will be the first part of the error/output filename.
#$ -N dae_revise

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
#$ -q COE-GPU,UI-GPU,all.q

#Select the number of slots the job will use
#$ -pe smp 28

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
feature_path='Models/features_revise.csv'
weight_path='Models/weights_revise.csv'
bias_v_path='Models/biasV_revise.csv'
bias_h_path='Models/biasH_revise.csv'
test_reconstruction='Models/test_reconstruction_revise.csv'
data=${HOME}'/.yadlt/data/dae/'

if [[ -d 'Models/' ]]; then
        #mkdir -p 'Models/'
        touch ${feature_path}
        touch ${weight_path}
        touch ${bias_h_path}
        touch ${bias_v_path}
        touch ${test_reconstruction}
fi

dataset=${data}'/Dataset/'

if [[ ! -d ${data} ]]; then
    mkdir -p ${dataset}
fi

model=${1}
name='dae_model'${model}
header="Model,Batch_Size,Learning_Rate,Corruption_Rate,Epochs,Encoder_Activation_Function,Decoder_Activation_Function"

# Hyper parameters declared = 1295 combination
batch_size=(1 10 20 50)
learning_rate=(0.01)
corruption_rate=(0.0 0.1 0.2)
epochs=(100 200 500)
act_fun_enc=(sigmoid tanh relu)
act_fun_dec=(none sigmoid tanh relu)

first='next_line'

echo 'Start Training Models'
for b in "${batch_size[@]}" ; do
    for l in "${learning_rate[@]}" ; do
        for c in "${corruption_rate[@]}" ; do
            for e in "${epochs[@]}"; do
                for enc in "${act_fun_enc[@]}" ; do
                    for dec in "${act_fun_dec[@]}" ; do
                        echo 'Model '"${model}"
                        mkdir -p "$data$name"

                        python ../autoencoders/run_autoencoder.py --n_components 100 --dataset custom \
                                --train_dataset ${dataset}'training_set.npy' \
                                --valid_dataset ${dataset}'validation_set.npy' \
                                --test_dataset ${dataset}'test_set.npy' --batch_size ${b} \
                                --num_epochs ${e} --learning_rate "${l}" --corr_type masking \
                                --corr_frac ${c} --enc_act_func ${enc} --dec_act_func ${dec} \
                                --loss_func cross_entropy --save_reconstructions "$data$name"'/'${name}'-reconstruction.npy' \
                                --save_parameters "$data$name"'/'${name} --name "$name" --seed 1 --normalize

                        printf '%s\n' ${name} ${b} ${l} ${c} ${e} ${enc} ${dec} | paste -d ',' -s - >> ${feature_path}

                        touch "$data$name"'/'${name}'-reconstruction.csv'
                        touch "$data$name"'/'${name}'-encoded_weight.csv'

                        printf '%s' 'Model' >> "$data$name"'/'${name}'-reconstruction.csv'
                        for i in `seq 1 20531`; do printf ",Feature %d" ${i}; done >> "$data$name"'/'${name}'-reconstruction.csv'
                        printf '%s' 'Model' >> "$data$name"'/'${name}'-encoded_weight.csv'
                        for i in `seq 1 20531`; do printf ",Feature %d" ${i}; done >> "$data$name"'/'${name}'-encoded_weight.csv'

                        python numpy_to_csv.py --input "$data$name"'/'${name}'-enc_w.npy' --output "$data$name"'/'${name}'-encoded_weight.csv' \
                        --name ${name} --mult_lines --next_line ${first}
                        python numpy_to_csv.py --input "$data$name"'/'${name}'-reconstruction.npy' \
                        --output "$data$name"'/'${name}'-reconstruction.csv' --name ${name} --mult_lines --next_line ${first}

                        python numpy_to_csv.py --input "$data$name"'/'${name}'-enc_w.npy' --output ${weight_path} \
                        --name ${name} --mult_lines --next_line ${first}
                        python numpy_to_csv.py --input "$data$name"'/'${name}'-enc_b.npy' --output ${bias_v_path} \
                        --name ${name} --next_line ${first}
                        python numpy_to_csv.py --input "$data$name"'/'${name}'-dec_b.npy' --output ${bias_h_path} \
                        --name ${name} --next_line ${first}
                        python numpy_to_csv.py --input "$data$name"'/'${name}'-reconstruction.npy' \
                        --output ${test_reconstruction} --name ${name} --mult_lines --next_line ${first}

                        first='hi'
                        ((model++))
                        name='dae_model'${model}
                    done
                done
            done
        done
	exit 0
    done
done
