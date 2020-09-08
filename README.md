# Denoising-Autoencoder-Analyzer

* Scripts are ran on python 2.7.15
* `rna_solidtumor_tcgahnsc.csv` must in the `cmd_line/Model/`

### Running Job on Argon to Generate Models

`cd cmd_line/Model/
qsub create_model.sh`

#### Other Scripts (Not very much important)

> Generates weight(s) and store them as csv. Parameters are given in the script

`run_get_weigths.sh
`
> Similar to *create_model.sh* but add hyperparameter to decoder functions

`cd cmd_line/Model/
qsub create_model_revise.sh` 

> Transfers input data into a smaller dimensional dataset based on the encocder. 

`cd cmd_line/autoencoders/
qsub run_compress_data.sh`


#### To continue where you left off if job is stopped 
`qsub restart_create_model.sh 5`

* second paramater (5) is the model to start on to continue generation

##### Questions
Email chibuzo-nwakama@uiowa.edu
