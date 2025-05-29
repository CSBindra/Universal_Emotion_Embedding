# Universal Emotion Embedding Using Supervised Contrastive Learning

Welcome to the official repository for the project "**Universal Emotion Embedding Using Supervised Contrastive Learning**", wherein we use Supervised Contrastive Loss to Fine Tuning of Bert Encoder to get Universal Emotion Embeddings across multiple datasets.

The code in this repository is a part of the research project for 'Advanced Techniques in Computation Semantics' course taught at the 'University of Amsterdam'.


## Table of Content
- [Installation Guide](#installation-guide)
- [Datasets](#datasets)
- [How to Run](#how-to-run)
  - [Running Supervise Contrastive Encoder Training Code](#running-supervise-contrastive-encoder-training-code)
  - [Running Evaluation Code by training Heads per test dataset](#running-evaluation-code-by-training-heads-per-test-dataset)


## Installation Guide
<ol type=1>
<li> Download the repository as a .zip file.</li>
<br>
<li> Run the following steps <b>only if</b> running the code on local machine:
<ol type='a'>
<li> Install the correct version of the used packages from the .yml file using the following command:
<b>conda env create -f universal_encoder_env.yml</b>
</li>
<li> Upon installation of the environment, it can be (de)activated using:
<br>
<b>conda activate universal_encoder_env</b>
<br>
or
<br>
<b>conda deactivate universal_encoder_env</b>
</li>

<li> The environment can be deleted using:
<b>conda remove -n universal_encoder_env --all</b>
</li>

<li> Additional packages can be installed using pip:
<b>pip install [package_name]</b>
</li>
</ol>
</li>
<br>
<li> Run the following steps <b>only if</b> running the code on snellius cluster:
<ol type='a'>
<li> Copy the extracted zip folder onto snellius.</li>
<li> Install the environment from the .yml file by running the create environment slurm job 
using the following command:
<b>sbatch [path_to_copied_zip_folder]/Snellius_Jobs/create_environment.job</b>
</li>
<li> Additional packages can be installed by uncommenting the last line of update_env.job and 
replacing the [package_name] with the desired package name. For multiple packages, copy 
and paste the last line once per package and replace the [package_name] with the 
desired package name. After saving the job, run it using:
<b>sbatch [path_to_copied_zip_folder]/Snellius_Jobs/update_environment.job</b>
</li>
</ol>

## Datasets
The Emotion datasets that are used by the code are already in the repository as zip file and 
do not need to be separately downloaded.

## How to Run
Now that the environment has been correctly installed, it is time to run the code.

### Running Supervise Contrastive Encoder Training Code

To run the training code for supervised contrastive encoder, run the following command:
<br>
<b>sbatch [path_to_copied_zip_folder]/Snellius_Jobs/train_sup_con_model.job</b>
<br><br>
You can also modify the default behaviour by editing the train_sup_con_model.job file by
adding command line arguments in the last line command.
<br><br>
The following command line arguments are available for use:
<br><br>
<ol type=1>
  <li> base_encoder_model - This argument is used to specify the name of the base encoder model to finetune using supervised contrastive learning. It's default value is 'FacebookAI/roberta-base'. Currently, it can only use the default value.</li>
  <br>
  <li> train_output_dir - This argument is used to specify the name of the base directory where training results will be saved. It's default value is 'Results'. It can take any string value.</li>
  <br>
  <li> raw_data_extract_dir_name - This argument is used to specify the directory where raw emotion dataset will be extracted (from the zip provided in the repository). It's default value is 'raw_data'. It can take any string value.</li>
  <br>
  <li> preprocessing_data_dir - This argument is used to specify the directory where the pre procesed datasets will be saved (per category) after concatenating into a single Hugging face dataset. It's default value is 'Concatenated_Datasets'. It can take any string value.</li>
  <br>
  <li> tensorboard_log_dir - This argument is used to specify the directory where the tensorboard logs will be saved for training. It's default value is 'tensorboard_logs'. It can take any string value.</li>
  <br>
  <li> val_check_interval - This argument is used to specify the number of steps to wait before running the validation. It's default value is 1000. It can take any positive integer value.</li>
  <br>
  <li> patience - This argument is used to specify the number of epochs to tolerate for no performance improvement before aborting the training. It's default value is 2. It can take any positive integer value.</li>
  <br>
  <li> min_delta - This argument is used to specify the minimum performance change to tolerate before considering the epoch as no improvement. It's default value is 1e-4. It can take any positive float value.</li>
  <br>
  <li> num_epochs - This argument is used to specify the maximum number of epochs to run the training (if the performance keeps on improving and training is not aborted). It's default value is 5. It can take any positive integer value.</li>
  <br>
  <li> batch_size - This argument is used to specify the batch size for the dataloader. It's default value is 32. It can take any positive integer value.</li>
  <br>
  <li> group_size - This argument is used to specify the number of distinct label anchors per batch. It's default value is 5. It can take any positive integer value less than the batch size.</li>
  <br>
  <li> learning_rate - This argument is used to specify the learning rate for AdamW optimizer for training. It's default value is 5e-5. It can take any positive float value.</li>
  <br>
  <li> weight_decay - This argument is used to specify the weight decay for AdamW optimizer for training. It's default value is 0.01. It can take any positive integer value.</li>
  <br>
  <li> num_warmup_steps - This argument is used to specify the number of warmup steps for the linear learning rate scheduler. It's default value is 500. It can take any positive integer value.</li>
</ol>
<br><br>
All training output will be saved in the [output_dir] folder (created automatically by the code).

### Running Evaluation Code by training Heads per test dataset

To run the evaluation code to train heads per test dataset in order to test the performance 
of supervised contrastive encoder, run the following command:
<br>
<b>sbatch [path_to_copied_zip_folder]/Snellius_Jobs/eval_sup_con_model.job</b>
<br><br>
You can also modify the default behaviour by editing the eval_sup_con_model.job file by
adding command line arguments in the last line command.
<br><br>
The following command line arguments are available for use:
<br><br>
<ol type=1>
  <li> base_encoder_model - This argument is used to specify the name of the base encoder model to finetune using supervised contrastive learning. It's default value is 'FacebookAI/roberta-base'. Currently, it can only use the default value.</li>
  <br>
  <li> pre_trained_encoder - This argument is used to specify whether to use the supervised contrastive learning trained encoder or go ahead with default encoder (in order to get baseline performance). It's default value is 'yes'. It can take one of two values: 'yes' or 'no'.</li>
  <br>
  <li> freeze_encoder - This argument is used to specify whether to freeze the parameters of the encoder while training the classification head on test datasets. It's default value is 'yes'. It can take one of two values: 'yes' or 'no'.</li>
  <br>
  <li> dataset_category - This argument is used to specify dataset category for test datasets for which the evaluation is to be run. It's default value is 'continuous_label'. It can take one of three values: 'continuous_label', 'discrete_single_label' or 'discrete_multi_label'.</li>
  <br>
  <li> eval_output_dir - This argument is used to specify the name of the base directory where evaluation results will be saved. It's default value is 'Eval_Results'. It can take any string value.</li>
  <br>
  <li> encoder_dir - This argument is used to specify the training output base directory where the trained encoder model was saved. It's default value is 'Results'. If some other value was passed as train_output_dir while training, the same needs to be passed here. It can take any string value.</li>
  <br>
  <li> raw_data_extract_dir_name - This argument is used to specify the directory where raw emotion dataset will be extracted (from the zip provided in the repository). It's default value is 'raw_data'. It can take any string value.</li>
  <br>
  <li> tensorboard_log_dir - This argument is used to specify the directory where the tensorboard logs will be saved for evaluation. It's default value is 'eval_tensorboard_logs'. It can take any string value.</li>
  <br>
  <li> num_epochs - This argument is used to specify the maximum number of epochs to run the training for heads before evaluation. It's default value is 10. It can take any positive integer value.</li>
  <br>
  <li> batch_size - This argument is used to specify the batch size for the dataloader. It's default value is 8. It can take any positive integer value.</li>
  <br>
  <li> learning_rate - This argument is used to specify the learning rate for Hugging face trainer for evaluation. It's default value is 1e-4. It can take any positive float value.</li>
  <br>
  <li> weight_decay - This argument is used to specify the weight decay for Hugging face trainer for evaluation. It's default value is 0.01. It can take any positive integer value.</li>
</ol>
<br><br>
All evaluation output will be saved in the [output_dir] folder (created automatically by the code).
