# Multimodal Learning Event Model


### Data
Here, we provide anonymous links for downloading data

1. [ABank](https://drive.google.com/file/d/1zHDbl1UE8Rv2HlLdcqvlqvtbUYi_ju76/view?usp=sharing)
2. [Age](https://drive.google.com/drive/folders/1Zwyc2EGEmfkbAsT5ilOZlHwNM3Y966MC?usp=sharing)
3. [PhysioNet](https://drive.google.com/drive/folders/1ZxJJ07WPEuziBW4tyrm_8171N0BTxkJl)
4. [Pendulum](https://drive.google.com/file/d/1E6L2kX4JSLZWv8fHi9fsdTZpKhbmB0oG/view?usp=sharing)
5. [TaoBao](https://drive.google.com/file/d/1nmFyuLb3TnXkgt2DiRaJJummz5Tbuw1A/view?usp=sharing)

After downloading those experiments they should be placed in directory ```experiments/{dataset_name}/data/```

### Configurations

Configurations are splitted into data configs and models configs. Data configs are for different paths to data, random seed, batch size, names of columns, and specifics for contrastive augmentations. Model configs are for changing model parameters, such as which model to train, hidden sizes, which encoder to use, normalizations, etc.

Data configurations for each dataset are placed in directory ```configs/data_configs/{dataset_name}.py``` for Generative and MLEM models and in directory ```configs/data_configs/contrastive/{dataset_name}.py``` for Contrastive and Naive models.

Data configurations for each dataset are placed in directory ```configs/model_configs/{required_model_name}/{dataset_name}.py```.

Due to origin of MLEM model technique here it is named Sigmoid, after the loss function. Naive method is called GC, that stands for Gen-Contrastive.

### Running experiments

All experiments run with the sh scripts. To change the dataset for experiment you simply need to pass your dataset config to this sh script.

#### Supervised Learning

```run_pipe_supervised.sh``` script will run the supervised experiment. Inside the script one could change which configs to take, how to name experiment, number of epochs and if to use checkpoints. 

#### Contrastive Learning

```run_pipe_contrastive.sh``` script does contrastive learning.

#### Generative modeling

```run_pipe_gen.sh``` does generative modeling

#### Naive method

```run_pipe_gen_contrastive.sh``` does the procedure described as Naive in paper

#### MLEM

```run_pipe_sigmoid.sh``` does MLEM modeling. However before running this script, path to contrastive checkpoint should be placed in MLEM config and pre-trained contrastive net configs should match with contrastive net configs inside MLEM model config.

### TPP and Robsutness Check

For evaluating TPP task and robustness of embeddings one needs first to run ```tpp_dataset.py``` with passing desired dataset as argument. Further this script will generate refactored data for evaluation. 

Then one should run ```run_tpp.sh``` script with argmunets that are suitable. This means one should provide path to configs of tested model, data config of dataset, name of current model and checkpoints to test. 

### Reproducing Results

In the current version of repo are appropriate configs. To obtain the same results one need to just pass right configs to the desired experiment setting.
