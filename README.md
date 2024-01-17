# Multimodal Learning Event Model


### Data
Here, we provide anonymous links for downloading data

1. [ABank](https://anonymfile.com/58A3/abanktar.gz)
2. [Age](https://anonymfile.com/WjKD/drive-download-20240117t105225z-001.zip)
3. [PhysioNet](https://anonymfile.com/7kao/drive-download-20240116t191010z-001.zip)
4. [Pendulum](https://anonymfile.com/Ja0z/pendulum.zip)
5. [TaoBao](https://anonymfile.com/XjKK/drive-download-20240116t191125z-001.zip)

After downloading those experiments they should be placed in directory ```experiments/{dataset_name}/data/```

### Configurations

Configurations are splitted into data configs and models configs. Data configs are for different paths to data, random seed, batch size, names of columns, and specifics for contrastive augmentations. Model configs are for changing model parameters, such as which model to train, hidden sizes, which encoder to use, normalizations, etc.

Data configurations for each dataset are placed in directory ```configs/data_configs/{dataset_name}.py``` for Generative and MLEM models and in directory ```configs/data_configs/contrastive/{dataset_name}.py``` for Contrastive and Naive models.

Data configurations for each dataset are placed in directory ```configs/model_configs/{required_model_name}/{dataset_name}.py```.

Due to origin of MLEM model technique here it is named Sigmoid, after the loss function. Naive method is called GC, that stands for Gen-Contrastive.

### Running experiments


