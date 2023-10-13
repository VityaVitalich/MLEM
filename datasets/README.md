I hope to update it with time. For now:

### 0. Raw data
- Do it yourself.

### 1. CSV file
Before using datasets, preprocessing has to be done:
- None filled
- Categorical cleaned, encoded
- amount features log applied
- time either in unix timestapm or float. Time column ALWAYS named "event_time"
- index field must be present
- Describe each feature type. Create dataset config as yaml file.

Transactions need to be grouped. Each group is assigned a sequential index, a target, and a temporal component.
Concatenate all transactions and put in one csv file. Or make ".parquet" folder with labeled and unlabeled transactions combined. Unlabeled transactions must have target value=None.

### 2. Dataset
1. load_dataset() 
    - train/test split
    - 