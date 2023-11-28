# Run experiment checklist
For each dataset:
- define _train_eval() for your pipeline. Ensure correct validation(metrics, datasets)
- check data and model configs
- get datasets inside data/ folder
- prepare sh script

For optuna:
- define grid in param_grids
- optuna_setup
    - metric
    - request_list
    - n_startup_trials # n_random trials. request included 
    - n_trials