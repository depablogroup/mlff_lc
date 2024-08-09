# Training

This directory holds the script that is used for learn machine learning potentials.

## Configs, Templates, and Config Series
Both the `configs` and `templates` directory stores raw yaml files that are 
later used to generate a series of Allegro config files.
Allegro model training involves lots of parameters, most of which doesn't need
to be changed across different experiments. As such, we store the subset of parameters
that we would like to vary inside the `configs` directory, and store the full
config that serves as a template for such edit in the `templates` directory.
To train a model, we will need to run `scripts/auto_generate_config.py` to combine
a `config` yaml with a `template` yaml, which turns them into a set of config
files in the `config_series` directory.

## Scripts
The `scripts` directory stores most of the scripts we created to train the
Allegro models, to compare their performance, and to deploy them for usage.
A more detailed discription can be found [here](./scripts/README.md)

## Results
The `results` directory stores the Allegro training results. Each subdirectory
corresponds to one or more config series.
The results includes the following:
- Trained model weights (*.pth)
- Deployed model weights (usually named to model.pth) that can be used as force fields
- Related Metrics during training (*.csv, *.png). Each experiment is accompanied with its
  own metrics file (*.csv). A series of experiments may be accompanied with a visualized
  summary (*.png).
- Preprocessed datasets (processed_dataset_*)

## Deploy
Deprecated. A sample script to deploy the models. Currently the deployment is
done with `scripts/auto_deploy.py`.
