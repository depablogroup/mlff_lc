# Training Scripts
This directory holds the scripts for Allegro model Training.

## General Pipeline for Training
1. Create a config yaml file under `../configs`, including all the parameters you
would like to tune or change. The set of all possible parameters to tune
can be found [here](https://github.com/mir-group/nequip/blob/main/configs/full.yaml).

2. (Optional) Create a new template yaml file under `../templates`. A sample
template can be found [here](../templates/example.yaml).

3. Generate a set of configs using `auto_generate_config.py`.  By default, the configs will be saved
under `../config_series`, which will be automatically provided as a prefix.

4. Submit Allegro training Jobs to the cluster. This is typically handled by
`auto_submit.py`. You will need to specify your username, the name of the partition
you would like to submit your jobs to, as well as the name of the
config series.
The script will automatically detect the following:
    - Whether there exists some training jobs under this config series that have
      not finished yet. If the job is ongoing, a new job will not be submitted.
    - Whether some of the jobs have already converged. If the job has already
      converged, a new job will not be submitted.
You can optionally use the `--dryrun` option to check what jobs will be submitted
without really submitting them to the cluster.

5. (Optional) Compare the performance of the trained model via `loss_comparison.py`.
It will generate a figure `your_selected_metric.png` under the selected config 
series directory. 

6. Deploy the model when training has finished due to time limit or converged.
By default, it will create a file called `model.pth` under 
`../results/config_serie_name/experiment_name`.


