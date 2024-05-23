from collections import OrderedDict
from datetime import datetime

import thr_students

from utils.plot_utils import plot_metrics as plot_metrics
from utils.experiments_utils import run_experiment as run_experiment_students


if __name__ == '__main__':

    # Approaches to test
    approaches = ['fairdas', 'baseline']


    # Define data
    config = dict(
        historical_batches=30,
        n_batches=100,
        batch_dim=32,

        # Define dynamics
        eigen=0.2,
    )


    # To seed data generation process
    seeds = [1234, 3245, 4242, 5627, 6785, 8282, 9864, 9921]


    # Threshold configuration to test
    experiments = thr_students.thresholds
    thresholds = [experiments[exp] for exp in experiments.keys()]
    list_metrics = list(experiments['exp0'].keys())

    for thr in thresholds:
        config['threshold'] = thr
        # Run experiment
        experiment_name = str(datetime.now())[:-7]
        records_path = run_experiment_students(approaches=approaches,
                                                   config=config,
                                                   list_metrics=list_metrics,
                                                   experiment_name=experiment_name,
                                                   seeds=seeds,
                                                   )
        plot_metrics(records_path)

