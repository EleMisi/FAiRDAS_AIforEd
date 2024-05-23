import json
import os
import pickle
from collections import OrderedDict

import numpy as np

from utils.data_utils import Batch_Gen

np.seterr(all='raise')
from scipy import optimize

import const_define as cd
from utils import metrics
from utils.dyn_systems import DynSystem
from utils.obj_functions import ObjFun_DynStateVar, ObjFun_BaselineSum, ObjFun_Normalization


def run_experiment(approaches: list, config: dict, list_metrics: list, experiment_name: str, seeds: list):
    alpha = 0.1  # alpha * Uniform + (1-alpha) * Sampling noise
    eigen = config['eigen']
    matrix_A = np.eye(len(list_metrics), len(list_metrics)) * eigen
    threshold_dict = config['threshold']
    scaling = 'IQR_normalization'
    # Define threshold
    current_thrs = tuple([threshold_dict[m] for m in list_metrics])

    record_path = os.path.join(cd.PROJECT_DIR, 'results', 'records', str(current_thrs), experiment_name)
    if not os.path.isdir(record_path):
        os.makedirs(record_path)

    # Save config file
    with open(os.path.join(record_path, 'config.json'), 'w',
              encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    # Records
    metrics_record = {a:
                          {m:
                               {seed: [] for seed in seeds}
                           for m in list_metrics + ['scaling_factors']}
                      for a in approaches}

    rankings_record = {a:
                           {seed: [] for seed in seeds}
                       for a in approaches}

    actions_record = {a:
                          {seed: [] for seed in seeds}
                      for a in approaches}

    batches_records = {seed: [] for seed in seeds}


    # Iterate over seed
    for seed in seeds:
        print()
        print('*' * 40)
        print(f'Test with seed: {seed}')
        print('*' * 40)
        data_gen = Batch_Gen(batch_dim=config['batch_dim'], n_batches=config['n_batches'],
                             n_historical_batches=config['historical_batches'], seed=seed)
        n_groups = data_gen.n_groups
        historical_batches, batches = data_gen.generate_batches(data_gen.uniform_distr, data_gen.sampling_noise_distr,
                                                                alpha=alpha)

        print(f'----Test with thr: {current_thrs}')
        print(f'----Test with alpha: {alpha}')

        # Iterate over approaches
        for approach in approaches:
            print('----Approach:', approach, )

            # Set global seed
            cd.set_seed(seed)

            # Store queries and resources for current seed
            if batches_records[seed] == []:
                batches_records[seed] = batches

            # Otherwise check if the new queries and resources are equal to the stored ones
            else:
                assert (batches_records[seed]['score'] == batches['score']).all() and (
                            batches_records[seed]['ESCS'] == batches[
                        'ESCS']).all(), f'Mismatch between generated and stored resources for seed {seed}'

            # Build approach
            approach_components = build_approach(historical_batches, approach, matrix_A, threshold_dict, list_metrics,
                                                 scaling=scaling)
            theta_vector = approach_components['theta_vector']

            # Store scaling factor metrics
            metrics_record[approach]['scaling_factors'][seed] = approach_components['scaling_factors']

            # Iterate over queries batches
            n_batches = len(batches['score'])
            for i in range(n_batches):
                batch_scores = batches['score'][i]
                batch_escs = batches['ESCS'][i]

                # Compute metrics given actions found in previous step
                current_metrics = np.array([approach_components['metrics_dict'][name](batch_scores=batch_scores,
                                                                                      batch_escs=batch_escs,
                                                                                      theta_vector=theta_vector) for
                                            name in approach_components['metrics_dict']])

                # Store current metrics
                for idx, m in enumerate(list_metrics):
                    metrics_record[approach][m][seed].append(float(current_metrics[idx]))

                # Store current actions
                actions_record[approach][seed].append(theta_vector)

                # Check if metrics are above threshold
                thresholds_arr = np.array(current_thrs)
                check = np.all(current_metrics <= thresholds_arr)
                if not check:

                    callback = None

                    res = optimizing(approach=approach, approach_components=approach_components,
                                     thresholds=thresholds_arr, batch_scores=batch_scores, batch_escs=batch_escs,
                                     current_metrics=current_metrics, theta_vector=theta_vector, callback=callback)
                    theta_vector = res.x[:n_groups]


    # Store results
    store_records(metrics_record, rankings_record, actions_record, batches_records, record_path)


    return os.path.join(str(current_thrs), experiment_name)


def optimizing(approach: str, approach_components: dict, thresholds: np.array, batch_scores: np.ndarray,
               batch_escs: np.ndarray, current_metrics: np.array, theta_vector: np.array, callback=None):
    # Load current batch in the objective function
    if approach.lower() == 'fairdas':

        # Compute dynamical state
        x = approach_components['dyn_sys'](current_metrics)
        thr_check = x >= thresholds

        if thr_check.all():

            approach_components['obj_fun'].load_current_batch(batch_scores=batch_scores, batch_escs=batch_escs, y=x)
            # Minimize
            res = optimize.minimize(approach_components['obj_fun'], x0=theta_vector,
                                    bounds=approach_components['bounds'], method='SLSQP',
                                    constraints=approach_components['cons'],
                                    callback=callback, options={'maxiter': 100, 'disp': False})


        else:
            y = np.zeros_like(x)
            var_idx = []
            # Cycle over metrics
            for i in range(thr_check.shape[0]):
                # If state geq than threshold
                if thr_check[i]:
                    y[i] = x[i]
                # If state less than threshold
                else:
                    y[i] = -1
                    var_idx.append(i)

            # Load current batch in the objective function
            approach_components['obj_fun'].load_current_batch(batch_scores=batch_scores, batch_escs=batch_escs, y=y)

            # Initial guess parameters
            initial_guess = np.zeros(len(theta_vector) + len(var_idx))
            initial_guess[:len(theta_vector)] = theta_vector
            for i, idx in enumerate(var_idx):
                initial_guess[len(theta_vector) + i] = x[idx]

            # Add bound for y
            if approach_components['obj_fun'].scaling == 'IQR_normalization':
                bounds = approach_components['bounds'] + tuple([(-1, thresholds[idx]) for idx in var_idx])
            else:
                bounds = approach_components['bounds'] + tuple([(0, thresholds[idx]) for idx in var_idx])

            # Load var idx in obj fn
            approach_components['obj_fun'].var_idx = var_idx

            res = optimize.minimize(approach_components['obj_fun'], x0=initial_guess,
                                    bounds=bounds, method='SLSQP',
                                    constraints=approach_components['cons'],
                                    callback=callback, options={'maxiter': 100, 'disp': False}, )



    elif approach.lower() == 'baseline':
        # Load current batch in the objective function
        approach_components['obj_fun'].load_current_batch(batch_scores=batch_scores, batch_escs=batch_escs,
                                                          thresholds=thresholds)
        # Minimize
        res = optimize.minimize(approach_components['obj_fun'], x0=theta_vector,
                                bounds=approach_components['bounds'], method='SLSQP',
                                constraints=approach_components['cons'],
                                callback=callback, options={'maxiter': 100, 'disp': False})

    return res


def build_approach(historical_data: np.ndarray, approach: str, matrix_A: np.ndarray, threshold: dict,
                   list_metrics: list, scaling: str):
    """
    Grounds approach building blocks and normalizes metrics.
    """

    n_groups = 4

    approach_components = {}

    metrics_dict = OrderedDict()
    for metric_name in list_metrics:
        if metric_name == 'DIDI':
            # Creating DIDI metric object
            didi_metric = metrics.DIDI(impact_function=metrics.impact_function)
            metrics_dict['DIDI'] = didi_metric

        if metric_name == 'score_absolute_error':
            # Create score distance object
            score_distance = metrics.ScoreAbsoluteError()
            metrics_dict['score_absolute_error'] = score_distance

    # Define initial value of theta
    theta_vector = np.zeros(n_groups)
    # Define bounds for theta (theta is in (0,1))
    bounds = ((0., 1.),) * n_groups
    # Define theta constraint (theta sum = 1)
    cons = ({'type': 'eq', 'fun': lambda x: 1 - sum(x[:n_groups])})

    # Find metrics minimum and maximum for normalization
    metrics_scaling_factors = metrics_scaling(metrics_dict=metrics_dict, historical_data=historical_data,
                                              theta_vector=theta_vector, bounds=bounds, cons=cons, scaling=scaling)

    if scaling == 'IQR_normalization':
        for name in metrics_scaling_factors:
            metrics_dict[name].q1 = metrics_scaling_factors[name]['q1']
            metrics_dict[name].q3 = metrics_scaling_factors[name]['q3']
            metrics_dict[name].scaling = scaling
    else:
        print('No metrics scaling performed!')


    # The metrics weights are 1
    metrics_weight = {name: 1 for name in metrics_dict}
    # metrics_weight['score_absolute_error'] = 10

    # Define objective function & dynamics
    if approach.lower() == 'fairdas':
        thr = tuple(threshold[metric_name] for metric_name in list_metrics)
        dyn_sys = DynSystem(matrix_A, np.array(thr))
        obj_fun = ObjFun_DynStateVar(metrics_dict=metrics_dict, metrics_weight=metrics_weight, n_thetas=n_groups,
                                     scaling=scaling)
        approach_components['dyn_sys'] = dyn_sys
    elif approach.lower() == 'baseline':
        obj_fun = ObjFun_BaselineSum(metrics_dict=metrics_dict, metrics_weight=metrics_weight, n_thetas=n_groups,
                                     scaling=scaling)

    # Store

    approach_components['cons'] = cons
    approach_components['bounds'] = bounds
    approach_components['theta_vector'] = theta_vector
    approach_components['scaling_factors'] = metrics_scaling_factors
    approach_components['metrics_weight'] = metrics_weight
    approach_components['metrics_dict'] = metrics_dict
    approach_components['obj_fun'] = obj_fun

    return approach_components


def metrics_scaling(metrics_dict: dict, historical_data: dict, theta_vector: np.array,
                    bounds: tuple,
                    cons, scaling: str):
    """
    Extracts values for metrics scaling
    """

    # Compose a single historical batch
    # hist_batch = np.vstack(historical_data)
    # Metrics record
    metrics_record = {name: [] for name in metrics_dict}

    # Obj fun for normalization
    obj_fun = ObjFun_Normalization(metrics_dict=metrics_dict,
                                   metrics_weight={name: 1 for name in metrics_dict},
                                   n_thetas=len(theta_vector),
                                   scaling='None')

    n_batches = len(historical_data['score'])
    for target_metric in metrics_dict:
        # Iterate over data
        for i in range(n_batches):
            scores = historical_data['score'][i]
            escs = historical_data['ESCS'][i]
            # Compute current metrics value with previous step actions
            for name, metric_fn in metrics_dict.items():
                metric = metric_fn(batch_scores=scores, batch_escs=escs, theta_vector=theta_vector)
                metrics_record[name].append(metric)

            # Optimize current target metric (if needed)
            if metrics_record[target_metric][-1] > 0.0:
                obj_fun.load_current_batch(batch_scores=scores, batch_escs=escs, target_metric=target_metric)
                res = optimize.minimize(obj_fun, x0=theta_vector, bounds=bounds, method='SLSQP', constraints=cons,
                                        callback=None, options={'maxiter': 100, 'disp': False})
                # Update theta vector
                theta_vector = res.x


    if scaling == 'IQR_normalization':
        # Extract q1 and q3 value for metrics
        metrics_q1_q3 = {name: {'q1': None, 'q3': None} for name in metrics_record}
        for name in metrics_record:
            metrics_q1_q3[name]['q1'] = np.quantile(metrics_record[name], 0.25)
            metrics_q1_q3[name]['q3'] = np.quantile(metrics_record[name], 0.75)

        return metrics_q1_q3


    else:
        return None

def store_records(metrics_record, rankings_record, actions_record, batches_records, record_path):
    # Metrics
    fname = os.path.join(record_path, f'metrics_record.pkl')
    with open(fname, 'wb') as f:
        pickle.dump(metrics_record, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Saved records {fname}', flush=True)

    # Ranking
    fname = os.path.join(record_path, f'ranking_record.pkl')
    with open(fname, 'wb') as f:
        pickle.dump(rankings_record, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Saved records {fname}', flush=True)
    # Actions
    fname = os.path.join(record_path, f'actions_record.pkl')
    with open(fname, 'wb') as f:
        pickle.dump(actions_record, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Saved records {fname}', flush=True)

    # Store batches
    fname = os.path.join(record_path, f'batches_record.pkl')
    with open(fname, 'wb') as f:
        pickle.dump(batches_records, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Saved records {fname}', flush=True)


