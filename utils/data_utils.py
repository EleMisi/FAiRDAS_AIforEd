import os
import pickle
import random

import numpy as np

import const_define as cd


class Batch_Gen():

    def __init__(self, batch_dim: int, n_batches: int, n_historical_batches: int, seed: int):

        self.seed = seed

        self.data, self.group_masks, self.group_labels, self.group_idxs = self.load_data()
        self.n_groups = len(self.group_labels)
        self.pool_data = len(self.data['score'])
        self.n_batches = n_batches
        self.n_historical_batches = n_historical_batches
        self.batch_dim = batch_dim
        self.n_samples = self.n_batches * self.batch_dim
        self.sampling_noise_distr = self.create_sampling_noise_distr(batch_dim, n_batches + n_historical_batches)
        self.uniform_distr = self.create_uniform_query_distr(batch_dim, n_batches + n_historical_batches)

    def load_data(self):
        """Load data and split in ESCS groups"""
        path = os.path.join(cd.PROJECT_DIR, 'utils', 'resources.pkl')

        with open(path, 'rb') as handle:
            data = pickle.load(handle)

        # Split data in groups
        group_labels = np.unique(data['ESCS_quantized'].argmax(1))
        group_masks = {}
        groups_idxs = {}
        tot_samples = 0
        for e in group_labels:
            group_masks[e] = data['ESCS_quantized'].argmax(1) == e
            tot_samples += group_masks[e].sum()
            groups_idxs[e] = list(np.arange(len(data['ESCS_quantized']))[group_masks[e]])
        assert tot_samples == len(data['ESCS_quantized'])

        return data, group_masks, group_labels, groups_idxs

    def create_sampling_noise_distr(self, batch_dim: int, n_batches: int):
        """
        """
        self.set_seed(self.seed)
        idxs = np.zeros((n_batches, batch_dim))
        # Each batch of queries belongs to a single topic
        for i in range(n_batches):
            current_group = random.choice(range(len(self.group_labels)))
            # Sampling from elements from current group
            sampled_idxs = random.sample(self.group_idxs[current_group], batch_dim)
            idxs[i] = sampled_idxs
        return idxs.reshape(-1)

    def create_uniform_query_distr(self, batch_dim: int, n_batches: int):
        """
        The students in a batch are randomly sampled
        """
        self.set_seed(self.seed)
        idxs = np.zeros((n_batches, batch_dim))
        # In each batch, the topic of a query is sampled randomly
        for i in range(n_batches):
            # Sampling elements for the current batch
            sampled_idxs = random.sample(range(self.pool_data), batch_dim)
            idxs[i] = sampled_idxs
        return idxs.reshape(-1)

    def generate_batches(self, d1: np.array, d2: np.array, alpha: float):
        """
        Sampling queries by interpolating two discrete distributions with weight (alpha).
        :param d1: possible values of 1st discrete random variable
        :param d2: possible values of 2nd discrete random variable
        :param alpha: probability of sampling from 1st distribution

        :return: vector of queries
        """
        self.set_seed(self.seed)

        # Sampling random numbers
        samples = np.random.random(self.n_samples + self.n_historical_batches * self.batch_dim)
        # If sample < alpha, then it's belong to d1; otherwise to d2
        d1_samples = samples < alpha
        d2_samples = 1 - d1_samples
        # Creating resources by interpolating d1 and d2
        idx = d1 * d1_samples + d2 * d2_samples

        # Reshape idx
        idx = idx.reshape((self.n_batches + self.n_historical_batches, self.batch_dim)).astype(int)

        historical_batches_score = np.zeros((self.n_historical_batches, self.batch_dim, 1))
        historical_batches_ESCS = np.zeros((self.n_historical_batches, self.batch_dim, len(self.group_labels)))
        for i in range(self.n_historical_batches):
            historical_batches_score[i] = self.data['score'][idx[i]][:, None]
            historical_batches_ESCS[i] = self.data['ESCS_quantized'][idx[i]]

        batches_score = np.zeros((self.n_batches, self.batch_dim, 1))
        batches_ESCS = np.zeros((self.n_batches, self.batch_dim, len(self.group_labels)))
        for i in range(self.n_historical_batches, self.n_batches + self.n_historical_batches):
            batches_score[i - self.n_historical_batches] = self.data['score'][idx[i]][:, None]
            batches_ESCS[i - self.n_historical_batches] = self.data['ESCS_quantized'][idx[i]]

        return {'score': historical_batches_score, 'ESCS': historical_batches_ESCS}, {'score': batches_score,
                                                                                      'ESCS': batches_ESCS}

    def set_seed(self, seed):
        """
        Fix seed for reproducibility
        :param seed: seed to be set
        :return: set seed
        """
        seed = cd.set_seed(seed)
        return seed
