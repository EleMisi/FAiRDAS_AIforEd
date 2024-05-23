import numpy as np


class ObjFun():
    def __init__(self, metrics_dict: dict, metrics_weight: dict, n_thetas: int, scaling: str):
        """
        :param metrics:
            dict of Metric object
        :param args_metrics:
            additional arguments of the fairness metrics
        :param metrics_weight:
            dict of weights to aggregate the metrics
        """

        self.metrics_dict = metrics_dict
        self.metrics_weight = metrics_weight
        self.n_metrics = len(metrics_dict)
        self.n_thetas = n_thetas
        self.scaling = scaling
        self.offset = 10  # To avoid non positive numbers

    def __call__(self, theta_vector: np.array):
        NotImplementedError()


class ObjFun_DynStateVar(ObjFun):

    def load_current_batch(self, batch_scores: np.ndarray, batch_escs: np.ndarray, y: np.ndarray):
        """

        :param queries_batch:
            pool of queries
        :param y:
            var to approximate

        """
        self.batch_scores = batch_scores
        self.batch_escs = batch_escs
        self.y = y

    def __call__(self, params: np.array):
        """
        Compute the squared error between real synamic state and approximated one.
        :param params:
            [vector of modifiers for country, var_to_approximate]

        :return:
            approximation error
        """

        # Distinguish between theta variable and threshold variables
        thr_vars = params[self.n_thetas:]
        theta_vector = params[:self.n_thetas]
        y = np.zeros_like(self.y)

        j = 0
        # Assign variable value
        for k, value in enumerate(self.y):
            # if == -1, then the dynamic state was less than the threshold and we want to be free to change it to improve the metrics
            if value == -1:
                y[k] = thr_vars[j]
                j += 1
            # else, the dynamic state was greater than the threshold and we want to move towards it
            else:
                y[k] = value

        # Compute x approximate
        x_approx = np.zeros((self.n_metrics,))
        weights = np.zeros((self.n_metrics,))
        for i, name in enumerate(self.metrics_dict):
            x_approx[i] = self.metrics_dict[name](batch_scores=self.batch_scores, batch_escs=self.batch_escs,
                                                  theta_vector=theta_vector)
            # Populate weight vector
            weights[i] = self.metrics_weight[name]

        # Scaling offset, if required
        if self.scaling == 'standardization' or self.scaling == 'IQR_normalization':
            y += self.offset
            x_approx += self.offset

        # Compute distance
        diff = x_approx - y
        # Square diff
        squared_diff = diff ** 2
        # Weighted sum
        cost = np.sum(squared_diff * weights) ** 0.5
        return cost


class ObjFun_BaselineSum(ObjFun):

    def load_current_batch(self, batch_scores: np.ndarray, batch_escs: np.ndarray, thresholds: dict):
        """

        :param queries_batch:
            pool of queries
        :param thresholds:
            dict with metrics thresholds

        """
        self.batch_scores = batch_scores
        self.batch_escs = batch_escs
        self.thresholds = thresholds

    def __call__(self, theta_vector: np.array):
        """
        Sum of metrics as cost function.
        :param theta_vector:
            vector of modifiers for country

        :return:
            sum of max(metric, threshold)
        """
        # Compute metrics
        metrics = np.zeros((self.n_metrics,))
        weights = np.zeros((self.n_metrics,))
        for i, name in enumerate(self.metrics_dict):
            # Max between metrics value and threshold
            metric = self.metrics_dict[name](batch_scores=self.batch_scores, batch_escs=self.batch_escs,
                                             theta_vector=theta_vector)
            metrics[i] = max(metric, self.thresholds[i])
            # Populate weight vector
            weights[i] = self.metrics_weight[name]

        # Scaling offset, if required
        if self.scaling == 'standardization' or self.scaling == 'IQR_normalization':
            metrics += self.offset

        cost = (metrics * weights).sum()
        return cost


class ObjFun_Normalization(ObjFun):

    def load_current_batch(self, batch_scores: np.ndarray, batch_escs: np.ndarray, target_metric: str):
        """

        :param queries_batch:
            pool of queries
        :param target_metric:
            name of the metric to minimize

        """
        self.batch_scores = batch_scores
        self.batch_escs = batch_escs
        self.target_metric = target_metric

    def __call__(self, theta_vector: np.array):
        """
        Computes target metric value given theta vector
        :param theta_vector:
            vector of modifiers for country
        :return:
            sum of max(metric, threshold)
        """
        cost = self.metrics_dict[self.target_metric](batch_scores=self.batch_scores, batch_escs=self.batch_escs,
                                                     theta_vector=theta_vector)
        return cost
