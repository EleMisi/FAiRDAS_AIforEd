import numpy as np


def impact_function(scores: np.ndarray, thetas: np.ndarray):
    """
    Compute impact function given the resource and pool of queries
    :param scores:
        pool of  score vectors
    :param thetas:
        vector of modifier for each resource
    :return:
        impact function value
    """
    # Compute actual score
    scores = (scores * (1 - thetas)).mean(0)

    return scores


class Metric:

    def __init__(self):

        self.scaling = 'None'
        self.offset = 10  # Offset to avoid non-positive numbers in standardization

    def __call__(self, batch_scores: np.ndarray, batch_escs: np.ndarray, theta_vector: np.ndarray):
        """
        Computes metric value given batch of queries and theta vector
        :param batch_scores:
             scores of current batch
        :param batch_escs:
             escs of current batch
        :param theta_vector:
            vector of modifiers for country
        """
        NotImplementedError()

    def scale(self, value):

        if self.scaling == 'normalization':
            value = (value - self.min) / (self.max - self.min)
        elif self.scaling == 'standardization':
            value = ((value - self.mean) / self.std)
        elif self.scaling == 'IQR_normalization':
            value = (value - self.q1) / (self.q3 - self.q1)
        elif self.scaling == 'min_std':
            value = (value - self.min) / self.std
        elif self.scaling == 'None':
            return value

        return value


class DIDI(Metric):
    """
    DIDI metric computation for students task
    """

    def __init__(self, impact_function):
        """
        Build DIDI object
        :param impact_function:
            function to compute impact of resource in the ranking
        """

        super().__init__()
        self.impact_function = impact_function

    def __call__(self, batch_scores: np.ndarray, batch_escs: np.ndarray, theta_vector: np.ndarray):
        """Computes the Disparate Impact Discrimination Index for Regression Tasks given the impact function output
        :param batch_scores:
             scores of current batch
        :param batch_escs:
             escs of current batch
        :param theta_vector:
            vector of modifiers for sensitive attribute
        :return:
            The (absolute) value of the DIDI.
        """

        # Define indicator matrix for pool of queries
        indicator_matrix = self.get_indicator_matrix(batch_escs)

        # Define vector of theta based on sensitive attribute
        thetas = np.zeros(len(batch_escs))
        for k in range(len(batch_escs)):
            # Find resource country
            country = np.argmax(batch_escs[k])
            # Select country multiplier
            thetas[k] = theta_vector[country]

        # Compute impact function which is coincident to the modified scores
        output = (batch_scores.ravel() * (1 - thetas))

        # Check indicator matrix shape
        assert indicator_matrix.shape[1] == output.shape[
            0], f"Wrong number of samples, expected {indicator_matrix.shape[1]} got {output.shape[0]}"
        # Compute DIDI
        didi = self.compute_DIDI(output=output, indicator_matrix=indicator_matrix)

        didi = self.scale(didi)

        return didi

    def compute_DIDI(self, output: np.array, indicator_matrix: np.array) -> float:
        """Computes the Disparate Impact Discrimination Index for Regression Tasks given the impact function output
        :param output:
            array with impact function values (ordered by reources idx)
        :return:
            The (absolute) value of the DIDI.
        """
        # Check indicator matrix shape
        assert indicator_matrix.shape[1] == output.shape[
            0], f"Wrong number of samples, expected {indicator_matrix.shape[1]} got {output.shape[0]}"

        # Compute DIDI
        didi = 0.0
        total_average = np.mean(output)
        # Loop over protected groups
        for protected_group in indicator_matrix:
            # Select output of sample belonging to protected attribute
            protected_targets = output[protected_group]
            # Compute partial DIDI over the protected attribute
            if len(protected_targets) > 0:
                protected_average = np.mean(protected_targets)
                didi += abs(protected_average - total_average)
        return didi

    def get_indicator_matrix(self, batch_escs_attribute: np.ndarray) -> np.array:
        """Computes the indicator matrix given the input data and a protected feature.
        :param batch_escs_attribute:
            vector of queries sensitive attribute value
        :return:
            indicator matrix, i.e., a matrix in which the i-th row represents a boolean vector stating whether or
            not the j-th sample (represented by the j-th column) is part of the i-th protected group.
        """
        n_samples = batch_escs_attribute.shape[0]
        protected_labels = range(batch_escs_attribute.shape[1])
        n_groups = len(protected_labels)
        matrix = np.zeros((n_samples, n_groups)).astype(int)
        for i in range(n_groups):
            for j in range(n_samples):
                label = protected_labels[i]
                matrix[j, i] = 1 if batch_escs_attribute[j, label] == 1. else 0
        return matrix.transpose().astype(bool)


class ScoreAbsoluteError(Metric):

    def __call__(self, batch_scores: np.ndarray, batch_escs: np.ndarray, theta_vector: np.ndarray, debug=None):
        """
        Compute distance between true scores and modified scores
        :param batch_scores:
             scores of current batch
        :param batch_escs:
             escs of current batch
        :param theta_vector:
            vector of modifiers for country

        """

        # Define vector of theta based on sensitive attribute
        thetas = np.zeros(len(batch_escs))
        for k in range(len(batch_escs)):
            # Find resource country
            country = np.argmax(batch_escs[k])
            # Select country multiplier
            thetas[k] = theta_vector[country]

        # Compute actual distance
        distance = (batch_scores * thetas).mean()

        distance = self.scale(distance)

        return distance

