import numpy as np
from likelihood import Likelihood
from ..util.linalg import jitchol


class Exact(Likelihood):
    """
    Likelihood class for handling Gaussian noise without any approximation

    :param data: observed output
    :type data: num_datax1 numpy.darray
    :param variance: noise parameter
    :param normalize:  whether to normalize the data before computing (predictions will be in original scales)
    :type normalize: False|True
    """
    def __init__(self, data, noise_model,normalize,offset=None,scale=None):#, variance=1., normalize=False):

        super(Exact, self).__init__(data,noise_model,normalize,offset,scale)
        #TODO we shoud standardize the use of N - num_data; D - num_outputs; N_params - num_params
        self.set_data(data)
        self.is_heteroscedastic = False

        self._variance = np.asarray(self.noise_model.variance) + 1.
        #self._set_params(np.asarray(variance))
        self._set_params(np.asarray(self.noise_model.variance))

    def set_data(self, data):
        self.data = data
        self.num_data, output_dim = data.shape
        assert output_dim == self.output_dim
        self.Y = (self.data - self._offset) / self._scale
        if output_dim > self.num_data:
            self.YYT = np.dot(self.Y, self.Y.T)
            self.trYYT = np.trace(self.YYT)
            self.YYT_factor = jitchol(self.YYT)
        else:
            self.YYT = None
            self.trYYT = np.sum(np.square(self.Y))
            self.YYT_factor = self.Y

    def _set_params(self, x):
        x = np.float64(x)
        self.noise_model._set_params(x)
        if np.all(self._variance != x):
            if x == 0.:#special case of zero noise
                self.precision = np.inf
                self.V = None
            else:
                self.precision = 1. / x
                self.V = (self.precision) * self.Y
                self.VVT_factor = self.precision * self.YYT_factor
            self.covariance_matrix = np.eye(self.num_data) * x
            self._variance = x

    def predictive_values(self, mu, var, full_cov, **likelihood_args):
        """
        Un-normalize the prediction and add the likelihood variance, then return the 5%, 95% interval
        """
        mean = mu * self._scale + self._offset
        if full_cov:
            if self.output_dim > 1:
                raise NotImplementedError, "TODO"
                # Note. for output_dim>1, we need to re-normalise all the outputs independently.
                # This will mess up computations of diag(true_var), below.
                # note that the upper, lower quantiles should be the same shape as mean
            # Augment the output variance with the likelihood variance and rescale.
            true_var = (var + np.eye(var.shape[0]) * self._variance) * self._scale ** 2
            _5pc = mean - 2.*np.sqrt(np.diag(true_var))
            _95pc = mean + 2.*np.sqrt(np.diag(true_var))
        else:
            true_var = (var + self._variance) * self._scale ** 2
            _5pc = mean - 2.*np.sqrt(true_var)
            _95pc = mean + 2.*np.sqrt(true_var)
        return mean, true_var, _5pc, _95pc
