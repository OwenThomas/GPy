import numpy as np
import copy
from ..core.parameterized import Parameterized

class Likelihood(Parameterized):
    """
    The atom for a likelihood class

    This object interfaces the GP and the data.
    The most basic likelihood (Gaussian) inherits directly from this,
    as does the EP and Laplace approximations.

    Some things must be defined for this to work properly:

        - self.Y : the effective Gaussian target of the GP
        - self.N, self.D : Y.shape
        - self.covariance_matrix : the effective (noise) covariance of the GP targets
        - self.Z : a factor which gets added to the likelihood (0 for a Gaussian, Z_EP for EP)
        - self.is_heteroscedastic : enables significant computational savings in GP
        - self.precision : a scalar or vector representation of the effective target precision
        - self.YYT : (optional) = np.dot(self.Y, self.Y.T) enables computational savings for D>N
        - self.V : self.precision * self.Y

    """
    def __init__(self,data,noise_model,normalize,offset=None,scale=None):
        Parameterized.__init__(self)

        self.noise_model = noise_model
        self.data = data
        self.num_data, self.output_dim = self.data.shape
        self.num_params = self._get_params().size
        self.Z = 0. # a correction factor which accounts for the approximation made
        self.dZ_dK = 0

        #Data transformation (aka normalization)
        if normalize:
            self._offset = data.mean(0)[None, :] if offset == None else offset
            self._scale = data.std(0)[None, :] if scale == None else scale
            # Don't scale outputs which have zero variance to zero.
            for pj in np.nonzero(self._scale == 0.)[0]:
                self._scale[pj] = 1.0e-3
            #self._scale[np.nonzero(self._scale == 0.)] = 1.0e-3
        else:
            self._offset = np.zeros((1, self.output_dim))
            self._scale = np.ones((1, self.output_dim))

        self.check_data()

    def check_data(self):
        """
        Checks if data values are appropiate for the noise model
        """
        self.noise_model.check_data(self.data)

    def _get_params(self):
        return np.asarray(self.noise_model._get_params())

    def _get_param_names(self):
        return self.noise_model._get_param_names()

    def _set_params(self, p):
        return self.noise_model._set_params(p)

    def fit_full(self, K):
        """
        No approximations needed by default
        """
        pass

    def restart(self):
        """
        No need to restart if not an approximation
        """
        pass

    def _gradients(self, partial):
        return self.noise_model._gradients(partial)

    def predictive_values(self, mu, var, full_cov, **noise_args):

        mean,_var,q1,q3 = self.noise_model.predictive_values(mu,var,full_cov,**noise_args)

        #Un-normalize data
        mean = mean * self._scale + self._offset
        q1 = q1 * self._scale + self._offset
        q3 = q3 * self._scale + self._offset
        var = var * self._scale**2
        return mean,var,q1,q3

    def log_predictive_density(self, y_test, mu_star, var_star):
        """
        Calculation of the predictive density

        .. math:
            p(y_{*}|D) = p(y_{*}|f_{*})p(f_{*}|\mu_{*}\\sigma^{2}_{*})

        :param y_test: test observations (y_{*})
        :type y_test: (Nx1) array
        :param mu_star: predictive mean of gaussian p(f_{*}|mu_{*}, var_{*})
        :type mu_star: (Nx1) array
        :param var_star: predictive variance of gaussian p(f_{*}|mu_{*}, var_{*})
        :type var_star: (Nx1) array
        """
        y_rescaled = (y_test - self._offset)/self._scale
        return self.noise_model.log_predictive_density(y_rescaled,mu_star,var_star)
