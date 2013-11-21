# Copyright (c) 2013, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)
#
#Parts of this file were influenced by the Matlab GPML framework written by
#Carl Edward Rasmussen & Hannes Nickisch, however all bugs are our own.
#
#The GPML code is released under the FreeBSD License.
#Copyright (c) 2005-2013 Carl Edward Rasmussen & Hannes Nickisch. All rights reserved.
#
#The code and associated documentation is available from
#http://gaussianprocess.org/gpml/code.

import numpy as np
import scipy as sp
from likelihood import Likelihood
from ..util.linalg import mdot, jitchol, pddet, dpotrs
from functools import partial as partial_func

class Laplace(Likelihood):
    """
    Laplace Approximation

    Find the moments \hat{f} and the hessian at this point
    (using Newton-Raphson) of the unnormalised posterior

    Compute the GP variables (i.e. generate some Y^{squiggle} and
    z^{squiggle} which makes a gaussian the same as the laplace
    approximation to the posterior, but normalised

    Arguments
    ---------

    :param data: array of data the likelihood function is approximating
    :type data: NxD
    :param noise_model: likelihood function - subclass of noise_model
    :type noise_model: noise_model
    :param extra_data: additional data used by some likelihood functions
    :type extra_data: numpy array
    """
    def __init__(self,data,noise_model,normalize=False,offset=None,scale=None, extra_data=None):
        self.data = data
        self.noise_model = noise_model
        self.extra_data = extra_data

        #Inital values
        self.N, self.D = self.data.shape
        self.is_heteroscedastic = False
        self.Nparams = 0 #TODO this is not being used
        self.NORMAL_CONST = ((0.5 * self.N) * np.log(2 * np.pi))

        self.restart()
        #likelihood.__init__(self)
        super(Laplace, self).__init__(data,noise_model,normalize,offset,scale)
        self.is_heteroscedastic = False

    def restart(self):
        """
        Reset likelihood variables to their defaults
        """
        #Initial values for the GP variables
        self.Y = np.zeros((self.N, 1))
        self.covariance_matrix = np.eye(self.N)
        self.precision = np.ones(self.N)[:, None]
        self.Z = 0
        self.YYT = None

        self.old_Ki_f = None
#Remove from here    
#    def predictive_values(self, mu, var, full_cov,**noise_args):
#        if full_cov:
#            raise NotImplementedError("Cannot make correlated predictions\
#                    with an Laplace likelihood")
#        return self.noise_model.predictive_values(mu, var)
#
#    def log_predictive_density(self, y_test, mu_star, var_star):
#        #TODO remove, this is defined in the upper class
#        """
#        Calculation of the log predictive density
#
#        .. math:
#            p(y_{*}|D) = p(y_{*}|f_{*})p(f_{*}|\mu_{*}\\sigma^{2}_{*})
#
#        :param y_test: test observations (y_{*})
#        :type y_test: (Nx1) array
#        :param mu_star: predictive mean of gaussian p(f_{*}|mu_{*}, var_{*})
#        :type mu_star: (Nx1) array
#        :param var_star: predictive variance of gaussian p(f_{*}|mu_{*}, var_{*})
#        :type var_star: (Nx1) array
#        """
#        return self.noise_model.log_predictive_density(y_test, mu_star, var_star)
#
#    def _get_params(self):
#        return np.asarray(self.noise_model._get_params())
#
#    def _get_param_names(self):
#        return self.noise_model._get_param_names()
#
#    def _set_params(self, p):
#        return self.noise_model._set_params(p)
#Remove up to this point

    def _shared_gradients_components(self):
        d3lik_d3fhat = self.noise_model.d3logpdf_df3(self.f_hat, self.data, extra_data=self.extra_data)
        dL_dfhat = 0.5*(np.diag(self.Ki_W_i)[:, None]*d3lik_d3fhat).T #why isn't this -0.5?
        I_KW_i = np.eye(self.N) - np.dot(self.K, self.Wi_K_i)
        return dL_dfhat, I_KW_i

    def _Kgradients(self):
        """
        Gradients with respect to prior kernel parameters dL_dK to be chained
        with dK_dthetaK to give dL_dthetaK
        :returns: dL_dK matrix
        :rtype: Matrix (1 x num_kernel_params)
        """
        dL_dfhat, I_KW_i = self._shared_gradients_components()
        dlp = self.noise_model.dlogpdf_df(self.f_hat, self.data, extra_data=self.extra_data)

        #Explicit
        #expl_a = np.dot(self.Ki_f, self.Ki_f.T)
        #expl_b = self.Wi_K_i
        #expl = 0.5*expl_a - 0.5*expl_b
        #dL_dthetaK_exp = dK_dthetaK(expl, X)

        #Implicit
        impl = mdot(dlp, dL_dfhat, I_KW_i)

        #No longer required as we are computing these in the gp already
        #otherwise we would take them away and add them back
        #dL_dthetaK_imp = dK_dthetaK(impl, X)
        #dL_dthetaK = dL_dthetaK_exp + dL_dthetaK_imp
        #dL_dK = expl + impl

        #No need to compute explicit as we are computing dZ_dK to account
        #for the difference between the K gradients of a normal GP,
        #and the K gradients including the implicit part
        dL_dK = impl
        return dL_dK

    def _gradients(self, partial):
        """
        Gradients with respect to likelihood parameters (dL_dthetaL)

        :param partial: Not needed by this likelihood
        :type partial: lambda function
        :rtype: array of derivatives (1 x num_likelihood_params)
        """
        dL_dfhat, I_KW_i = self._shared_gradients_components()
        dlik_dthetaL, dlik_grad_dthetaL, dlik_hess_dthetaL = self.noise_model._laplace_gradients(self.f_hat, self.data, extra_data=self.extra_data)

        #len(dlik_dthetaL)
        num_params = len(self._get_param_names())
        # make space for one derivative for each likelihood parameter
        dL_dthetaL = np.zeros(num_params)
        for thetaL_i in range(num_params):
            #Explicit
            dL_dthetaL_exp = ( np.sum(dlik_dthetaL[:, thetaL_i])
                             #- 0.5*np.trace(mdot(self.Ki_W_i, (self.K, np.diagflat(dlik_hess_dthetaL[thetaL_i]))))
                             + np.dot(0.5*np.diag(self.Ki_W_i)[:,None].T, dlik_hess_dthetaL[:, thetaL_i])
                             )

            #Implicit
            dfhat_dthetaL = mdot(I_KW_i, self.K, dlik_grad_dthetaL[:, thetaL_i])
            dL_dthetaL_imp = np.dot(dL_dfhat, dfhat_dthetaL)
            dL_dthetaL[thetaL_i] = dL_dthetaL_exp + dL_dthetaL_imp

        return dL_dthetaL

    def _compute_GP_variables(self):
        """
        Generate data Y which would give the normal distribution identical
        to the laplace approximation to the posterior, but normalised

        GPy expects a likelihood to be gaussian, so need to caluclate
        the data Y^{\tilde} that makes the posterior match that found
        by a laplace approximation to a non-gaussian likelihood but with
        a gaussian likelihood

        Firstly,
        The hessian of the unormalised posterior distribution is (K^{-1} + W)^{-1},
        i.e. z*N(f|f^{\hat}, (K^{-1} + W)^{-1}) but this assumes a non-gaussian likelihood,
        we wish to find the hessian \Sigma^{\tilde}
        that has the same curvature but using our new simulated data Y^{\tilde}
        i.e. we do N(Y^{\tilde}|f^{\hat}, \Sigma^{\tilde})N(f|0, K) = z*N(f|f^{\hat}, (K^{-1} + W)^{-1})
        and we wish to find what Y^{\tilde} and \Sigma^{\tilde}
        We find that Y^{\tilde} = W^{-1}(K^{-1} + W)f^{\hat} and \Sigma^{tilde} = W^{-1}

        Secondly,
        GPy optimizes the log marginal log p(y) = -0.5*ln|K+\Sigma^{\tilde}| - 0.5*Y^{\tilde}^{T}(K^{-1} + \Sigma^{tilde})^{-1}Y + lik.Z
        So we can suck up any differences between that and our log marginal likelihood approximation
        p^{\squiggle}(y) = -0.5*f^{\hat}K^{-1}f^{\hat} + log p(y|f^{\hat}) - 0.5*log |K||K^{-1} + W|
        which we want to optimize instead, by equating them and rearranging, the difference is added onto
        the log p(y) that GPy optimizes by default

        Thirdly,
        Since we have gradients that depend on how we move f^{\hat}, we have implicit components
        aswell as the explicit dL_dK, we hold these differences in dZ_dK and add them to dL_dK in the
        gp.py code
        """
        Wi = 1.0/self.W
        self.Sigma_tilde = np.diagflat(Wi)

        Y_tilde = Wi*self.Ki_f + self.f_hat

        self.Wi_K_i = self.W12BiW12
        self.ln_det_Wi_K = pddet(self.Sigma_tilde + self.K)
        self.lik = self.noise_model.logpdf(self.f_hat, self.data, extra_data=self.extra_data)
        self.y_Wi_Ki_i_y = mdot(Y_tilde.T, self.Wi_K_i, Y_tilde)

        Z_tilde = (+ self.lik
                   - 0.5*self.ln_B_det
                   + 0.5*self.ln_det_Wi_K
                   - 0.5*self.f_Ki_f
                   + 0.5*self.y_Wi_Ki_i_y
                  )

        #Convert to float as its (1, 1) and Z must be a scalar
        self.Z = np.float64(Z_tilde)
        self.Y = Y_tilde
        self.YYT = np.dot(self.Y, self.Y.T)
        self.covariance_matrix = self.Sigma_tilde
        self.precision = 1.0 / np.diag(self.covariance_matrix)[:, None]

        #Compute dZ_dK which is how the approximated distributions gradients differ from the dL_dK computed for other likelihoods
        self.dZ_dK = self._Kgradients()
        #+ 0.5*self.Wi_K_i - 0.5*np.dot(self.Ki_f, self.Ki_f.T) #since we are not adding the K gradients explicit part theres no need to compute this again

    def fit_full(self, K):
        """
        The laplace approximation algorithm, find K and expand hessian
        For nomenclature see Rasmussen & Williams 2006 - modified for numerical stability

        :param K: Prior covariance matrix evaluated at locations X
        :type K: NxN matrix
        """
        self.K = K.copy()

        #Find mode
        self.f_hat = self.rasm_mode(self.K)

        #Compute hessian and other variables at mode
        self._compute_likelihood_variables()

        #Compute fake variables replicating laplace approximation to posterior
        self._compute_GP_variables()

    def _compute_likelihood_variables(self):
        """
        Compute the variables required to compute gaussian Y variables
        """
        #At this point get the hessian matrix (or vector as W is diagonal)
        self.W = -self.noise_model.d2logpdf_df2(self.f_hat, self.data, extra_data=self.extra_data)

        #TODO: Could save on computation when using rasm by returning these, means it isn't just a "mode finder" though
        self.W12BiW12, self.ln_B_det = self._compute_B_statistics(self.K, self.W, np.eye(self.N))

        self.Ki_f = self.Ki_f
        self.f_Ki_f = np.dot(self.f_hat.T, self.Ki_f)
        self.Ki_W_i = self.K - mdot(self.K, self.W12BiW12, self.K)

    def _compute_B_statistics(self, K, W, a):
        """
        Rasmussen suggests the use of a numerically stable positive definite matrix B
        Which has a positive diagonal element and can be easyily inverted

        :param K: Prior Covariance matrix evaluated at locations X
        :type K: NxN matrix
        :param W: Negative hessian at a point (diagonal matrix)
        :type W: Vector of diagonal values of hessian (1xN)
        :param a: Matrix to calculate W12BiW12a
        :type a: Matrix NxN
        :returns: (W12BiW12, ln_B_det)
        """
        if not self.noise_model.log_concave:
            #print "Under 1e-10: {}".format(np.sum(W < 1e-10))
            W[W < 1e-6] = 1e-6  # FIXME-HACK: This is a hack since GPy can't handle negative variances which can occur
                                # If the likelihood is non-log-concave. We wan't to say that there is a negative variance
                                # To cause the posterior to become less certain than the prior and likelihood,
                                # This is a property only held by non-log-concave likelihoods


        #W is diagonal so its sqrt is just the sqrt of the diagonal elements
        W_12 = np.sqrt(W)
        B = np.eye(self.N) + W_12*K*W_12.T
        L = jitchol(B)

        W12BiW12 = W_12*dpotrs(L, np.asfortranarray(W_12*a), lower=1)[0]
        ln_B_det = 2*np.sum(np.log(np.diag(L)))
        return W12BiW12, ln_B_det

    def rasm_mode(self, K, MAX_ITER=30):
        """
        Rasmussen's numerically stable mode finding
        For nomenclature see Rasmussen & Williams 2006
        Influenced by GPML (BSD) code, all errors are our own

        :param K: Covariance matrix evaluated at locations X
        :type K: NxD matrix
        :param MAX_ITER: Maximum number of iterations of newton-raphson before forcing finish of optimisation
        :type MAX_ITER: scalar
        :returns: f_hat, mode on which to make laplace approxmiation
        :rtype: NxD matrix
        """
        #old_Ki_f = np.zeros((self.N, 1))

        #Start f's at zero originally
        if self.old_Ki_f is None:
            old_Ki_f = np.zeros((self.N, 1))
            f = np.dot(K, old_Ki_f)
        else:
            #Start at the old best point
            old_Ki_f = self.old_Ki_f.copy()
            f = self.f_hat.copy()

        new_obj = -np.inf
        old_obj = np.inf

        def obj(Ki_f, f):
            return -0.5*np.dot(Ki_f.T, f) + self.noise_model.logpdf(f, self.data, extra_data=self.extra_data)

        difference = np.inf
        epsilon = 1e-5
        #step_size = 1
        #rs = 0
        i = 0

        while difference > epsilon and i < MAX_ITER:
            W = -self.noise_model.d2logpdf_df2(f, self.data, extra_data=self.extra_data)

            W_f = W*f
            grad = self.noise_model.dlogpdf_df(f, self.data, extra_data=self.extra_data)

            b = W_f + grad
            W12BiW12Kb, _ = self._compute_B_statistics(K, W.copy(), np.dot(K, b))

            #Work out the DIRECTION that we want to move in, but don't choose the stepsize yet
            full_step_Ki_f = b - W12BiW12Kb
            dKi_f = full_step_Ki_f - old_Ki_f

            f_old = f.copy()
            def inner_obj(step_size, old_Ki_f, dKi_f, K):
                Ki_f = old_Ki_f + step_size*dKi_f
                f = np.dot(K, Ki_f)
                # This is nasty, need to set something within an optimization though
                self.tmp_Ki_f = Ki_f.copy()
                self.tmp_f = f.copy()
                return -obj(Ki_f, f)

            i_o = partial_func(inner_obj, old_Ki_f=old_Ki_f, dKi_f=dKi_f, K=K)
            #Find the stepsize that minimizes the objective function using a brent line search
            #The tolerance and maxiter matter for speed! Seems to be best to keep them low and make more full
            #steps than get this exact then make a step, if B was bigger it might be the other way around though
            new_obj = sp.optimize.minimize_scalar(i_o, method='brent', tol=1e-4, options={'maxiter':5}).fun
            f = self.tmp_f.copy()
            Ki_f = self.tmp_Ki_f.copy()

            #Optimize without linesearch
            #f_old = f.copy()
            #update_passed = False
            #while not update_passed:
                #Ki_f = old_Ki_f + step_size*dKi_f
                #f = np.dot(K, Ki_f)

                #old_obj = new_obj
                #new_obj = obj(Ki_f, f)
                #difference = new_obj - old_obj
                ##print "difference: ",difference
                #if difference < 0:
                    ##print "Objective function rose", np.float(difference)
                    ##If the objective function isn't rising, restart optimization
                    #step_size *= 0.8
                    ##print "Reducing step-size to {ss:.3} and restarting optimization".format(ss=step_size)
                    ##objective function isn't increasing, try reducing step size
                    #f = f_old.copy() #it's actually faster not to go back to old location and just zigzag across the mode
                    #old_obj = new_obj
                    #rs += 1
                #else:
                    #update_passed = True

            #old_Ki_f = self.Ki_f.copy()

            #difference = abs(new_obj - old_obj)
            #old_obj = new_obj.copy()
            #difference = np.abs(np.sum(f - f_old))
            difference = np.abs(np.sum(Ki_f - old_Ki_f))
            old_Ki_f = Ki_f.copy()
            i += 1

        self.old_Ki_f = old_Ki_f.copy()
        if difference > epsilon:
            print "Not perfect f_hat fit difference: {}".format(difference)

        self.Ki_f = Ki_f
        return f
