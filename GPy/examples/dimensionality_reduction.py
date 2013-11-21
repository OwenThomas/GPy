# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from matplotlib import pyplot as plt, cm

import GPy
from GPy.core.transformations import logexp
from GPy.likelihoods.gaussian import Gaussian
from GPy.models import BayesianGPLVM

default_seed = np.random.seed(123344)

def BGPLVM(seed=default_seed):
    N = 13
    num_inducing = 5
    Q = 6
    D = 25
    # generate GPLVM-like data
    X = np.random.rand(N, Q)
    lengthscales = np.random.rand(Q)
    k = (GPy.kern.rbf(Q, .5, lengthscales, ARD=True)
         + GPy.kern.white(Q, 0.01))
    K = k.K(X)
    Y = np.random.multivariate_normal(np.zeros(N), K, D).T
    #lik = Gaussian(Y, normalize=True)
    lik = GPy.likelihoods.likelihood_constructors._gaussian(Y, normalize=True)

    # k = GPy.kern.rbf_inv(Q, .5, np.ones(Q) * 2., ARD=True) + GPy.kern.bias(Q) + GPy.kern.white(Q)
    # k = GPy.kern.linear(Q) + GPy.kern.bias(Q) + GPy.kern.white(Q, 0.00001)
    # k = GPy.kern.rbf(Q, ARD = False)  + GPy.kern.white(Q, 0.00001)
    # k = GPy.kern.rbf(Q, .5, np.ones(Q) * 2., ARD=True) + GPy.kern.rbf(Q, .3, np.ones(Q) * .2, ARD=True)
    k = GPy.kern.rbf(Q, .5, np.ones(Q) * 2., ARD=True) + GPy.kern.linear(Q, np.ones(Q) * .2, ARD=True)
    # k = GPy.kern.rbf(Q, .5, 2., ARD=0) + GPy.kern.rbf(Q, .3, .2, ARD=0)

    m = GPy.models.BayesianGPLVM(lik, Q, kernel=k, num_inducing=num_inducing)
    m.lengthscales = lengthscales
    # m.constrain_positive('(rbf|bias|noise|white|S)')
    # m.constrain_fixed('S', 1)

    # pb.figure()
    # m.plot()
    # pb.title('PCA initialisation')
    # pb.figure()
    # m.optimize(messages = 1)
    # m.plot()
    # pb.title('After optimisation')
    # m.randomize()
    # m.checkgrad(verbose=1)

    return m

def GPLVM_oil_100(optimize=True):
    data = GPy.util.datasets.oil_100()
    Y = data['X']

    # create simple GP model
    kernel = GPy.kern.rbf(6, ARD=True) + GPy.kern.bias(6)
    m = GPy.models.GPLVM(Y, 6, kernel=kernel)
    m.data_labels = data['Y'].argmax(axis=1)

    # optimize
    if optimize:
        m.optimize('scg', messages=1)

    # plot
    print(m)
    m.plot_latent(labels=m.data_labels)
    return m

def sparseGPLVM_oil(optimize=True, N=100, Q=6, num_inducing=15, max_iters=50):
    np.random.seed(0)
    data = GPy.util.datasets.oil()

    Y = data['X'][:N]
    Y = Y - Y.mean(0)
    Y /= Y.std(0)

    # create simple GP model
    kernel = GPy.kern.rbf(Q, ARD=True) + GPy.kern.bias(Q)
    m = GPy.models.SparseGPLVM(Y, Q, kernel=kernel, num_inducing=num_inducing)
    m.data_labels = data['Y'].argmax(axis=1)

    # optimize
    if optimize:
        m.optimize('scg', messages=1, max_iters=max_iters)

    # plot
    print(m)
    # m.plot_latent(labels=m.data_labels)
    return m

def swiss_roll(optimize=True, num_samples=1000, num_inducing=15, Q=4, sigma=.2, plot=False):
    from GPy.util.datasets import swiss_roll_generated
    from GPy.core.transformations import logexp_clipped

    data = swiss_roll_generated(num_samples=num_samples, sigma=sigma)
    Y = data['Y']
    Y -= Y.mean()
    Y /= Y.std()

    t = data['t']
    c = data['colors']

    try:
        from sklearn.manifold.isomap import Isomap
        iso = Isomap().fit(Y)
        X = iso.embedding_
        if Q > 2:
            X = np.hstack((X, np.random.randn(num_samples, Q - 2)))
    except ImportError:
        X = np.random.randn(num_samples, Q)

    if plot:
        from mpl_toolkits import mplot3d
        import pylab
        fig = pylab.figure("Swiss Roll Data")
        ax = fig.add_subplot(121, projection='3d')
        ax.scatter(*Y.T, c=c)
        ax.set_title("Swiss Roll")

        ax = fig.add_subplot(122)
        ax.scatter(*X.T[:2], c=c)
        ax.set_title("Initialization")


    var = .5
    S = (var * np.ones_like(X) + np.clip(np.random.randn(num_samples, Q) * var ** 2,
                                         - (1 - var),
                                         (1 - var))) + .001
    Z = np.random.permutation(X)[:num_inducing]

    kernel = GPy.kern.rbf(Q, ARD=True) + GPy.kern.bias(Q, np.exp(-2)) + GPy.kern.white(Q, np.exp(-2))

    m = BayesianGPLVM(Y, Q, X=X, X_variance=S, num_inducing=num_inducing, Z=Z, kernel=kernel)
    m.data_colors = c
    m.data_t = t

    m['rbf_lengthscale'] = 1. # X.var(0).max() / X.var(0)
    m['noise_variance'] = Y.var() / 100.
    m['bias_variance'] = 0.05

    if optimize:
        m.optimize('scg', messages=1)
    return m

def BGPLVM_oil(optimize=True, N=200, Q=7, num_inducing=40, max_iters=1000, plot=False, **k):
    np.random.seed(0)
    data = GPy.util.datasets.oil()

    # create simple GP model
    kernel = GPy.kern.rbf_inv(Q, 1., [.1] * Q, ARD=True) + GPy.kern.bias(Q, np.exp(-2))

    Y = data['X'][:N]
    #Yn = Gaussian(Y, normalize=True)
    Yn = GPy.likelihoods.likelihood_constructors._gaussian(Y,normalize=True)
#     Yn = Y - Y.mean(0)
#     Yn /= Yn.std(0)

    m = GPy.models.BayesianGPLVM(Yn, Q, kernel=kernel, num_inducing=num_inducing, **k)
    m.data_labels = data['Y'][:N].argmax(axis=1)

    # m.constrain('variance|leng', logexp_clipped())
    # m['.*lengt'] = m.X.var(0).max() / m.X.var(0)
    m['noise'] = Yn.Y.var() / 100.


    # optimize
    if optimize:
        m.constrain_fixed('noise')
        m.optimize('scg', messages=1, max_iters=200, gtol=.05)
        m.constrain_positive('noise')
        m.constrain_bounded('white', 1e-7, 1)
        m.optimize('scg', messages=1, max_iters=max_iters, gtol=.05)

    if plot:
        y = m.likelihood.Y[0, :]
        fig, (latent_axes, sense_axes) = plt.subplots(1, 2)
        plt.sca(latent_axes)
        m.plot_latent()
        data_show = GPy.util.visualize.vector_show(y)
        lvm_visualizer = GPy.util.visualize.lvm_dimselect(m.X[0, :], m, data_show, latent_axes=latent_axes) # , sense_axes=sense_axes)
        raw_input('Press enter to finish')
        plt.close(fig)
    return m

def oil_100():
    data = GPy.util.datasets.oil_100()
    m = GPy.models.GPLVM(data['X'], 2)

    # optimize
    m.optimize(messages=1, max_iters=2)

    # plot
    print(m)
    # m.plot_latent(labels=data['Y'].argmax(axis=1))
    return m



def _simulate_sincos(D1, D2, D3, N, num_inducing, Q, plot_sim=False):
    x = np.linspace(0, 4 * np.pi, N)[:, None]
    s1 = np.vectorize(lambda x: np.sin(x))
    s2 = np.vectorize(lambda x: np.cos(x))
    s3 = np.vectorize(lambda x:-np.exp(-np.cos(2 * x)))
    sS = np.vectorize(lambda x: np.sin(2 * x))

    s1 = s1(x)
    s2 = s2(x)
    s3 = s3(x)
    sS = sS(x)

    S1 = np.hstack([s1, sS])
    S2 = np.hstack([s2, s3, sS])
    S3 = np.hstack([s3, sS])

    Y1 = S1.dot(np.random.randn(S1.shape[1], D1))
    Y2 = S2.dot(np.random.randn(S2.shape[1], D2))
    Y3 = S3.dot(np.random.randn(S3.shape[1], D3))

    Y1 += .3 * np.random.randn(*Y1.shape)
    Y2 += .2 * np.random.randn(*Y2.shape)
    Y3 += .25 * np.random.randn(*Y3.shape)

    Y1 -= Y1.mean(0)
    Y2 -= Y2.mean(0)
    Y3 -= Y3.mean(0)
    Y1 /= Y1.std(0)
    Y2 /= Y2.std(0)
    Y3 /= Y3.std(0)

    slist = [sS, s1, s2, s3]
    slist_names = ["sS", "s1", "s2", "s3"]
    Ylist = [Y1, Y2, Y3]

    if plot_sim:
        import pylab
        import itertools
        fig = pylab.figure("MRD Simulation Data", figsize=(8, 6))
        fig.clf()
        ax = fig.add_subplot(2, 1, 1)
        labls = slist_names
        for S, lab in itertools.izip(slist, labls):
            ax.plot(S, label=lab)
        ax.legend()
        for i, Y in enumerate(Ylist):
            ax = fig.add_subplot(2, len(Ylist), len(Ylist) + 1 + i)
            ax.imshow(Y, aspect='auto', cmap=cm.gray) # @UndefinedVariable
            ax.set_title("Y{}".format(i + 1))
        pylab.draw()
        pylab.tight_layout()

    return slist, [S1, S2, S3], Ylist

def bgplvm_simulation_matlab_compare():
    from GPy.util.datasets import simulation_BGPLVM
    sim_data = simulation_BGPLVM()
    Y = sim_data['Y']
    S = sim_data['S']
    mu = sim_data['mu']
    num_inducing, [_, Q] = 3, mu.shape

    from GPy.models import mrd
    from GPy import kern
    reload(mrd); reload(kern)
    k = kern.linear(Q, ARD=True) + kern.bias(Q, np.exp(-2)) + kern.white(Q, np.exp(-2))
    m = BayesianGPLVM(Y, Q, init="PCA", num_inducing=num_inducing, kernel=k,
#                        X=mu,
#                        X_variance=S,
                       _debug=False)
    m.auto_scale_factor = True
    m['noise'] = Y.var() / 100.
    m['linear_variance'] = .01
    return m

def bgplvm_simulation(optimize='scg',
                      plot=True,
                      max_iters=2e4,
                      plot_sim=False):
#     from GPy.core.transformations import logexp_clipped
    D1, D2, D3, N, num_inducing, Q = 15, 5, 8, 30, 3, 10
    slist, Slist, Ylist = _simulate_sincos(D1, D2, D3, N, num_inducing, Q, plot_sim)

    from GPy.models import mrd
    from GPy import kern
    reload(mrd); reload(kern)

    Y = Ylist[0]

    k = kern.linear(Q, ARD=True) + kern.bias(Q, np.exp(-2)) + kern.white(Q, np.exp(-2)) # + kern.bias(Q)
    m = BayesianGPLVM(Y, Q, init="PCA", num_inducing=num_inducing, kernel=k)

    # m.constrain('variance|noise', logexp_clipped())
    m['noise'] = Y.var() / 100.

    if optimize:
        print "Optimizing model:"
        m.optimize(optimize, max_iters=max_iters,
                   messages=True, gtol=.05)
    if plot:
        m.plot_X_1d("BGPLVM Latent Space 1D")
        m.kern.plot_ARD('BGPLVM Simulation ARD Parameters')
    return m

def mrd_simulation(optimize=True, plot=True, plot_sim=True, **kw):
    D1, D2, D3, N, num_inducing, Q = 60, 20, 36, 60, 6, 5
    slist, Slist, Ylist = _simulate_sincos(D1, D2, D3, N, num_inducing, Q, plot_sim)

    #likelihood_list = [Gaussian(x, normalize=True) for x in Ylist]
    likelihood_list = [GPy.likelihoods.likelihood_constructors._gaussian(x, normalize=True) for x in Ylist]

    from GPy.models import mrd
    from GPy import kern

    reload(mrd); reload(kern)

    k = kern.linear(Q, ARD=True) + kern.bias(Q, np.exp(-2)) + kern.white(Q, np.exp(-2))
    m = mrd.MRD(likelihood_list, input_dim=Q, num_inducing=num_inducing, kernels=k, initx="", initz='permute', **kw)
    m.ensure_default_constraints()

    for i, bgplvm in enumerate(m.bgplvms):
        m['{}_noise'.format(i)] = bgplvm.likelihood.Y.var() / 500.


    # DEBUG
    # np.seterr("raise")

    if optimize:
        print "Optimizing Model:"
        m.optimize(messages=1, max_iters=8e3, gtol=.1)
    if plot:
        m.plot_X_1d("MRD Latent Space 1D")
        m.plot_scales("MRD Scales")
    return m

def brendan_faces():
    from GPy import kern
    data = GPy.util.datasets.brendan_faces()
    Q = 2
    Y = data['Y']
    Yn = Y - Y.mean()
    Yn /= Yn.std()

    m = GPy.models.GPLVM(Yn, Q)

    # optimize
    m.constrain('rbf|noise|white', GPy.core.transformations.logexp_clipped())

    m.optimize('scg', messages=1, max_iters=1000)

    ax = m.plot_latent(which_indices=(0, 1))
    y = m.likelihood.Y[0, :]
    data_show = GPy.util.visualize.image_show(y[None, :], dimensions=(20, 28), transpose=True, order='F', invert=False, scale=False)
    lvm_visualizer = GPy.util.visualize.lvm(m.X[0, :].copy(), m, data_show, ax)
    raw_input('Press enter to finish')

    return m

def olivetti_faces():
    from GPy import kern
    data = GPy.util.datasets.olivetti_faces()
    Q = 2
    Y = data['Y']
    Yn = Y - Y.mean()
    Yn /= Yn.std()

    m = GPy.models.GPLVM(Yn, Q)
    m.optimize('scg', messages=1, max_iters=1000)

    ax = m.plot_latent(which_indices=(0, 1))
    y = m.likelihood.Y[0, :]
    data_show = GPy.util.visualize.image_show(y[None, :], dimensions=(112, 92), transpose=False, invert=False, scale=False)
    lvm_visualizer = GPy.util.visualize.lvm(m.X[0, :].copy(), m, data_show, ax)
    raw_input('Press enter to finish')

    return m

def stick_play(range=None, frame_rate=15):

    data = GPy.util.datasets.osu_run1()
    # optimize
    if range == None:
        Y = data['Y'].copy()
    else:
        Y = data['Y'][range[0]:range[1], :].copy()
    y = Y[0, :]
    data_show = GPy.util.visualize.stick_show(y[None, :], connect=data['connect'])
    GPy.util.visualize.data_play(Y, data_show, frame_rate)
    return Y

def stick(kernel=None):
    data = GPy.util.datasets.osu_run1()
    # optimize
    m = GPy.models.GPLVM(data['Y'], 2, kernel=kernel)
    m.optimize(messages=1, max_f_eval=10000)
    if GPy.util.visualize.visual_available:
        plt.clf
        ax = m.plot_latent()
        y = m.likelihood.Y[0, :]
        data_show = GPy.util.visualize.stick_show(y[None, :], connect=data['connect'])
        lvm_visualizer = GPy.util.visualize.lvm(m.X[0, :].copy(), m, data_show, ax)
        raw_input('Press enter to finish')

    return m

def bcgplvm_linear_stick(kernel=None):
    data = GPy.util.datasets.osu_run1()
    # optimize
    mapping = GPy.mappings.Linear(data['Y'].shape[1], 2)
    m = GPy.models.BCGPLVM(data['Y'], 2, kernel=kernel, mapping=mapping)
    m.optimize(messages=1, max_f_eval=10000)
    if GPy.util.visualize.visual_available:
        plt.clf
        ax = m.plot_latent()
        y = m.likelihood.Y[0, :]
        data_show = GPy.util.visualize.stick_show(y[None, :], connect=data['connect'])
        lvm_visualizer = GPy.util.visualize.lvm(m.X[0, :].copy(), m, data_show, ax)
        raw_input('Press enter to finish')

    return m

def bcgplvm_stick(kernel=None):
    data = GPy.util.datasets.osu_run1()
    # optimize
    back_kernel=GPy.kern.rbf(data['Y'].shape[1], lengthscale=5.)
    mapping = GPy.mappings.Kernel(X=data['Y'], output_dim=2, kernel=back_kernel)
    m = GPy.models.BCGPLVM(data['Y'], 2, kernel=kernel, mapping=mapping)
    m.optimize(messages=1, max_f_eval=10000)
    if GPy.util.visualize.visual_available:
        plt.clf
        ax = m.plot_latent()
        y = m.likelihood.Y[0, :]
        data_show = GPy.util.visualize.stick_show(y[None, :], connect=data['connect'])
        lvm_visualizer = GPy.util.visualize.lvm(m.X[0, :].copy(), m, data_show, ax)
        raw_input('Press enter to finish')

    return m

def robot_wireless():
    data = GPy.util.datasets.robot_wireless()
    # optimize
    m = GPy.models.GPLVM(data['Y'], 2)
    m.optimize(messages=1, max_f_eval=10000)
    m._set_params(m._get_params())
    plt.clf
    ax = m.plot_latent()

    return m

def stick_bgplvm(model=None):
    data = GPy.util.datasets.osu_run1()
    Q = 6
    kernel = GPy.kern.rbf(Q, ARD=True) + GPy.kern.bias(Q, np.exp(-2)) + GPy.kern.white(Q, np.exp(-2))
    m = BayesianGPLVM(data['Y'], Q, init="PCA", num_inducing=20, kernel=kernel)
    # optimize
    m.ensure_default_constraints()
    m.optimize('scg', messages=1, max_iters=200, xtol=1e-300, ftol=1e-300)
    m._set_params(m._get_params())
    plt.clf, (latent_axes, sense_axes) = plt.subplots(1, 2)
    plt.sca(latent_axes)
    m.plot_latent()
    y = m.likelihood.Y[0, :].copy()
    data_show = GPy.util.visualize.stick_show(y[None, :], connect=data['connect'])
    lvm_visualizer = GPy.util.visualize.lvm_dimselect(m.X[0, :].copy(), m, data_show, latent_axes=latent_axes, sense_axes=sense_axes)
    raw_input('Press enter to finish')

    return m


def cmu_mocap(subject='35', motion=['01'], in_place=True):

    data = GPy.util.datasets.cmu_mocap(subject, motion)
    Y = data['Y']
    if in_place:
        # Make figure move in place.
        data['Y'][:, 0:3] = 0.0
    m = GPy.models.GPLVM(data['Y'], 2, normalize_Y=True)

    # optimize
    m.optimize(messages=1, max_f_eval=10000)

    ax = m.plot_latent()
    y = m.likelihood.Y[0, :]
    data_show = GPy.util.visualize.skeleton_show(y[None, :], data['skel'])
    lvm_visualizer = GPy.util.visualize.lvm(m.X[0, :].copy(), m, data_show, ax)
    raw_input('Press enter to finish')
    lvm_visualizer.close()

    return m

# def BGPLVM_oil():
#     data = GPy.util.datasets.oil()
#     Y, X = data['Y'], data['X']
#     X -= X.mean(axis=0)
#     X /= X.std(axis=0)
#
#     Q = 10
#     num_inducing = 30
#
#     kernel = GPy.kern.rbf(Q, ARD=True) + GPy.kern.bias(Q) + GPy.kern.white(Q)
#     m = GPy.models.BayesianGPLVM(X, Q, kernel=kernel, num_inducing=num_inducing)
#     # m.scale_factor = 100.0
#     m.constrain_positive('(white|noise|bias|X_variance|rbf_variance|rbf_length)')
#     from sklearn import cluster
#     km = cluster.KMeans(num_inducing, verbose=10)
#     Z = km.fit(m.X).cluster_centers_
#     # Z = GPy.util.misc.kmm_init(m.X, num_inducing)
#     m.set('iip', Z)
#     m.set('bias', 1e-4)
#     # optimize
#
#     import pdb; pdb.set_trace()
#     m.optimize('tnc', messages=1)
#     print m
#     m.plot_latent(labels=data['Y'].argmax(axis=1))
#     return m

