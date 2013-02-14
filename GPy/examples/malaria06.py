"""
Multioutput GP for malaria counts
---------------------------------
dataset: ../../../playground/malaria/malaria_data20130213.dat
B matrix controls the relation between districts
"""
#NOTE This is a non-sparse model

import numpy as np
import pylab as pb
import shelve
import GPy
import datetime
pb.ion()
pb.close('all')

#Load data
malaria_data = shelve.open('../../../playground/malaria/malaria_data_20130213.dat',writeback=False)
all_districts = malaria_data['districts']
all_variables = malaria_data['headers']
stations = malaria_data['stations']

#Define districts to analize
d_names = ['Mubende','Masindi','Mbarara','Kampala']
#['Mubende','Nakasongola']#,'Kamuli','Kampala','Mukono','Luwero','Tororo']
d_numbers = np.hstack([np.arange(len(malaria_data['districts']))[np.array(malaria_data['districts']) == d_i] for d_i in d_names])
if len(d_names) > len(d_numbers):
    print 'Warning: some districts were not found in malaria_data'

#Define output
Y_names = ['incidences']
Y_numbers = np.hstack([np.arange(len(malaria_data['headers']))[np.array(malaria_data['headers']) == Y_i] for Y_i in Y_names])
Y_list = [malaria_data['data'][d_i][:,Y_numbers] for d_i in d_numbers]
if len(Y_names) > 1:
    for num_i in range(len(d_numbers)):
        Y_list[num_i] = np.vstack(np.log(Y_list[num_i]))

#Define input
X_names = ['district','rain']#'time']#,'rain','ndvi','humidity_06','humidity_12','rain','temperature_min','temperature_max']
X_numbers = np.hstack([np.arange(len(malaria_data['headers']))[np.array(malaria_data['headers']) == n_i] for n_i in X_names])
X_list = [malaria_data['data'][d_i][:,X_numbers] for d_i in d_numbers]
for X_i,num_i in zip(X_list,range(len(d_names))):
    X_i[:,0] = np.repeat(num_i,X_i.shape[0]) #Change district number according to data analyzed

#Close data file
malaria_data.close()

#Create likelihood
likelihoods = []
for Y_i in Y_list:
    likelihoods.append(GPy.likelihoods.Gaussian(Y_i))

#Define coreg_kern and base kernels
R = len(d_names)
D = len(X_names) - 1 #-1 if district is in X_names
rbf = GPy.kern.rbf(D)
noise = GPy.kern.white(D)
base = rbf + rbf.copy() + noise
kernel = GPy.kern.icm(base,R,index=0,Dw=2)

#Define model
m = GPy.models.mGP(X_list, likelihoods, kernel, normalize_X=True) #NOTE: better to normalize X and Y

#Constraints
m.scale_factor = 1.
m.ensure_default_constraints()
m.unconstrain('rbf_1_var')
m.constrain_fixed('rbf_1_var',1.) # Variance parameter will be given by the elements of the coregionalization matrix
m.unconstrain('rbf_2_var')
m.constrain_fixed('rbf_2_var',1.) # Variance parameter will be given by the elements of the coregionalization matrix
m.constrain_positive('kappa')
m.set('_1_len',10)
m.set('_2_len',.1)
m.set('W',np.random.rand(R*2))
#Optimize
print m.checkgrad(verbose=True)
m.optimize()

#Plots
#m.plot_f()
m.plot()
"""
for r in range(R):
    for i in range(len(X_names)-1):
        pb.figure()
        m.plot_HD(input_col=i,output_num=r)
        pb.title('%s: %s' %(d_names[r],X_names[i+1]))

"""
#Print model
print m

# Print B matrix
print np.round(m.kern.parts[0].B,2)

#Plot W matrix
pb.figure()
W = m.kern.parts[0].W
pb.plot(W[0,:],W[1,:],'kx')
for wi_0, wi_1, name_i in zip(W[0,:],W[1,:],d_names):
    pb.text(x = wi_0, y = wi_1, s = name_i)
