import random

import math
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import sklearn.gaussian_process as gp
from mpl_toolkits.mplot3d import Axes3D

from tensorflow import keras
import tensorflow as tf

import tensorflow.keras.optimizers as tko
import tensorflow.keras.activations as tka
import tensorflow.keras.losses as tkl
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

##########################################################################################################

def __build_activation_function(af):
    if af.lower() == 'none':
        return None
    exp_af = 'lambda _ : tka.' + af
    return eval(exp_af)(None)

def __build_optimizer(optimizer):
    opt_init = optimizer
    exp_po = 'lambda _ : tko.' + opt_init
    optimizer = eval(exp_po)(None)
    return optimizer

def __build_loss(loss):
    exp_loss = 'lambda _ : tkl.' + loss
    return eval(exp_loss)(None)

def __build_model (hidden_layers_layout, activation_functions):
    inputs = Input(shape=(2,))
    hidden = inputs
    for i in range(0, len(hidden_layers_layout)):
        hidden = Dense(hidden_layers_layout[i], \
                activation= __build_activation_function(activation_functions[i]))(hidden)
    outputs = Dense(1)(hidden)
    model = Model(inputs=inputs, outputs=outputs)
   
    return model

##########################################################################################################

def build_model_NN_2 ():

    hidden_layers_layout = [100, 100]
    activation_functions = ["relu", "relu"]
    inoptimizer = "SGD(decay=1e-6, momentum=0.9, nesterov=True)"
    loss = 'MeanSquaredError()'

    model = __build_model (hidden_layers_layout, activation_functions)
    optimizer = __build_optimizer(inoptimizer)
    model.compile(loss= __build_loss(loss), optimizer=optimizer)
    #model.summary()

    return model

##########################################################################################################

def build_model_NN_3 ():

    model = keras.Sequential()
    model.add(keras.layers.Dense(units = 2, activation = 'linear', input_shape=[2]))
    model.add(keras.layers.Dense(units = 100, activation = 'relu'))
    model.add(keras.layers.Dense(units = 100, activation = 'relu'))
    model.add(keras.layers.Dense(units = 1, activation = 'linear'))
    model.compile(loss='mse', optimizer="adam", metrics='mse')

    return model

##########################################################################################################

def build_model_NN_1 ():

    model = keras.Sequential()
    model.add(keras.layers.Dense(units = 2, activation = 'linear', input_shape=[2]))
    model.add(keras.layers.Dense(units = 32, activation = 'relu'))
    model.add(keras.layers.Dense(units = 64, activation = 'relu'))
    model.add(keras.layers.Dense(units = 128, activation = 'relu'))
    model.add(keras.layers.Dense(units = 32, activation = 'relu'))
    #model.add(keras.layers.Dense(units = 16, activation = 'relu'))
    model.add(keras.layers.Dense(units = 1, activation = 'linear'))
    model.compile(loss='mse', optimizer="adam", metrics='mse')
    #model.summary()

    return model

##########################################################################################################3

def build_model_GP_1 (train_x, train_y):
    #kernel = gp.kernels.ConstantKernel(1.0, (1e-5, 1e5))* gp.kernels.RBF(length_scale=1)
    kernel = gp.kernels.ConstantKernel(1.0, (1e-3, 1e3)) * gp.kernels.RBF([5,5], (1e-2, 1e2))
    model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50, \
        normalize_y=False)
    print("Start training ")
    model.fit(train_x, train_y)
    print("Done ")

    return model

##########################################################################################################3

def build_model_GP_2 (train_x, train_y):
    #kernel = gp.kernels.ConstantKernel(1.0, (1e-5, 1e5))* gp.kernels.RBF(length_scale=1)
    nuval = 5.0/2.0
    kernel = 1.0 * gp.kernels.Matern(length_scale=1.0, nu=nuval)
    model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50, \
        normalize_y=False)
    print("Start training ")
    model.fit(train_x, train_y)
    print("Done ")

    return model

##########################################################################################################3

def build_model_GP_2D (train_x, train_y):
    #kernel = gp.kernels.ConstantKernel(1.0, (1e-5, 1e5))* gp.kernels.RBF(length_scale=1)
    nuval = 5.0/2.0
    kernel = 1.0 * gp.kernels.Matern(length_scale=1.0, nu=nuval)
    model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50, \
        normalize_y=False)
    print("Start training ")
    model.fit(train_x, train_y)
    print("Done ")

    return model

##########################################################################################################

def build_model_GP_3D (train_x, train_y):
    #kernel = gp.kernels.ConstantKernel(1.0, (1e-5, 1e5))* gp.kernels.RBF(length_scale=1)
    nuval = 5.0/2.0
    kernel = 1.0 * gp.kernels.Matern(length_scale=1.0, nu=nuval)
    model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50, \
        normalize_y=False)
    print("Start training ")
    model.fit(train_x, train_y)
    print("Done ")

    return model

#####################################################################################################

def filterinitialset (filename, headername = "vibrational level v\Temperature(K)",  \
    factor = 1.0, normalize = False):

    dfin = pd.read_excel(filename)

    #print(dfin)
    
    dfdict = {}
    
    tempvalues = list(dfin.columns[1:])
    vibvalues = list(dfin[headername].values)

    min = float("inf")
    max = float("-inf")
    for c in dfin.columns:
        dfdict[c] = []
        if c == headername:
            dfdict[c] = list(dfin[c].values)
        else:
            for v in dfin[c].values:
                val = math.log10(factor*v)
                if val > max:
                    max = val
                if val < min:
                    min = val

    for c in dfin.columns:
        if c != headername:
            for v in dfin[c].values:
                val = math.log10(factor*v)
                valp = (val - min) / (max - min)
                if normalize:
                    dfdict[c].append(valp)
                else:
                    dfdict[c].append(val)

    df = pd.DataFrame.from_dict(dfdict)

    return df, vibvalues, tempvalues, min, max

##############################################################################

def plotfull3dcurve (df, vib_values, temp_values):

    y = []
    x = []
    for t in temp_values:
        for idx in range(len(vib_values)):
            x.append([float(t), float(vib_values[idx])])
            y.append(df[t].values[idx])

    X = np.array(x)
    Y = np.array(y)

    xdim = len(temp_values)
    ydim = len(vib_values)

    Xp = np.zeros((xdim, ydim), dtype=float)
    Yp = np.zeros((xdim, ydim), dtype=float)
    Zp = np.zeros((xdim, ydim), dtype=float)
    for xidx in range(xdim):
        t = temp_values[xidx]
        for yidx in range(ydim):
            v =  vib_values[yidx]
            Xp[xidx, yidx] = float(t)
            Yp[xidx, yidx] = float(v)
            Zp[xidx, yidx] = df[t].values[yidx]

    #fig = plt.figure(figsize=(10,8))
    fig = plt.figure(figsize=plt.figaspect(2.))
    plt.gcf().set_size_inches(40, 30)
    ax = fig.add_subplot(2,1,1, projection='3d')
    surf = ax.plot_surface(Xp, Yp, Zp, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=False)
    plt.show()

##########################################################################################################

def get_train_and_test_rmt (temp_values, vib_values, df, \
    removetemps=[]):

    maxt = max(temp_values)
    mint = min(temp_values)

    minv = min(vib_values)
    maxv = max(vib_values)

    train_xy = []
    train_z = []

    test_xy = []
    test_z = []

    maxz = float("-inf")
    minz = float("+inf")

    for tidx, t in enumerate(temp_values):
        for vidx, v in enumerate(vib_values):
            zval = df[t].values[vidx]

            if zval < minz:
                minz = zval
            elif zval > maxz:
                maxz = zval

    for t in temp_values:
        if t not in removetemps:
            tnorm = (t - mint)/(maxt - mint)

            for vidx, v in enumerate(vib_values):
                vnorm  = (v - minv)/(maxv - minv)
                train_xy.append([tnorm, vnorm])
        
                z = df[t].values[vidx]
                znorm = (z - minz)/(maxz - minz)
                train_z.append(znorm)
        else:
            tnorm = (t - mint)/(maxt - mint)
            for vidx, v in enumerate(vib_values):
                vnorm  = (v - minv)/(maxv - minv)
                test_xy.append([tnorm, vnorm])

                z = df[t].values[vidx]
                znorm = (z - minz)/(maxz - minz)
                test_z.append(znorm)


    train_xy = np.asarray(train_xy)
    train_z = np.asarray(train_z)

    test_xy = np.asarray(test_xy)
    test_z = np.asarray(test_z)

    return train_xy, train_z, test_xy, test_z

##########################################################################################################

def get_train_and_test_rmv (temp_values, vib_values, df, \
    removevibs=[]):

    maxt = max(temp_values)
    mint = min(temp_values)

    minv = min(vib_values)
    maxv = max(vib_values)

    train_xy = []
    train_z = []

    test_xy = []
    test_z = []

    maxz = float("-inf")
    minz = float("+inf")

    for tidx, t in enumerate(temp_values):
        for vidx, v in enumerate(vib_values):
            zval = df[t].values[vidx]

            if zval < minz:
                minz = zval
            elif zval > maxz:
                maxz = zval

    for t in temp_values:
        tnorm = (t - mint)/(maxt - mint)

        for vidx, v in enumerate(vib_values):
            if v not in removevibs:
                vnorm  = (v - minv)/(maxv - minv)
                train_xy.append([tnorm, vnorm])
        
                z = df[t].values[vidx]
                znorm = (z - minz)/(maxz - minz)
                train_z.append(znorm)
            else:
                vnorm  = (v - minv)/(maxv - minv)
                test_xy.append([tnorm, vnorm])

                z = df[t].values[vidx]
                znorm = (z - minz)/(maxz - minz)
                test_z.append(znorm)

    train_xy = np.asarray(train_xy)
    train_z = np.asarray(train_z)

    test_xy = np.asarray(test_xy)
    test_z = np.asarray(test_z)

    return train_xy, train_z, test_xy, test_z

##########################################################################################################

def get_train_and_test_random (temp_values, vib_values, df, \
    perc):

    maxt = max(temp_values)
    mint = min(temp_values)

    minv = min(vib_values)
    maxv = max(vib_values)

    train_xy = []
    train_z = []

    test_xy = []
    test_z = []

    maxz = float("-inf")
    minz = float("+inf")

    totnumber = 0;
    for t in temp_values:
        for vidx, v in enumerate(vib_values):
            zval = df[t].values[vidx]

            totnumber += 1

            if zval < minz:
                minz = zval
            elif zval > maxz:
                maxz = zval

    random.seed(1)

    for t in temp_values:
        tnorm = (t - mint)/(maxt - mint)
        for vidx, v in enumerate(vib_values):
            rv = random.uniform(0.0, float(totnumber))
            #print(rv, perc*float(totnumber))
            if rv > (perc*float(totnumber)):
                vnorm  = (v - minv)/(maxv - minv)
                train_xy.append([tnorm, vnorm])
        
                z = df[t].values[vidx]
                znorm = (z - minz)/(maxz - minz)
                train_z.append(znorm)
            else:
                vnorm  = (v - minv)/(maxv - minv)
                test_xy.append([tnorm, vnorm])

                z = df[t].values[vidx]
                znorm = (z - minz)/(maxz - minz)
                test_z.append(znorm)


    train_xy = np.asarray(train_xy)
    train_z = np.asarray(train_z)

    test_xy = np.asarray(test_xy)
    test_z = np.asarray(test_z)

    return train_xy, train_z, test_xy, test_z

##############################################################################

def get_train_full (temp_values, vib_values, df, vtopredict):

    maxt = max(temp_values)
    mint = min(temp_values)

    train_xy = []
    train_z = []

    maxz = float("-inf")
    minz = float("+inf")

    minv = min(vib_values)
    maxv = max(vib_values)

    totnumber = 0;
    for t in temp_values:
        for vidx, v in enumerate(vib_values):
            zval = df[t].values[vidx]

            totnumber += 1

            if zval < minz:
                minz = zval
            elif zval > maxz:
                maxz = zval

    for t in temp_values:
        tnorm = (t - mint)/(maxt - mint)
        for vidx, v in enumerate(vib_values):
            vnorm  = (v - minv)/(maxv - minv)
            train_xy.append([tnorm, vnorm])
            z = df[t].values[vidx]
            znorm = (z - minz)/(maxz - minz)
            train_z.append(znorm)

    test_xy = []
    for t in temp_values:
        tnorm = (t - mint)/(maxt - mint)
        for v in vtopredict:
            vnorm  = (v - minv)/(maxv - minv)
            test_xy.append([tnorm, vnorm])

    train_xy = np.asarray(train_xy)
    train_z = np.asarray(train_z)
    test_xy = np.asarray(test_xy)

    return train_xy, train_z, test_xy

##############################################################################

def plotfull3dcurve (columntorm, x, y):

    yv = []
    xv = []
    for i, v in enumerate(x):
        toappend = []
        for j in range(len(v)):
            if j != columntorm:
                toappend.append(v[j])
        xv.append(toappend)
        yv.append(y[i]) 

    X = np.array(xv)
    Y = np.array(yv)

    #for i in range(X.shape[0]):
    #    print("%4d %5d %10.5e"%(X[i,0], X[i,1], Y[i]))

    #print(X.shape)
    #print(Y.shape)

    x1set = sorted(list(set(X[:,0])))
    x2set = sorted(list(set(X[:,1])))

    x1dim = len(x1set)
    x2dim = len(x2set)

    Xp = np.zeros((x1dim, x2dim), dtype=float)
    Yp = np.zeros((x1dim, x2dim), dtype=float)
    Zp = np.zeros((x1dim, x2dim), dtype=float)
    for x1idx in range(x1dim):
        x1 = x1set[x1idx]
        for x2idx in range(x2dim):
            x2 =  x2set[x2idx]
            Xp[x1idx, x2idx] = float(x1)
            Yp[x1idx, x2idx] = float(x2)

            zval = None
            for i in range(X.shape[0]):
                if X[i,0] == x1 and X[i,1] == x2:
                    zval = Y[i]
                    break

            Zp[x1idx, x2idx] = zval

    #print(Xp.shape, Yp.shape, Zp.shape)

    #fig = plt.figure(figsize=(10,8))
    fig = plt.figure(figsize=plt.figaspect(2.))
    #plt.gcf().set_size_inches(40, 30)
    ax = fig.add_subplot(2, 1, 1, projection='3d')
    surf = ax.plot_surface(Xp, Yp, Zp, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=True)
    plt.show()

#######################################################################

def plotfull3dscatter (columntorm, x, y):

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for i, v in enumerate(x):
        toappend = []
        for j in range(len(v)):
            if j != columntorm:
                toappend.append(v[j])
        xs = toappend[0]
        ys = toappend[1]
        zs = y[i]
        ax.scatter(xs, ys, zs, marker='o')
    
    plt.show()

#######################################################################

def test_train_split (column, valuestotest, x, y):
    
    xtest = []
    ytest = []
    xtrain = []
    ytrain = []

    for i, xv in enumerate(x[:,column]):
        if xv in valuestotest:
            xtest.append(x[i,:])
            ytest.append(y[i])
        else:
            xtrain.append(x[i,:])
            ytrain.append(y[i])  
    
    #print(len(xtest), len(ytest))
    #print(len(xtrain), len(ytrain))

    return np.asarray(xtrain), np.asarray(xtest), \
        np.asarray(ytrain), np.asarray(ytest)

#######################################################################

def buildmodel(modelshape):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(3)))

    for n in modelshape:
        model.add(keras.layers.Dense(units = n, activation = 'relu'))

    model.add(keras.layers.Dense(units = 1, activation = 'linear'))
    model.compile(loss='mse', optimizer="adam", metrics='mse')

    return model

#######################################################################
