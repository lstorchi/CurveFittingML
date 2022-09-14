from cProfile import label
import sys
from tabnanny import verbose

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sklearn.gaussian_process as gp

from sklearn.ensemble import RandomForestRegressor

filename = ""

if len(sys.argv) != 2:
    print("Usage: %s filename", sys.argv[0])
    exit(1)
else:
    filename = sys.argv[1]

df = pd.read_excel(filename)

vib= []
for v in df["vibrational level v\Temperature(K)"].values:
    vib.append(float(v))
    #print("\"",v,"\"")

y = []
x = []
for t in df.columns[1:]:
    T = float(t)
    #print(T)
    for idx in range(len(vib)):
        #print(T, vib[idx])
        x.append([T, vib[idx]])
        y.append(df[t].values[idx])

X = np.array(x)
Y = np.array(y)

Xtrain = X
Ytrain = Y

Xtest = X
Ytest = Y

print("Training set shapes: ", Ytrain.shape, Xtrain.shape)
print("    Test set shapes: ", Ytest.shape, Xtest.shape)

xdim = len(df.columns)-1
ydim = len(vib)

Xp = np.zeros((xdim, ydim), dtype=float)
Yp = np.zeros((xdim, ydim), dtype=float)
Zp = np.zeros((xdim, ydim), dtype=float)
for xidx in range(xdim):
    T = float(df.columns[xidx+1])
    t = df.columns[xidx+1]
    for yidx in range(ydim):
        v =  vib[yidx]
        Xp[xidx, yidx] = T
        Yp[xidx, yidx] = v
        Zp[xidx, yidx] = df[t].values[yidx]

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(Xp, Yp, Zp, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=False)

"""
for xidx in range(xdim):
    for yidx in range(ydim):
        xs = Xp[xidx, yidx]
        ys = Yp[xidx, yidx]
        zs = Zp[xidx, yidx]
        ax.scatter(xs, ys, zs)
"""

plt.show()

kernel = gp.kernels.ConstantKernel(1.0, (1e-3, 1e3)) * gp.kernels.RBF([5,5], (1e-2, 1e2))
model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15)
#, normalize_y=False)

#model = RandomForestRegressor()

model.fit(Xtrain, Ytrain)

y_pred = model.predict(Xtest)
MSE = ((y_pred-Ytest)**2).mean()
print ("MSE: ", MSE)

vsvib = False
xtest = []
ytest = []
columnidx = -1

if vsvib:
    t = 1000
    for idx in range(len(vib)):
        xtest.append([t, vib[idx]])
        ytest.append(df[t].values[idx])
    columnidx = 1 
else:
    v = 14
    vidx = -1
    for idx in range(len(vib)):
        if vib[idx] == v:
            vidx = idx
            break
    for t in df.columns[1:]:
        T = float(t)
        xtest.append([T, v])
        ytest.append(df[t].values[vidx])
    columnidx = 0

Xtest_single = np.array(xtest)
Ytest_single = np.array(ytest)

y_pred, std = model.predict(Xtest_single, return_std=True)

for idx in range(len(y_pred)):
    print("%10.5e %10.5e"%(y_pred[idx], std[idx]))

plt.plot(Xtest_single[:,columnidx], Ytest_single, label="True values")
plt.plot(Xtest_single[:,columnidx], y_pred, label="Predicted values")
#plt.errorbar(Xtest_single[:,columnidx], y_pred, std, label="Predicted values")
plt.xlabel("T")
plt.ylabel("Rate")
plt.legend()

plt.show()