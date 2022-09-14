from cProfile import label
import sys

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

print(Y.shape, X.shape)

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

#model = RandomForestRegressor()

model.fit(X, Y)

xtest = []
ytest = []
t = 1000
for idx in range(len(vib)):
    xtest.append([t, vib[idx]])
    ytest.append(df[t].values[idx])

xtest = []
ytest = []
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

#y_pred, std = model.predict(X, return_std=True)
y_pred = model.predict(X)

MSE = ((y_pred-Y)**2).mean()

#for i in range(Y.shape[0]):
#    print(y_pred[i], Y[i])

#print("MSE: ", MSE)
Xtest = np.array(xtest)
Ytest = np.array(ytest)
#plt.scatter(y_pred*10**20, Y*10**20)
#plt.show()

y_pred, std = model.predict(Xtest, return_std=True)
#y_pred = model.predict(Xtest)

print(len(y_pred), len(std))

plt.plot(Xtest[:,0], Ytest, label="True values")
#plt.errorbar(Xtest[:,1], y_pred, std, linestyle='None', marker='^')
plt.plot(Xtest[:,0], y_pred, label="Predicted values")
plt.xlabel("T")
plt.ylabel("Rate")
plt.legend()

plt.show()