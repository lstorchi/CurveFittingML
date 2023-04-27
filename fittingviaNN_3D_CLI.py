import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

#######################################################################

def test_train_split (column, valuestotest, x, y):
    
    xtest = []
    ytest = []
    xtrain = []
    ytrain = []
    
    for v in valuestotest:
        for i, xv in enumerate(x[:,column]):
            if xv == v:
                xtest.append(x[i,:])
                ytest.append(y[i])
            else:
                xtrain.append(x[i,:])
                ytrain.append(y[i])   

    return np.asarray(xtrain), np.asarray(xtest), \
        np.asarray(ytrain), np.asarray(ytest)

#######################################################################

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

filename = "N2H2_VVdata_3variables.xlsx"
df = pd.read_excel(filename)

x = df[['v', 'w', 'T(K)']].values
y = df[['k(cm^3/s)']].values

scalerx = MinMaxScaler()
scalerx.fit(x)
x_s = scalerx.transform(x)

vset = set(x_s[:,0])
wset = set(x_s[:,1])
tset = set(x_s[:,2])

scalery = MinMaxScaler()
scalery.fit(y)
y_s = scalery.transform(y)

w = 0

train_x, test_x, train_y, test_y = test_train_split (1, [w], x_s, y_s)

print("Train shape: ", train_x.shape, "Test shape: ", test_x.shape)

toplotx = []
toploty = []
for i in range(test_x.shape[0]):
    v = test_x[i,0]
    w = test_x[i,1]
    T = test_x[i,2]
    k = test_y[i]

    ix = scalerx.inverse_transform(np.asarray([v, w, T]).reshape(1, -1))
    iy = scalery.inverse_transform(np.asarray([k]))

    toplotx.append([int(ix[0,0]), int(ix[0,1]),int(ix[0,2])])
    toploty.append(iy)

    #print("%4d %4d %5d %10.5e"%(int(ix[0,0]), int(ix[0,1]),int(ix[0,2]), iy[0]))

plotfull3dcurve (1, np.asarray(toplotx), np.asarray(toploty))