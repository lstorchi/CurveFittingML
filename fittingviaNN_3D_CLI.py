import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

import commonmodules as cm

#######################################################################

if __name__ == "__main__":

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

    train_x, test_x, train_y, test_y = cm.test_train_split (1, [w], x_s, y_s)

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

    cm.plotfull3dcurve (1, np.asarray(toplotx), np.asarray(toploty))