import os
import math

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import sklearn.gaussian_process as gp
import sklearn.preprocessing as skp

import commonmodules as cm

if __name__ == "__main__":
    filename = "HeCS.xlsx"
    df = pd.read_excel(filename, sheet_name="dv1")
    #print(df.head())
    #for cname in df.columns:
    #    print(cname, df[cname].dtype)
    x = df[['v', 'T']].values
    y = df['RC'].values
    X = np.array(x)
    Y = np.array(y)
    temp_values = np.unique(X[:, 1])
    vib_values = np.unique(X[:, 0])

    print("X shape :", X.shape)
    print("Y shape :", Y.shape)
    vib_values_uniq = np.unique(X[:, 0])
    print("Vibration values:", vib_values_uniq)
    vibtest = vib_values_uniq[int(len(vib_values_uniq)/2)]
    print("Testing with vibration value:", vibtest)
    train_xy = []
    train_z = []
    test_xy = []
    test_z = []
    for i in range(X.shape[0]):
        if X[i][0] == vibtest:
            test_xy.append(X[i])
            test_z.append(Y[i])
        else:
            train_xy.append(X[i])
            train_z.append(Y[i])
    train_xy = np.array(train_xy)
    train_z = np.array(train_z)
    test_xy = np.array(test_xy)
    test_z = np.array(test_z)
    print("Train X shape:", train_xy.shape)
    print("Train Y shape:", train_z.shape)
    print("Test X shape:", test_xy.shape)
    print("Test Y shape:", test_z.shape)        


    # log scale for Y
    train_z = np.log(train_z)
    test_z = np.log(test_z)
    # scale values to [0, 1]
    scaler = skp.MinMaxScaler()
    train_xy = scaler.fit_transform(train_xy)
    test_xy = scaler.transform(test_xy)
    # same for z
    train_z = scaler.fit_transform(train_z.reshape(-1, 1))
    test_z = scaler.transform(test_z.reshape(-1, 1))

    model = cm.build_model_GP_1 (train_xy, train_z)

    z_pred, std = model.predict(train_xy, return_std=True)
    trainmse = 0.0
    cont = 0.0
    for i in range(train_z.shape[0]):
        x = train_xy[i,0]
        y = train_xy[i,1]
        z = train_z[i]
        zpred = z_pred[i]
        zstd = std[i]

        trainmse += (zpred-z)**2
        cont += 1.0

        print("Train, %10.7f , %10.7f , %10.7f , %10.7f , %10.7f"%(z, y, z, zpred, zstd))

    trainmse = trainmse/cont

    z_pred, std = model.predict(test_xy, return_std=True)
    mse = 0.0
    cont = 0.0
    for i in range(test_z.shape[0]):
        x = test_xy[i,0]
        y = test_xy[i,1]
        z = test_z[i]
        zpred = z_pred[i]
        zstd = std[i]

        mse += (zpred-z)**2
        cont += 1.0

        print("Test, %10.7f , %10.7f , %10.7f , %10.7f , %10.7f"%(z, y, z, zpred, zstd))

    mse = mse/cont

    print("MSE ", mse, " , TrainMSE ,", trainmse, flush=True)

