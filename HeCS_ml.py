import os
import math

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import sklearn.gaussian_process as gp
import sklearn.preprocessing as skp
import sklearn.gaussian_process as gp
from sklearn.gaussian_process.kernels import RationalQuadratic, \
    Matern, RBF, ConstantKernel, DotProduct
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
    scalerxy = skp.MinMaxScaler()
    train_xy = scalerxy.fit_transform(train_xy)
    test_xy = scalerxy.transform(test_xy)
    # same for z
    scalerz = skp.MinMaxScaler()
    train_z = scalerz.fit_transform(train_z.reshape(-1, 1))
    test_z = scalerz.transform(test_z.reshape(-1, 1))

    for nu in [0.5, 1.0, 1.5, 2.0]:
        print("Using nu:", nu)
        scale = 1.0
        kernel = scale * Matern(length_scale=scale, nu=nu)

        matn_gp = gp.GaussianProcessRegressor(kernel=kernel, \
            n_restarts_optimizer=50, \
            normalize_y=False)
        print("Starting fit...")
        matn_gp.fit(train_xy, train_z)
        print("Fit completed.")

        z_pred, std = matn_gp.predict(train_xy, return_std=True)
        z_pred_sb = scalerz.inverse_transform(z_pred.reshape(-1, 1))
        train_z_sb = scalerz.inverse_transform(train_z.reshape(-1, 1))
        trainmse = np.mean((z_pred_sb - train_z_sb) ** 2)
        print("Train MSE:", trainmse)

        z_pred = matn_gp.predict(test_xy)
        z_pred_sb = scalerz.inverse_transform(z_pred.reshape(-1, 1))
        test_z_sb = scalerz.inverse_transform(test_z.reshape(-1, 1))
        mse = np.mean((z_pred_sb - test_z_sb) ** 2)
        print("Test MSE:", mse)
