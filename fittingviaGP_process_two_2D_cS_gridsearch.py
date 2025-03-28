import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import commonmodules as cm
import sklearn.gaussian_process as gp
from sklearn.gaussian_process.kernels import RationalQuadratic, Matern, RBF, ConstantKernel, LinearKernel

import time

if __name__ == "__main__":

    filename = "N2H2_VT_process.xlsx"
    sheetname = "dv=1"
    if len(sys.argv) == 3:
        filename = sys.argv[1]
        sheetname = sys.argv[2]
    elif len(sys.argv) == 2:
        sheetname = sys.argv[1]

    df = pd.read_excel(filename, sheet_name=sheetname)
    debug = False

    x = df[['v', 'cE']].values
    y = np.log10(df[['cS']].values)

    scalery = MinMaxScaler()
    scalery.fit(y)
    y_s = scalery.transform(y)

    scalerx = MinMaxScaler()
    scalerx.fit(x)
    x_s = scalerx.transform(x)

    vmap_toreal = {}

    for i, vn in enumerate(x_s[:,0]):
        vmap_toreal[vn] = x[i,0]

    dim  = len(vmap_toreal)
    ntorm = int(dim/2)
    print("V map: ")
    v = None
    for i, a in enumerate(vmap_toreal):
        print("%4.2f --> %3d"%(a, vmap_toreal[a]))
        if i == ntorm:
            print("ntorm = %4.2f --> %3d"%(a, vmap_toreal[a]))
            v = a

    vset = set(x_s[:,0])

    print("v in test = ", vmap_toreal[v])
    train_x, test_x, train_y, test_y = cm.test_train_split (0, [v], x_s, y_s)

    models = []

    # Gaussian Process Regression

    # 1. Using Rational Quadratic Kernel
    kernel_rq = ConstantKernel(constant_value=1.0, constant_value_bounds="fixed") * \
          RationalQuadratic(length_scale=1.0, alpha=0.1)
    gpr_rq = gp.GaussianProcessRegressor(kernel=kernel_rq, alpha=0.1, n_restarts_optimizer=10)
    models.append(gpr_rq)

    # 2. Using Matern Kernel (nu=1.5)
    kernel_matern = ConstantKernel(constant_value=1.0, constant_value_bounds="fixed") * \
        Matern(length_scale=1.0, nu=1.5)
    gpr_matern = gp.GaussianProcessRegressor(kernel=kernel_matern, alpha=0.1, n_restarts_optimizer=10)
    models.append(gpr_matern)

    # 3. Using a combination (RQ + Linear)
    kernel_combined = ConstantKernel(constant_value=1.0, constant_value_bounds="fixed") * \
        RationalQuadratic(length_scale=1.0, alpha=0.1) + LinearKernel(gradient=1.0)
    gpr_combined = gp.GaussianProcessRegressor(kernel=kernel_combined, alpha=0.1, n_restarts_optimizer=10)
    models.append(gpr_combined)

    # 4. Using Matern Kernel (nu=0.5)
    kernel_matern = ConstantKernel(constant_value=1.0, constant_value_bounds="fixed") * \
        Matern(length_scale=1.0, nu=0.5)
    gpr_matern = gp.GaussianProcessRegressor(kernel=kernel_matern, alpha=0.1, n_restarts_optimizer=10)
    models.append(gpr_matern)

    modelnum = 1     
    for model in models:
        starttime = time.time()
        ofp = open("vremoved_GP_"+str(modelnum)+".csv", "w")
    
        avgr2test = 0.0
        avgmsetest = 0.0
        avgr2train = 0.0
        avgmsetrain = 0.0
        print ("Modelnum , v Removed , Test MSE , Test R2 , Train MSE , Train R2")
        print ("Modelnum , v Removed , Test MSE , Test R2 , Train MSE , Train R2", file=ofp)

        test_y_sb = scalery.inverse_transform(test_y)

        model.fit(train_x, train_y)

        test_x_sb = scalerx.inverse_transform(test_x)
        pred_y = model.predict(test_x)
        pred_y_sb = scalery.inverse_transform(pred_y.reshape(-1,1))
        test_y_sb = scalery.inverse_transform(test_y)

        with open("vremoved_GP_"+\
                  str(modelnum)+"_test.csv", "w") as ofptest:
            for ix, x_sb in enumerate(test_x_sb):
                for xx in x_sb:
                    print("%15.19e , "%(xx), end="", file=ofptest)
                print("%15.19e , %15.19e "%(test_y_sb[ix], pred_y_sb[ix]),
                        file=ofptest, flush=True)

        testmse = metrics.mean_absolute_error(test_y_sb, pred_y_sb)
        testr2 = metrics.r2_score(test_y_sb, pred_y_sb)
        
        pred_y = model.predict(train_x)
        pred_y_sb = scalery.inverse_transform(pred_y.reshape(-1,1))
        train_y_sb = scalery.inverse_transform(train_y.reshape(-1,1))
        train_x_sb = scalerx.inverse_transform(train_x)
        with open("vremoved_GP_"+\
                    str(modelnum)+"_train.csv", "w") as ofptrain:
            for ix, x_sb in enumerate(train_x_sb):
                for xx in x_sb:
                    print("%15.19e , "%(xx), end="", file=ofptrain)
                print("%15.19e , %15.19e "%(train_y_sb[ix], pred_y_sb[ix]),
                        file=ofptrain, flush=True)
        trainmse = metrics.mean_absolute_error(train_y_sb, pred_y_sb)
        trainr2 = metrics.r2_score(train_y_sb, pred_y_sb)
        
        print("%3d , %3d , %10.6e , %10.6f , %10.6e , %10.6f"%(\
            modelnum, vmap_toreal[v], testmse, testr2, \
            trainmse,  trainr2))
        
        print("%3d , %3d , %10.6e , %10.6f , %10.6e , %10.6f"%(\
            modelnum, vmap_toreal[v], testmse, testr2, \
            trainmse,  trainr2), file=ofp)
        
        ofp.close()
        endtime = time.time()
        print("Time taken for nu = %4.2f is %10.6f"%(nu, endtime-starttime))
        modelnum += 1