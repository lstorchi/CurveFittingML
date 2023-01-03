import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import sklearn.gaussian_process as gp
from mpl_toolkits.mplot3d import Axes3D

import commonmodules as cm

if __name__ == "__main__":

    ofp = open("results_GP.csv", "w")

    filename = "testdv1.xlsx"
    df, vib_values , temp_values = cm.filterinitialset (filename)
    #plotfull3dcurve (df, vib_values, temp_values)
    
    overallmse = 0.0
    overalltrainmse = 0.0
    tot = 0
    traintot = 0
    for trm in temp_values:
        print("Removing TEMP ", trm, flush=True)
        temp_torm = [trm]
    
        train_xy, train_z, test_xy, test_z = cm.get_train_and_test_rmt (temp_values, vib_values, \
            df, temp_torm)
    
        model = cm.fitusingscikitl (train_xy, train_z)
    
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
            overalltrainmse += (zpred-z)**2
            traintot += 1
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
            overallmse += (zpred-z)**2
            tot += 1
            cont += 1.0
    
            print("Test, %10.7f , %10.7f , %10.7f , %10.7f , %10.7f"%(z, y, z, zpred, zstd))
    
        mse = mse/cont

        print("Removed TEMP , ", trm, " ,", mse[0], " ,", trainmse[0], flush=True, file=ofp)
    
        print("Removed TEMP , ", trm, " , MSE , ", mse, " , TrainMSE ,", trainmse, flush=True)
    
    print("Overall TEMP MSE , ", overallmse/float(tot), \
        " , Train MSE , ", overalltrainmse/float(traintot))
    
    overallmse = 0.0
    overalltrainmse = 0.0
    tot = 0
    traintot = 0
    for vrm in vib_values:
        vib_torm = [vrm]
        print("Removing VIB ", vrm, flush=True)
    
        train_xy, train_z, test_xy, test_z = cm.get_train_and_test_rmv (temp_values, vib_values, \
            df, vib_torm)
    
        model = cm.fitusingscikitl (train_xy, train_z)
    
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
            overalltrainmse += (zpred-z)**2
            traintot += 1
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
            overallmse += (zpred-z)**2
            tot += 1
            cont += 1.0
        
            print("Test, %10.7f , %10.7f , %10.7f , %10.7f , %10.7f"%(z, y, z, zpred, zstd))
    
        mse = mse/cont
    
        print("Removed VIB  , ", vrm, " ,", mse[0], " ,", trainmse[0], flush=True, file=ofp)

        print("Removed VIB  , ", vrm, " , MSE , ", mse, " , TrainMSE ,", trainmse, flush=True)
    
    print("Overall VIB MSE , ", overallmse/float(tot), \
        " , Train MSE , ", overalltrainmse/float(traintot))
    
    for perc in [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]:
    
        train_xy, train_z, test_xy, test_z = cm.get_train_and_test_random (temp_values, vib_values, \
            df, perc)
    
        model = cm.fitusingscikitl (train_xy, train_z)
    
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

        print("Removed RND  , ", perc, " ,", mse[0], " ,", trainmse[0], flush=True, file=ofp)
    
        print("Removed random values ", perc ,"  MSE ", mse, " , TrainMSE ,", \
            trainmse, flush=True) 