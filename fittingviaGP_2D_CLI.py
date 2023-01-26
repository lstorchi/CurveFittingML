import sys
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import sklearn.gaussian_process as gp
from mpl_toolkits.mplot3d import Axes3D

import commonmodules as cm

if __name__ == "__main__":

    filename = "testdv1.xlsx"

    if (len(sys.argv) == 2):
        filename = sys.argv[1]

    basename = filename.split(".")[0]

    for modelname in ["model2", "model1"]:

        ofp = open(basename+"_results_GP"+modelname+".csv", "w")
    
        print("Type , Index , Testset MSE , Trainingset MSE", flush=True, file=ofp)
    
        df, vib_values , temp_values, minvalue, maxvalue = cm.filterinitialset (filename)
        #plotfull3dcurve (df, vib_values, temp_values)
    
        overallmse = 0.0
        overalltrainmse = 0.0
        tot = 0
        traintot = 0

        # for dv=1
        #vib_values_torm = [[2, 4, 6, 8, 10, 14, 18, 22, 26, 30, 35], \
        #                   [1, 3, 5, 7, 9, 12, 16, 20, 24, 28, 32, 40], \
        #                   [2, 3, 5, 6, 8, 9, 12, 14, 18, 20, 24, 26, 30, 32], \
        #                   [1, 2, 4, 5, 7, 8, 10, 12, 16, 18, 22, 24, 28, 30, 35, 40]]

        # for dv=2
        #vib_values_torm = [[2, 4, 6, 8, 10, 14, 18, 22, 26, 30, 35], \
        #                   [3, 5, 7, 9, 12, 16, 20, 24, 28, 32, 40], \
        #                   [2, 3, 5, 6, 8, 9, 12, 14, 18, 20, 24, 26, 30, 32, 40], \
        #                   [3, 4, 6, 7, 9, 10, 14, 16, 20, 22, 26, 28, 32, 35]]
        
        # for dv=3
        vib_values_torm = [[2, 4, 6, 8, 10, 14, 18, 22, 26, 30, 35], \
                           [3, 5, 7, 9, 12, 16, 20, 24, 28, 32], \
                           [2, 3, 5, 6, 8, 9, 12, 14, 18, 20, 24, 26, 30, 32], \
                           [3, 4, 6, 7, 9, 10, 14, 16, 20, 22, 26, 28, 32, 35]]
        for vrm in vib_values_torm:
            print("Removing VIB set ", str(vrm).replace(",", ";"), flush=True)
        
            train_xy, train_z, test_xy, test_z = cm.get_train_and_test_rmv (temp_values, vib_values, \
                df, vrm)
 
            model = None
            if modelname == "model1":
                model = cm.build_model_GP_1 (train_xy, train_z)
            elif modelname == "model2":
                model = cm.build_model_GP_2 (train_xy, train_z)

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
        
            print("Removed VIB  , ", str(vrm).replace(",", ";"), " ,", mse, " ,", trainmse, flush=True, file=ofp)
        
            print("Removed VIB  , ", vrm, " , MSE , ", mse, " , TrainMSE ,", trainmse, flush=True)
        
        print("Overall VIB MSE , ", overallmse/float(tot), \
            " , Train MSE , ", overalltrainmse/float(traintot))
    
        overallmse = 0.0
        overalltrainmse = 0.0
        tot = 0
        traintot = 0
        for trm in temp_values:
            print("Removing TEMP ", trm, flush=True)
            temp_torm = [trm]
        
            train_xy, train_z, test_xy, test_z = cm.get_train_and_test_rmt (temp_values, vib_values, \
                df, temp_torm)

            model = None
            if modelname == "model1":
                model = cm.build_model_GP_1 (train_xy, train_z)
            elif modelname == "model2":
                model = cm.build_model_GP_2 (train_xy, train_z)

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
        
            print("Removed TEMP , ", trm, " ,", mse, " ,", trainmse, flush=True, file=ofp)
        
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

            model = None
            if modelname == "model1":
                model = cm.build_model_GP_1 (train_xy, train_z)
            elif modelname == "model2":
                model = cm.build_model_GP_2 (train_xy, train_z)
                
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
        
            print("Removed VIB  , ", vrm, " ,", mse, " ,", trainmse, flush=True, file=ofp)
        
            print("Removed VIB  , ", vrm, " , MSE , ", mse, " , TrainMSE ,", trainmse, flush=True)
        
        print("Overall VIB MSE , ", overallmse/float(tot), \
            " , Train MSE , ", overalltrainmse/float(traintot))
        
        for perc in [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]:
        
            train_xy, train_z, test_xy, test_z = cm.get_train_and_test_random (temp_values, vib_values, \
                df, perc)

            model = None
            if modelname == "model1":
                model = cm.build_model_GP_1 (train_xy, train_z)
            elif modelname == "model2":
                model = cm.build_model_GP_2 (train_xy, train_z)
                        
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
        
            print("Removed RND  , ", perc, " ,", mse, " ,", trainmse, flush=True, file=ofp)
        
            print("Removed random values ", perc ,"  MSE ", mse, " , TrainMSE ,", \
                trainmse, flush=True) 

