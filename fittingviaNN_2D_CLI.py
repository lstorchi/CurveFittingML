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

    for modelname in ["model1", "model2", "model3"]:

        ofp = open(basename+"_results_NN_"+modelname+".csv", "w")

        print("Type , Index , Testset MSE , Trainingset MSE", flush=True, file=ofp)
    
        df, vib_values , temp_values = cm.filterinitialset (filename)
        #plotfull3dcurve (df, vib_values, temp_values)
    
        epochs = 50
        batch_size = 50
        model = None
        history = None

        overallmse = 0.0
        overalltrainmse = 0.0
        tot = 0
        traintot = 0
        vib_values_torm = [[2, 4, 6, 8, 10, 14, 18, 22, 26, 30, 35], \
                      [3, 5, 7, 9, 12, 16, 20, 24, 28, 32, 40], \
                      [2, 3, 5, 6, 8, 9, 12, 14, 18, 20, 24, 26, 30, 32], \
                      [3, 4, 6, 7, 9, 10, 14, 16, 20, 22, 26, 28,  32, 35]]
        for vrm in vib_values_torm:
            print("Removing VIB set", str(vrm).replace(",", ";"), flush=True)
        
            train_xy, train_z, test_xy, test_z = cm.get_train_and_test_rmv (temp_values, vib_values, \
                df, vrm)
        
            if modelname == "model1":
                model = cm.build_model_NN_1()
                history = model.fit(train_xy, train_z, epochs=epochs,  batch_size=batch_size, \
                        verbose=1)
            elif modelname == "model2":
                model = cm.build_model_NN_2()
                history = model.fit(train_xy, train_z, epochs=epochs, batch_size=batch_size, \
                     verbose=1)
            elif modelname == "model3":
                model = cm.build_model_NN_3()
                history = model.fit(train_xy, train_z, epochs=epochs, batch_size=batch_size, \
                     verbose=1)
        
            z_pred = model.predict(train_xy)
            trainmse = 0.0
            cont = 0.0
            for i in range(train_z.shape[0]):
                x = train_xy[i,0]
                y = train_xy[i,1]
                z = train_z[i]
                zpred = z_pred[i]
                
                trainmse += (zpred-z)**2
                overalltrainmse += (zpred-z)**2
                traintot += 1
                cont += 1.0
        
                print("Train, %10.7f , %10.7f , %10.7f , %10.7f"%(z, y, z, zpred))
        
            trainmse = trainmse/cont
        
            z_pred = model.predict(test_xy)
            mse = 0.0
            cont = 0.0
            for i in range(test_z.shape[0]):
                x = test_xy[i,0]
                y = test_xy[i,1]
                z = test_z[i]
                zpred = z_pred[i]
                
                mse += (zpred-z)**2
                overallmse += (zpred-z)**2
                tot += 1
                cont += 1.0
            
                print("Test, %10.7f , %10.7f , %10.7f , %10.7f"%(z, y, z, zpred))
        
            mse = mse/cont
        
            print("Removed VIB  , ", str(vrm).replace(",", ";"), " ,", mse[0], " ,", trainmse[0], flush=True, file=ofp)
        
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
        
            train_xy, train_z, test_xy, test_z = \
                cm.get_train_and_test_rmt (temp_values, vib_values, \
                df, temp_torm)
        
            if modelname == "model1":
                model = cm.build_model_NN_1()
                history = model.fit(train_xy, train_z, epochs=epochs,  batch_size=batch_size, \
                        verbose=1)
            elif modelname == "model2":
                model = cm.build_model_NN_2()
                history = model.fit(train_xy, train_z, epochs=epochs, batch_size=batch_size, \
                     verbose=1)
            elif modelname == "model3":
                model = cm.build_model_NN_3()
                history = model.fit(train_xy, train_z, epochs=epochs, batch_size=batch_size, \
                     verbose=1)
        
            z_pred = model.predict(train_xy)
            trainmse = 0.0
            cont = 0.0
            for i in range(train_z.shape[0]):
                x = train_xy[i,0]
                y = train_xy[i,1]
                z = train_z[i]
                zpred = z_pred[i]
                
                trainmse += (zpred-z)**2
                overalltrainmse += (zpred-z)**2
                traintot += 1
                cont += 1.0
        
                print("Train, %10.7f , %10.7f , %10.7f , %10.7f"%(z, y, z, zpred))
        
            trainmse = trainmse/cont
        
            z_pred = model.predict(test_xy)
            mse = 0.0
            cont = 0.0
            for i in range(test_z.shape[0]):
                x = test_xy[i,0]
                y = test_xy[i,1]
                z = test_z[i]
                zpred = z_pred[i]
        
                mse += (zpred-z)**2
                overallmse += (zpred-z)**2
                tot += 1
                cont += 1.0
        
                print("Test, %10.7f , %10.7f , %10.7f , %10.7f"%(z, y, z, zpred))
        
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
        
            if modelname == "model1":
                model = cm.build_model_NN_1()
                history = model.fit(train_xy, train_z, epochs=epochs,  batch_size=batch_size, \
                        verbose=1)
            elif modelname == "model2":
                model = cm.build_model_NN_2()
                history = model.fit(train_xy, train_z, epochs=epochs, batch_size=batch_size, \
                     verbose=1)
            elif modelname == "model3":
                model = cm.build_model_NN_3()
                history = model.fit(train_xy, train_z, epochs=epochs, batch_size=batch_size, \
                     verbose=1)
        
            z_pred = model.predict(train_xy)
            trainmse = 0.0
            cont = 0.0
            for i in range(train_z.shape[0]):
                x = train_xy[i,0]
                y = train_xy[i,1]
                z = train_z[i]
                zpred = z_pred[i]
                
                trainmse += (zpred-z)**2
                overalltrainmse += (zpred-z)**2
                traintot += 1
                cont += 1.0
        
                print("Train, %10.7f , %10.7f , %10.7f , %10.7f"%(z, y, z, zpred))
        
            trainmse = trainmse/cont
        
            z_pred = model.predict(test_xy)
            mse = 0.0
            cont = 0.0
            for i in range(test_z.shape[0]):
                x = test_xy[i,0]
                y = test_xy[i,1]
                z = test_z[i]
                zpred = z_pred[i]
                
                mse += (zpred-z)**2
                overallmse += (zpred-z)**2
                tot += 1
                cont += 1.0
            
                print("Test, %10.7f , %10.7f , %10.7f , %10.7f"%(z, y, z, zpred))
        
            mse = mse/cont
        
            print("Removed VIB  , ", vrm, " ,", mse[0], " ,", trainmse[0], flush=True, file=ofp)
        
            print("Removed VIB  , ", vrm, " , MSE , ", mse, " , TrainMSE ,", trainmse, flush=True)
        
        print("Overall VIB MSE , ", overallmse/float(tot), \
            " , Train MSE , ", overalltrainmse/float(traintot))
        
        for perc in [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]:
        
            train_xy, train_z, test_xy, test_z = cm.get_train_and_test_random (temp_values, vib_values, \
                df, perc)
        
            if modelname == "model1":
                model = cm.build_model_NN_1()
                history = model.fit(train_xy, train_z, epochs=epochs,  batch_size=batch_size, \
                        verbose=1)
            elif modelname == "model2":
                model = cm.build_model_NN_2()
                history = model.fit(train_xy, train_z, epochs=epochs, batch_size=batch_size, \
                     verbose=1)
            elif modelname == "model3":
                model = cm.build_model_NN_3()
                history = model.fit(train_xy, train_z, epochs=epochs, batch_size=batch_size, \
                     verbose=1)
        
            z_pred = model.predict(train_xy)
            trainmse = 0.0
            cont = 0.0
            for i in range(train_z.shape[0]):
                x = train_xy[i,0]
                y = train_xy[i,1]
                z = train_z[i]
                zpred = z_pred[i]
            
                trainmse += (zpred-z)**2
                cont += 1.0
        
                print("Train, %10.7f , %10.7f , %10.7f , %10.7f"%(z, y, z, zpred))
        
            trainmse = trainmse/cont
        
            z_pred = model.predict(test_xy)
            mse = 0.0
            cont = 0.0
            for i in range(test_z.shape[0]):
                x = test_xy[i,0]
                y = test_xy[i,1]
                z = test_z[i]
                zpred = z_pred[i]
            
                mse += (zpred-z)**2
                cont += 1.0
                
                print("Test, %10.7f , %10.7f , %10.7f , %10.7f "%(z, y, z, zpred))
        
            mse = mse/cont
        
            print("Removed RND  , ", perc, " ,", mse[0], " ,", trainmse[0], flush=True, file=ofp)
        
            print("Removed random values ", perc ,"  MSE ", mse, " , TrainMSE ,", \
                trainmse, flush=True) 

 