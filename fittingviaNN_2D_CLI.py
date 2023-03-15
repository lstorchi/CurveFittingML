import sys
import time

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import sklearn.gaussian_process as gp
from mpl_toolkits.mplot3d import Axes3D

import commonmodules as cm

#################################################################################################

def get_mseresults (initialstring, ofp, model, train_xy, train_z, test_xy, test_z, verbose=False):

    overallmse = 0.0 
    denorm_overallmse = 0.0 
    overalltrainmse = 0.0 
    denorm_overalltrainmse = 0.0

    tot = 0
    traintot = 0

    z_pred = model.predict(train_xy)
    
    trainmse = 0.0
    denorm_trainmse = 0.0
    cont = 0.0

    for i in range(train_z.shape[0]):
        x = train_xy[i,0]
        y = train_xy[i,1]

        z = train_z[i]
        denorm_z = (z * (maxvalue - minvalue))+minvalue

        zpred = z_pred[i]
        denorm_zpred = (zpred * (maxvalue - minvalue))+minvalue
        
        trainmse += (zpred-z)**2
        denorm_trainmse += (denorm_zpred-denorm_z)**2

        overalltrainmse += (zpred-z)**2
        denorm_overalltrainmse += (denorm_zpred-denorm_z)**2

        traintot += 1
        cont += 1.0
    
        if verbose:
            print("Train, %10.7f , %10.7f , %10.7f , %10.7f"%(z, y, z, zpred))
    
    trainmse = trainmse/cont
    denorm_trainmse = denorm_trainmse/cont
    
    z_pred = model.predict(test_xy)
    mse = 0.0
    denorm_mse = 0.0
    cont = 0.0
    for i in range(test_z.shape[0]):
        x = test_xy[i,0]
        y = test_xy[i,1]

        z = test_z[i]
        denorm_z = (z * (maxvalue - minvalue))+minvalue

        zpred = z_pred[i]
        denorm_zpred = (zpred * (maxvalue - minvalue))+minvalue

        mse += (zpred-z)**2
        denorm_mse += (denorm_zpred-denorm_z)**2

        overallmse += (zpred-z)**2
        tot += 1
        cont += 1.0
    
        if verbose:
            print("Test, %10.7f , %10.7f , %10.7f , %10.7f"%(x, y, z, zpred))
    
    mse = mse/cont
    denorm_mse = denorm_mse/cont
    
    print(initialstring, " ,", mse[0], " ,", trainmse[0], " ,", denorm_mse[0] , \
        " ,",  denorm_trainmse[0], flush=True, file=ofp)
    
    if verbose:
        print(initialstring, " , MSE , ", mse[0], " , TrainMSE ,", trainmse[0], \
            " , Denorm. MSE , ", denorm_mse[0], " , Denorm. TrainMSE ,", \
                denorm_trainmse[0], flush=True)

    return overallmse, denorm_overallmse, overalltrainmse, \
        denorm_overalltrainmse, tot, traintot

#################################################################################################

if __name__ == "__main__":

    filename = "testdv1.xlsx"

    setofv = 0
    if (len(sys.argv) == 3):
        filename = sys.argv[1]
        setofv = int(sys.argv[2])

    basename = filename.split(".")[0]

    for modelname in ["model1", "model2", "model3"]:

        ofp = open(basename+"_results_NN_"+modelname+".csv", "w")

        print("Type , Index , Testset MSE , Trainingset MSE , ", \
            "Denorm. Testset MSE , Denorm. Trainingset MSE ", \
                flush=True, file=ofp)
    
        df, vib_values , temp_values, minvalue, maxvalue = cm.filterinitialset (filename)
        #plotfull3dcurve (df, vib_values, temp_values)
    
        epochs = 50
        batch_size = 50
        model = None
        history = None

        overallmse = 0.0
        overalltrainmse = 0.0
        denorm_overalltrainmse = 0.0
        denorm_overallmse = 0.0
        tot = 0
        traintot = 0
        for vrm in vib_values:
            vib_torm = [vrm]
            print("Removing VIB ", vrm, flush=True)
        
            train_xy, train_z, test_xy, test_z = cm.get_train_and_test_rmv (temp_values, vib_values, \
                df, vib_torm)
        
            st = time.time()
            stp = time.process_time()
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
            etp = time.process_time()
            et = time.time()

            elapsed_time = et - st
            res = etp - stp
            print('Execution time: ', elapsed_time, ' seconds', flush = True)
            print('CPU Execution time: ', res, ' seconds')

            initialstring = "Removed VIB  , " + str(vrm)
            l_overallmse, l_denorm_overallmse, \
                l_overalltrainmse, l_denorm_overalltrainmse, \
                    l_tot,  l_traintot = get_mseresults (initialstring, ofp, model, \
                    train_xy, train_z, test_xy, test_z)

            overallmse += l_overallmse
            overalltrainmse += l_overalltrainmse
            tot += l_tot
            traintot += l_traintot

        print("Overall VIB MSE , ", overallmse/float(tot), \
            ", Train MSE , ", overalltrainmse/float(traintot), \
            ", Denorm. MSE , ", denorm_overallmse/float(tot), \
            ", Denorm. Train MSE , ", denorm_overalltrainmse/float(traintot))

        overallmse = 0.0
        overalltrainmse = 0.0
        denorm_overalltrainmse = 0.0
        denorm_overallmse = 0.0
        tot = 0
        traintot = 0

        vib_values_torm = []

        if setofv == 1:
            # for dv=1
            vib_values_torm = [[2, 4, 6, 8, 10, 14, 18, 22, 26, 30, 35], \
                           [1, 3, 5, 7, 9, 12, 16, 20, 24, 28, 32, 40], \
                           [2, 3, 5, 6, 8, 9, 12, 14, 18, 20, 24, 26, 30, 32], \
                           [1, 2, 4, 5, 7, 8, 10, 12, 16, 18, 22, 24, 28, 30, 35, 40]]
        elif setofv == 2:
            # for dv=2
            vib_values_torm = [[2, 4, 6, 8, 10, 14, 18, 22, 26, 30, 35], \
                           [3, 5, 7, 9, 12, 16, 20, 24, 28, 32, 40], \
                           [2, 3, 5, 6, 8, 9, 12, 14, 18, 20, 24, 26, 30, 32, 40], \
                           [3, 4, 6, 7, 9, 10, 14, 16, 20, 22, 26, 28, 32, 35]]
        elif setofv == 3:
            # for dv=3
            vib_values_torm = [[2, 4, 6, 8, 10, 14, 18, 22, 26, 30, 35], \
                           [3, 5, 7, 9, 12, 16, 20, 24, 28, 32], \
                           [2, 3, 5, 6, 8, 9, 12, 14, 18, 20, 24, 26, 30, 32], \
                           [3, 4, 6, 7, 9, 10, 14, 16, 20, 22, 26, 28, 32, 35]]
        for vrm in vib_values_torm:
            print("Removing VIB set", str(vrm).replace(",", ";"), flush=True)
        
            train_xy, train_z, test_xy, test_z = cm.get_train_and_test_rmv (temp_values, vib_values, \
                df, vrm)
            
            st = time.time()
            stp = time.process_time()
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
            etp = time.process_time()
            et = time.time()

            elapsed_time = et - st
            res = etp - stp
            print('Execution time: ', elapsed_time, ' seconds', flush = True)
            print('CPU Execution time: ', res, ' seconds')     

            initialstring = "Removed VIB  , " + str(vrm).replace(",", ";")
            l_overallmse, l_denorm_overallmse, \
                l_overalltrainmse, l_denorm_overalltrainmse, \
                    l_tot,  l_traintot = get_mseresults (initialstring, ofp, model, \
                    train_xy, train_z, test_xy, test_z)

            overallmse += l_overallmse
            overalltrainmse += l_overalltrainmse
            tot += l_tot
            traintot += l_traintot

        print("Overall VIB MSE , ", overallmse/float(tot), \
            ", Train MSE , ", overalltrainmse/float(traintot), \
            ", Denorm. MSE , ", denorm_overallmse/float(tot), \
            ", Denorm. Train MSE , ", denorm_overalltrainmse/float(traintot))

        overallmse = 0.0
        overalltrainmse = 0.0
        denorm_overalltrainmse = 0.0
        denorm_overallmse = 0.0
        tot = 0
        traintot = 0
        for trm in temp_values:
            print("Removing TEMP ", trm, flush=True)
            temp_torm = [trm]
        
            train_xy, train_z, test_xy, test_z = \
                cm.get_train_and_test_rmt (temp_values, vib_values, \
                df, temp_torm)

            st = time.time()
            stp = time.process_time()     
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
            etp = time.process_time()
            et = time.time()

            elapsed_time = et - st
            res = etp - stp
            print('Execution time: ', elapsed_time, ' seconds', flush = True)
            print('CPU Execution time: ', res, ' seconds')     

            initialstring = "Removed TEMP  , " + str(trm)
            l_overallmse, l_denorm_overallmse, \
                l_overalltrainmse, l_denorm_overalltrainmse, \
                    l_tot,  l_traintot = get_mseresults (initialstring, ofp, model, \
                    train_xy, train_z, test_xy, test_z)

            overallmse += l_overallmse
            overalltrainmse += l_overalltrainmse
            tot += l_tot
            traintot += l_traintot

        print("Overall TEMP MSE , ", overallmse/float(tot), \
            ", Train MSE , ", overalltrainmse/float(traintot), \
            ", Denorm. MSE , ", denorm_overallmse/float(tot), \
            ", Denorm. Train MSE , ", denorm_overalltrainmse/float(traintot))
        
        perclist = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
        for perc in perclist:
        
            train_xy, train_z, test_xy, test_z = cm.get_train_and_test_random (temp_values, vib_values, \
                df, perc)
            
            st = time.time()
            stp = time.process_time()     
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
            etp = time.process_time()
            et = time.time()

            elapsed_time = et - st
            res = etp - stp
            print('Execution time: ', elapsed_time, ' seconds', flush = True)
            print('CPU Execution time: ', res, ' seconds')     

            initialstring = "Removed RND  , " + str(perc)
            l_overallmse, l_denorm_overallmse, \
                l_overalltrainmse, l_denorm_overalltrainmse, \
                    l_tot,  l_traintot = get_mseresults (initialstring, ofp, model, \
                    train_xy, train_z, test_xy, test_z)