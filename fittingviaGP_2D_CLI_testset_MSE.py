import os
import math

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import sklearn.gaussian_process as gp
from mpl_toolkits.mplot3d import Axes3D

##########################################################################################################

def filterinitialset_rmnan (data,  coltorm, headername, \
    factor = 1.0, normalize = False):

    dfin = data.parse(sheetname)
    dfin = dfin.drop(columns=[coltorm])
    
    dfdict = {}
    
    tempvalues = list(dfin.columns[1:])
    vibvalues = list(dfin[headername].values)

    min = float("inf")
    max = float("-inf")
    for c in dfin.columns:
        dfdict[c] = []

        if c == headername:
            dfdict[c] = list(dfin[c].values)
        else:
            for v in dfin[c].values:
                val = math.log10(factor*v)
                if val > max:
                    max = val
                if val < min:
                    min = val

    for c in dfin.columns:
        if c != headername:
            for v in dfin[c].values:
                val = math.log10(factor*v)
                valp = (val - min) / (max - min)
                if normalize:
                    dfdict[c].append(valp)
                else:
                    dfdict[c].append(val)

    df = pd.DataFrame.from_dict(dfdict)

    vib_torm = []
    for v in df[headername]:
        if np.isnan(df[df[headername] == v].values[0][1:-1].astype(float)).all():
            vib_torm.append(v)

    for v in vib_torm:
        vibvalues.remove(v)
        df = df.drop(df[df[headername] == v].index)

    return df, vibvalues, tempvalues

##########################################################################################################3

def fitusingscikitl (train_x, train_y, nuval=5.0/2.0):

    #kernel = gp.kernels.ConstantKernel(1.0, (1e-5, 1e5))* gp.kernels.RBF(length_scale=1)
    kernel = 1.0 * gp.kernels.Matern(length_scale=1.0, nu=nuval)
    model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50, \
        normalize_y=False)
    print("Start training ")
    model.fit(train_x, train_y)
    print("Done ")

    return model

##########################################################################################################

def get_train_and_test_rmv (temp_values, vib_values, df, \
    removevibs=[], normalize=False):

    maxt = max(temp_values)
    mint = min(temp_values)

    minv = min(vib_values)
    maxv = max(vib_values)

    train_xy = []
    train_z = []

    test_xy = []
    test_z = []

    maxz = float("-inf")
    minz = float("+inf")

    for tidx, t in enumerate(temp_values):
        for vidx, v in enumerate(vib_values):
            zval = df[t].values[vidx]

            if zval < minz:
                minz = zval
            elif zval > maxz:
                maxz = zval

    for t in temp_values:
        tnorm = (t - mint)/(maxt - mint)

        for vidx, v in enumerate(vib_values):
            if v not in removevibs:
                vnorm  = (v - minv)/(maxv - minv)
                train_xy.append([tnorm, vnorm])
        
                z = df[t].values[vidx]
                if normalize:
                    znorm = (z - minz)/(maxz - minz)
                    train_z.append(znorm)
                else:
                    train_z.append(z)
            else:
                vnorm  = (v - minv)/(maxv - minv)
                test_xy.append([tnorm, vnorm])

                z = df[t].values[vidx]
                if normalize:
                    znorm = (z - minz)/(maxz - minz)
                    test_z.append(znorm)
                else:
                    test_z.append(z)

    train_xy = np.asarray(train_xy)
    train_z = np.asarray(train_z)

    test_xy = np.asarray(test_xy)
    test_z = np.asarray(test_z)

    return train_xy, train_z, test_xy, test_z

##########################################################################################################

if __name__  == "__main__":

    filename = "COCO_touse.xlsx"
    USEV = True

    headername = "vibrational level v\Temperature(K)"
    coltorm = "DE(cm-1)"

    if USEV:
        headername = "vibrational level v\Temperature(K)"
        coltorm = "DE(cm-1)"
    else:
        coltorm = "vibrational level v\Temperature(K)"
        headername = "DE(cm-1)"

    nuvals = [1.0, 1.0/2.0, 3.0/2.0, 4.0/3.0, 2.0, 5.0/2.0, 7.0/2.0, 7.0/3.0]
    selectedsheets = ["1","2","3","4","5","6","7","8"]

    data = pd.ExcelFile(filename)
    for sheetname in data.sheet_names:
        if sheetname in selectedsheets:
        
            print("Using sheet: ", sheetname, flush=True)

            df, vib_values , temp_values = filterinitialset_rmnan (data, coltorm, \
                                                                   sheetname, headername)
        
            maxt = max(temp_values)
            mint = min(temp_values)
        
            minv = min(vib_values)
            maxv = max(vib_values)
            
            for nuval in nuvals:
                global_cont_train = 0.0
                global_trainmse = 0.0

                global_cont_test = 0.0
                global_testmse = 0.0

                for vrm in df[headername]:
                    print("To remove: ", vrm)
                    vib_torm = []
                    vib_torm.append(vrm)
            
                    train_xy, train_z, test_xy, test_z = get_train_and_test_rmv (temp_values, vib_values, \
                        df, vib_torm)
                    
                    model = fitusingscikitl (train_xy, train_z, nuval)
                    
                    z_pred, std = model.predict(train_xy, return_std=True)
                    trainmse = 0.0
                    cont = 0.0
                    for i in range(train_z.shape[0]):
                        x = train_xy[i,0]
                        t = int(x*(maxt - mint)+mint)
                        y = train_xy[i,1]
                        v = int(y*(maxv - minv)+minv)
                        z = train_z[i]
                        zpred = z_pred[i]
                        zstd = std[i]
                        
                        trainmse += (zpred-z)**2
                        cont += 1.0
                                        
                    trainmse = trainmse/cont
                    global_trainmse += trainmse
                    global_cont_train += 1.0

                    print(sheetname, nuval, " Train MSE : %f %e"%(vrm, trainmse), flush=True)
                    
                    z_pred, std = model.predict(test_xy, return_std=True)
                    
                    ofp = open(sheetname+"_"+str(nuval)+"_"+str(vrm)+"_results.csv", "w")
                    print ("T , v , Zpred, Zstd ", file=ofp , flush=True)
                    testmse = 0.0
                    cont = 0.0
                    for i in range(test_z.shape[0]):
                        x = test_xy[i,0]
                        t = int(x*(maxt - mint)+mint)
                        y = test_xy[i,1]
                        v = int(y*(maxv - minv)+minv)
                        z = test_z[i]
                        zpred = z_pred[i]
                        zstd = std[i]
            
                        testmse += (zpred-z)**2
                        cont += 1.0
                    
                        print("%10.7f , %10.7f , %10.7f , %10.7f"%(t, v, zpred, zstd), file=ofp , \
                              flush=True)
            
                    testmse = testmse/cont
                    global_testmse += testmse
                    global_cont_test += 1.0

                    print(sheetname, nuval, " Test MSE : %f %e"%(vrm, testmse), flush=True)
                            
                    ofp.close()

                print(sheetname, nuval, " Average Train MSE : %e"%(\
                    global_trainmse/global_cont_train), flush=True)
                print(sheetname, nuval, " Average Test MSE : %e"%(\
                    global_testmse/global_cont_test), flush=True)

 
