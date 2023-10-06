import os
import math

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import sklearn.gaussian_process as gp
from mpl_toolkits.mplot3d import Axes3D

##########################################################################################################

def filterinitialset (data, coltorm, sheetname, headername, \
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

    return df, vibvalues, tempvalues

##############################################################################

def plotfull3dcurve (df, vib_values, temp_values):

    y = []
    x = []
    for t in temp_values:
        for idx in range(len(vib_values)):
            x.append([float(t), float(vib_values[idx])])
            y.append(df[t].values[idx])

    X = np.array(x)
    Y = np.array(y)

    xdim = len(temp_values)
    ydim = len(vib_values)

    Xp = np.zeros((xdim, ydim), dtype=float)
    Yp = np.zeros((xdim, ydim), dtype=float)
    Zp = np.zeros((xdim, ydim), dtype=float)
    for xidx in range(xdim):
        t = temp_values[xidx]
        for yidx in range(ydim):
            v =  vib_values[yidx]
            Xp[xidx, yidx] = float(t)
            Yp[xidx, yidx] = float(v)
            Zp[xidx, yidx] = df[t].values[yidx]

    #fig = plt.figure(figsize=(10,8))
    fig = plt.figure(figsize=plt.figaspect(2.))
    plt.gcf().set_size_inches(40, 30)
    ax = fig.add_subplot(2,1,1, projection='3d')
    surf = ax.plot_surface(Xp, Yp, Zp, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=False)
    plt.show()

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

    #headername = "vibrational level v\Temperature(K)"
    #coltorm = "DE(cm-1)"
    coltorm = "vibrational level v\Temperature(K)"
    headername = "DE(cm-1)"

    nuvals = [1.0, 1.0/2.0, 3.0/2.0, 4.0/3.0, 2.0, 5.0/2.0, 7.0/2.0, 7.0/3.0]
    #nuvals = [5.0/2.0]
    selectedseeh = ["3","7","8"]

    data = pd.ExcelFile(filename)
    for sheetname in data.sheet_names:
        if sheetname in sheetname:
        
            print("Using sheet: ", sheetname, flush=True)
        
            df, vib_values , temp_values = filterinitialset (data, coltorm, sheetname, headername)
        
            maxt = max(temp_values)
            mint = min(temp_values)
        
            minv = min(vib_values)
            maxv = max(vib_values)
        
            vib_torm = []
            for v in df[headername]:
                #print(type(df[df[headername] == v].values[0]))
                #print(df[df[headername] == v].values[0][1:-1])
                if np.isnan(df[df[headername] == v].values[0][1:-1].astype(float)).all():
                    vib_torm.append(v)
        
            #git plotfull3dcurve (df, vib_values, temp_values)
        
            train_xy, train_z, test_xy, test_z = get_train_and_test_rmv (temp_values, vib_values, \
                df, vib_torm)
            
            for nuval in nuvals:
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
                
                    print(sheetname, " Train, %10.7f , %10.7f , %10.7f , %10.7f , %10.7f"%(t, v, z, zpred, zstd), flush=True)
                
                trainmse = trainmse/cont
                print(sheetname, "Nu %6.3f Train MSE : %12.8e"%(nuval, trainmse), flush=True)
                
                z_pred, std = model.predict(test_xy, return_std=True)
            
                ofp = open(sheetname+"_"+str(nuval)+"_results.csv", "w")
        
                avgstd = 0.0
            
                print ("T , v , Zpred, Zstd ", file=ofp , flush=True)
                #print ("T , DE , Zpred, Zstd ", file=ofp , flush=True)
                for i in range(test_z.shape[0]):
                    x = test_xy[i,0]
                    t = int(x*(maxt - mint)+mint)
                    y = test_xy[i,1]
                    v = int(y*(maxv - minv)+minv)
                    zpred = z_pred[i]
                    zstd = std[i]
                    
                    avgstd += zstd
            
                    print("%10.7f , %10.7f , %10.7f , %10.7f"%(t, v, zpred, zstd), file=ofp , \
                          flush=True)
                
                print ("Nu %6.3f Average std: "%(nuval), avgstd/test_z.shape[0], flush=True)  
        
                ofp.close()
 
