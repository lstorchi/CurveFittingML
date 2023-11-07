import sys
import time

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import sklearn.gaussian_process as gp
from mpl_toolkits.mplot3d import Axes3D

import commonmodules as cm

from sklearn.preprocessing import MinMaxScaler

from tensorflow import keras
import tensorflow as tf

import tensorflow.keras.optimizers as tko
import tensorflow.keras.activations as tka
import tensorflow.keras.losses as tkl
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

from sklearn.model_selection import train_test_split
from sklearn import metrics

#################################################################################################

if __name__ == "__main__":

    filename = "N2H2_2D.xlsx"
    excf = pd.ExcelFile(filename)
    debug = False

    df1 = pd.read_excel(excf, "dv=1")
    df2 = pd.read_excel(excf, "dv=2")
    df3 = pd.read_excel(excf, "dv=3")

    x = {}
    x_s = {}
    y = {} 
    y_s = {}
    scalerx = {}
    scalery = {}
    x1map_toreal = {}
    f1set = {}
    
    x["1_v_cE"] = df1[['v', 'cE']].values
    x["1_dE_cE"] = df1[['dE', 'cE']].values
    y["1"] = np.log10(df1[["cS"]].values)
    
    x["2_v_cE"] = df2[['v', 'cE']].values
    x["2_dE_cE"] = df2[['dE', 'cE']].values
    y["2"] = np.log10(df2[["cS"]].values)
    
    x["3_v_cE"] = df3[['v', 'cE']].values
    x["3_dE_cE"] = df3[['dE', 'cE']].values
    y["3"] = np.log10(df3[["cS"]].values)
    
    xkey = ["1_v_cE", "1_dE_cE", \
            "2_v_cE", "2_dE_cE", \
            "3_v_cE", "3_dE_cE"]
    
    ykey = ["1", "2", "3"]
    
    for k in xkey:
        scalerx[k] = MinMaxScaler()
        scalerx[k].fit(x[k])
        x_s[k] = scalerx[k].transform(x[k])
    
        x1map = {}
    
        for i, vn in enumerate(x_s[k][:,0]):
            x1map[vn] = x[k][i,0]
    
        x1map_toreal[k] = x1map
        
        f1set[k] = set(x_s[k][:,0])
    
        if debug:
            for i, xs in enumerate(x_s[k]):
                print(xs, x[k][i])
    
    for k in ykey:
        scalery[k] = MinMaxScaler()
        scalery[k].fit(y[k])
        y_s[k] = scalery[k].transform(y[k])
    
        if debug:
            for i, ys in enumerate(y_s[k]):
                print(ys, y[k][i])


    modelshapes = [[2, 32, 64, 128, 32],
                   [2, 16, 32, 64, 128, 32],
                   [2, 16, 32, 64, 128, 32, 16],
                   [2, 8, 16, 32, 64, 32, 16, 8],
                    [ 8,  8,  8,  8, 8],
                    [16, 16, 16, 16, 16],
                    [32, 32, 32, 32, 32],
                    [64, 64, 64, 64, 64],
                    [128, 128, 128, 128, 128],
                    [ 8,  8,  8,  8], 
                    [16, 16, 16, 16],
                    [32, 32, 32, 32],
                    [64, 64, 64, 64],
                    [128, 128, 128, 128],
                    [ 8,  8,  8], 
                    [16, 16, 16],
                    [32, 32, 32],
                    [64, 64, 64],
                    [128, 128, 128]]
    epochs_s = [10, 20]
    batch_sizes = [2, 5, 10]
    
    print (" xK , ModelShape , BatchSize , Epochs , avg TrainMSE , avg TrainR2,  avg TestMSE ,avg TestR2 ")
    
    for xk in xkey:
        yk = xk.split("_")[0]
        f1 = xk.split("_")[1]
        f2 = xk.split("_")[2]
    
        for modelshape in modelshapes:
            for batch_size in batch_sizes:
                for epochs in epochs_s:
    
                    thefirst = True
                
                    testmses  = []
                    testr2s   = []
                    trainmses = []
                    trainr2s  = []
                
                    for x1 in f1set[xk]:
                        train_x, test_x, train_y, test_y = cm.test_train_split (0, [x1], x_s[xk], y_s[yk])
                        
                        if thefirst:
                            model = cm.buildmodel(modelshape, inputshape=2)
                            #print(model.summary())
                            history = model.fit(train_x, train_y, epochs=epochs,  batch_size=batch_size, \
                                verbose=0)
                            thefirst = False
                
                        model = cm.buildmodel(modelshape, inputshape=2)
                        history = model.fit(train_x, train_y, epochs=epochs,  batch_size=batch_size, \
                            verbose=0)
                    
                        test_x_sp = scalerx[xk].inverse_transform(test_x)
                        pred_y = model.predict(test_x, verbose=0)
                        pred_y_sb = scalery[yk].inverse_transform(pred_y)
                        test_y_sb = scalery[yk].inverse_transform(test_y)
                
                        testmse = metrics.mean_absolute_error(test_y_sb, pred_y_sb)
                        testr2 = metrics.r2_score(test_y_sb, pred_y_sb)
                        testmses.append(testmse)
                        testr2s.append(testr2)
                
                        pred_y = model.predict(train_x, verbose=0)
                        pred_y_sb = scalery[yk].inverse_transform(pred_y)
                        train_y_sb = scalery[yk].inverse_transform(train_y)
                        train_x_sp = scalerx[xk].inverse_transform(train_x)
                
                        trainmse = metrics.mean_absolute_error(train_y_sb, pred_y_sb)
                        trainr2 = metrics.r2_score(train_y_sb, pred_y_sb)
                        trainmses.append(trainmse)
                        trainr2s.append(trainr2)
                
                
                    print (xk, " , ", str(modelshape).replace(",", ";") , \
                           " , ", batch_size , \
                           " , ", epochs , \
                           " , ", np.average(trainmses), \
                           " , ", np.average(trainr2s), \
                           " , ", np.average(testmses), \
                           " , ", np.average(testr2s), flush=True)
    
       
        
        