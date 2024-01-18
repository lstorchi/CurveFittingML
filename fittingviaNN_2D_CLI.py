import numpy as np
import pandas as pd

from tensorflow import keras
import tensorflow as tf

from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

import commonmodules as cm
import sys

###############################################################################

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd, file = sys.stderr)
    # Print New Line on Complete
    if iteration == total: 
        print(file = sys.stderr)

###############################################################################

def build_vsettorm (vlist):

    vset_torm = []

    vtoremove = []
    for i in range(1,len(vlist),2):
        vtoremove.append(vlist[i])
    vset_torm.append(vtoremove)

    vtoremove = []
    for i in range(0,len(vlist),2):
        vtoremove.append(vlist[i])
    vset_torm.append(vtoremove)

    vtoremove = []
    for i in range(1,len(vlist),3):
        vtoremove.append(vlist[i])
        if (i+1 < len(vlist)):
            vtoremove.append(vlist[i+1])
    vset_torm.append(vtoremove)

    vtoremove = []
    for i in range(0,len(vlist),3):
        vtoremove.append(vlist[i])
        if (i+1 < len(vlist)):
            vtoremove.append(vlist[i+1])
    vset_torm.append(vtoremove)

    vtoremove = []
    for i in range(1,len(vlist),4):
        vtoremove.append(vlist[i])
        if (i+1 < len(vlist)):
            vtoremove.append(vlist[i+1])
        if (i+2 < len(vlist)):
            vtoremove.append(vlist[i+2])
    vset_torm.append(vtoremove)

    vtoremove = []
    for i in range(0,len(vlist),4):
        vtoremove.append(vlist[i])
        if (i+1 < len(vlist)):
            vtoremove.append(vlist[i+1])
        if (i+2 < len(vlist)):
            vtoremove.append(vlist[i+2])
    vset_torm.append(vtoremove)

    return vset_torm

###############################################################################

def read_excel_file_and_norm (filename, debug=False):

    excf = pd.ExcelFile(filename)

    if debug:
        print(excf.sheet_names)

    df1 = pd.read_excel(excf, "dv=1")
    df2 = pd.read_excel(excf, "dv=2")
    df3 = pd.read_excel(excf, "dv=3")

    if debug:
        print(df1.columns)
        print(df2.columns)
        print(df3.columns)

    x = {}
    x_s = {}
    y = {} 
    y_s = {}
    scalerx = {}
    scalery = {}
    x1map_toreal = {}
    f1set = {}
    f1list = {}

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
        lista = list(set(x_s[k][:,0]))
        lista.sort(reverse=False)
        f1list[k] = lista

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

    return xkey, ykey, x_s, y_s, scalerx, scalery, x1map_toreal, f1set, f1list

###############################################################################

def read_excel_file_and_norm_tfile (filename, debug=False):

    excf = pd.ExcelFile(filename)

    if debug:
        print(excf.sheet_names)

    df1 = pd.read_excel(excf, "dv=1")
    df2 = pd.read_excel(excf, "dv=2")
    df3 = pd.read_excel(excf, "dv=3")

    if debug:
        print(df1.columns)
        print(df2.columns)
        print(df3.columns)

    x = {}
    x_s = {}
    y = {} 
    y_s = {}
    scalerx = {}
    scalery = {}
    x1map_toreal = {}
    f1set = {}
    f1list = {}

    x["1_v_T"] = df1[['v', 'T']].values
    y["1"] = np.log10(df1[["RateC"]].values)

    x["2_v_T"] = df2[['v', 'T']].values
    y["2"] = np.log10(df2[["RateC"]].values)

    x["3_v_T"] = df3[['v', 'T']].values
    y["3"] = np.log10(df3[["RateC"]].values)

    xkey = ["1_v_T", \
            "2_v_T", \
            "3_v_T"]

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
        lista = list(set(x_s[k][:,0]))
        lista.sort(reverse=False)
        f1list[k] = lista

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

    return xkey, ykey, x_s, y_s, scalerx, scalery, x1map_toreal, f1set, f1list

###############################################################################

if __name__ == "__main__":

    #cE = Collision Energy
    #dE = Delta E 
    #cS = Cross Section
    dumppredictions = True

    np.random.seed(812)
    keras.utils.set_random_seed(812)
    tf.config.experimental.enable_op_determinism()

    # first file
    filename = "N2H2_2D_VT_process.xlsx"
    xkey, ykey, x_s, y_s, scalerx, scalery, x1map_toreal, f1set, f1list = \
        read_excel_file_and_norm (filename)

    # second file
    #filename = "N2H2_2D_VT_process_using_T.xlsx"
    #xkey, ykey, x_s, y_s, scalerx, scalery, x1map_toreal, f1set, f1list = \
    #    read_excel_file_and_norm_tfile (filename)

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
    epochs_s = [10, 20, 30, 60, 80, 90, 100]
    batch_sizes = [2, 5, 10]
    
    print (" xK , split , ModelShape , BatchSize , Epochs , avg TrainMSE , avg TrainR2,  avg TestMSE ,avg TestR2 ")

    totvalues = len(xkey) * len(modelshapes) * len(batch_sizes) * len(epochs_s)
    counter = 0

    for xk in xkey:
        if len(xk.split("_")) != 3:
            print("Error: xk.split('_') != 3")
            exit(1)

        yk = xk.split("_")[0]
        f1 = xk.split("_")[1]
        f2 = xk.split("_")[2]

        for modelshape in modelshapes:
            for batch_size in batch_sizes:
                for epochs in epochs_s:

                    counter += 1
                    print("vsplit: ", counter, "/", totvalues, flush=True, file=sys.stderr)
                    
                    testmses  = []
                    testr2s   = []
                    trainmses = []
                    trainr2s  = []
                
                    for x1 in f1set[xk]:
                        train_x, test_x, train_y, test_y = cm.test_train_split (0, [x1], \
                                                                                x_s[xk], y_s[yk])
                
                        model = cm.buildmodel(modelshape, inputshape=2)
                        history = model.fit(train_x, train_y, epochs=epochs,  batch_size=batch_size, \
                            verbose=0)
                        
                        fp = None
                        if dumppredictions:
                            fp = open("model_"+str(counter)+"_"+\
                                    "xk_"+xk+"_"+\
                                    "yk_"+yk+"_"+\
                                    "rmset_"+str(x1)+"_"+\
                                    "_predictions.csv", "w")
                            
                            print ("Set,"+f1+","+f2+",y,pred_y", file=fp)
                    
                        test_x_sp = scalerx[xk].inverse_transform(test_x)
                        pred_y = model.predict(test_x, verbose=0)
                        pred_y_sb = scalery[yk].inverse_transform(pred_y)
                        test_y_sb = scalery[yk].inverse_transform(test_y)

                        if dumppredictions:
                            for i, x1 in enumerate(test_x_sp):
                                if len(x1) != 2:
                                    print("Error: len(x1) != 2")
                                    exit(1)

                                print("Test,",
                                      x1[0], ",",\
                                      x1[1], ",",\
                                      test_y_sb[i][0],",", \
                                      pred_y_sb[i][0],\
                                    file=fp)
                
                        testmse = metrics.mean_absolute_error(test_y_sb, pred_y_sb)
                        testr2 = metrics.r2_score(test_y_sb, pred_y_sb)
                        testmses.append(testmse)
                        testr2s.append(testr2)
                
                        pred_y = model.predict(train_x, verbose=0)
                        pred_y_sb = scalery[yk].inverse_transform(pred_y)
                        train_y_sb = scalery[yk].inverse_transform(train_y)
                        train_x_sp = scalerx[xk].inverse_transform(train_x)
                
                        if dumppredictions:
                            for i, x1 in enumerate(train_x_sp):
                                if len(x1) != 2:
                                    print("Error: len(x1) != 2")
                                    exit(1)

                                print("Train,",
                                      x1[0], ",",\
                                      x1[1], ",",\
                                      train_y_sb[i][0],",", \
                                      pred_y_sb[i][0],\
                                    file=fp)
                        trainmse = metrics.mean_absolute_error(train_y_sb, pred_y_sb)
                        trainr2 = metrics.r2_score(train_y_sb, pred_y_sb)
                        trainmses.append(trainmse)
                        trainr2s.append(trainr2)

                        printProgressBar (len(testmses), len(f1set[xk]), \
                                           prefix = 'Progress:', \
                                            suffix = 'Complete', length = 50)
                
                
                    print (xk, " , vsplit , ", str(modelshape).replace(",", ";") , \
                           " , ", batch_size , \
                           " , ", epochs , \
                           " , ", np.average(trainmses), \
                           " , ", np.average(trainr2s), \
                           " , ", np.average(testmses), \
                           " , ", np.average(testr2s), flush=True)

    counter = 0

    for xk in xkey:
        yk = xk.split("_")[0]
        f1 = xk.split("_")[1]
        f2 = xk.split("_")[2]

        vsettorm = build_vsettorm (f1list[xk])    

        for modelshape in modelshapes:
            for batch_size in batch_sizes:
                for epochs in epochs_s:  
                                     
                    testmses  = []
                    testr2s   = []
                    trainmses = []
                    trainr2s  = []

                    counter += 1
                    print("vsetsplit: ", counter, "/", totvalues, flush=True, file=sys.stderr)

                    for vset in vsettorm:

                        train_x, test_x, train_y, test_y = cm.test_train_split (0, vset, \
                                                                                x_s[xk], y_s[yk])
                
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

                        printProgressBar (len(testmses), len(vsettorm),\
                                           prefix = 'Progress:',\
                                              suffix = 'Complete', length = 50)

                    print (xk, " , vsetsplit , ", str(modelshape).replace(",", ";") , \
                           " , ", batch_size , \
                           " , ", epochs , \
                           " , ", np.average(trainmses), \
                           " , ", np.average(trainr2s), \
                           " , ", np.average(testmses), \
                           " , ", np.average(testr2s), flush=True)
