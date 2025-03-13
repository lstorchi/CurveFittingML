import random 
import pickle

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from tensorflow import keras
import tensorflow as tf
import keras 

import keras.optimizers as tko
import keras.activations as tka
import keras.losses as tkl
from keras.layers import Input, Dense
from keras.models import Model

from sklearn.model_selection import train_test_split

from sklearn import metrics

import commonmodules as cm
import time

FIXEDSEED = True

#######################################################################

def build_perc_split (x_s, y_s, modelshape, batch_size, epochs, \
                      modelfname="", verbose=False):

    ofp = None 
    if modelfname != "":
        ofp = open(modelfname, "w")

    avgr2_test = 0.0
    avgr2_train = 0.0
    avgmse_test = 0.0
    avgmse_train = 0.0

    num = 0.0

    if verbose:
        print (" Perc. Split , Test MSE , Test R2 , Train MSE , Train R2")
    if modelfname != "":
        print (" Perc. Split , Test MSE , Test R2 , Train MSE , Train R2", file=ofp)

    for perc in [0.05, 0.10, 0.25, 0.30, 0.50]:

        if FIXEDSEED:
            # to fix seed
            np.random.seed(42)
            tf.random.set_seed(42)
            random.seed(42)

        train_x, test_x, train_y, test_y = train_test_split(x_s, y_s, \
                    test_size=perc, random_state=42)
        

        modelshape = [32, 32, 32, 32]
        epochs = 10
        batch_size = 50

        model = cm.buildmodel(modelshape)
        history = model.fit(train_x, train_y, epochs=epochs,  batch_size=batch_size, \
            verbose=0)

        pred_y = model.predict(test_x, verbose=0)
        testmse = metrics.mean_absolute_error(test_y, pred_y)
        testr2 = metrics.r2_score(test_y, pred_y)

        avgr2_test += testr2
        avgmse_test += testmse

        pred_y = model.predict(train_x, verbose=0)
        trainmse = metrics.mean_absolute_error(train_y, pred_y)
        trainr2 = metrics.r2_score(train_y, pred_y)

        avgr2_train += trainr2
        avgmse_train += trainmse
        
        num += 1.0

        if verbose:
            print("%5.2f , %10.6f , %10.6f , %10.6f , %10.6f"%(perc, testmse, testr2, \
                                                       trainmse,  trainr2))
        if modelfname != "":
            print("%5.2f , %10.6f , %10.6f , %10.6f , %10.6f"%(perc, testmse, testr2, \
                                                       trainmse,  trainr2), file=ofp)
            
    if modelfname != "":
        ofp.close()

    return avgr2_train/num, avgmse_train/num, avgr2_test/num, avgmse_test/num

####################################################################################

def build_w_split (x_s, y_s, wset, modelshape, batch_size, epochs, \
                   modelfname="", verbose=False):
    ofp = None
    if modelfname != "":
        ofp = open("wremoved.csv", "w")

    thefirst = True

    avgr2_test = 0.0
    avgr2_train = 0.0
    avgmse_test = 0.0
    avgmse_train = 0.0

    num = 0.0

    if verbose:
        print (" w Removed , Test MSE , Test R2 , Train MSE , Train R2")
    if modelfname != "":
        print (" w Removed , Test MSE , Test R2 , Train MSE , Train R2", file=ofp)

    for w in wset:

        if FIXEDSEED:
            # to fix seed
            np.random.seed(42)
            tf.random.set_seed(42)
            random.seed(42)

        train_x, test_x, train_y, test_y = cm.test_train_split (1, [w], x_s, y_s)

        if thefirst:
            model = cm.buildmodel(modelshape)
            history = model.fit(train_x, train_y, epochs=epochs,  batch_size=batch_size, \
                verbose=0)
            thefirst = False

        model = cm.buildmodel(modelshape)
        history = model.fit(train_x, train_y, epochs=epochs,  batch_size=batch_size, \
            verbose=0)

        pred_y = model.predict(test_x, verbose=0)
        testmse = metrics.mean_absolute_error(test_y, pred_y)
        testr2 = metrics.r2_score(test_y, pred_y)

        avgr2_test += testr2
        avgmse_test += testmse

        pred_y = model.predict(train_x, verbose=0)
        trainmse = metrics.mean_absolute_error(train_y, pred_y)
        trainr2 = metrics.r2_score(train_y, pred_y)

        avgr2_train += trainr2
        avgmse_train += trainmse

        num += 1.0

        if verbose:
            print("%5.2f , %10.6f , %10.6f , %10.6f , %10.6f"%(w, testmse, testr2, \
                                                        trainmse,  trainr2))
        if modelfname != "":
            print("%5.2f , %10.6f , %10.6f , %10.6f , %10.6f"%(w, testmse, testr2, \
                                                        trainmse,  trainr2), file=ofp)
    if modelfname != "":  
        ofp.close()

    return avgr2_train/num, avgmse_train/num, avgr2_test/num, avgmse_test/num

#######################################################################

def build_t_split (x_s, y_s, tset, modelshape, batch_size, epochs, \
                   modelfname="", verbose=False):
    ofp = None
    if modelfname != "":
        ofp = open("tremoved.csv", "w")

    thefirst = True

    avgr2_test = 0.0
    avgr2_train = 0.0
    avgmse_test = 0.0
    avgmse_train = 0.0

    num = 0.0

    if verbose:
        print (" T Removed , Test MSE , Test R2 , Train MSE , Train R2")
    if modelfname != "":
        print (" T Removed , Test MSE , Test R2 , Train MSE , Train R2", file=ofp)
    for t in tset:

        if FIXEDSEED:
            # to fix seed
            np.random.seed(42)
            tf.random.set_seed(42)
            random.seed(42)

        train_x, test_x, train_y, test_y = cm.test_train_split (2, [t], x_s, y_s)

        modelshape = [32, 32, 32, 32]
        epochs = 10
        batch_size = 50

        if thefirst:
            model = cm.buildmodel(modelshape)
            history = model.fit(train_x, train_y, epochs=epochs,  batch_size=batch_size, \
                verbose=0)
            thefirst = False

        model = cm.buildmodel(modelshape)
        history = model.fit(train_x, train_y, epochs=epochs,  batch_size=batch_size, \
            verbose=0)

        pred_y = model.predict(test_x, verbose=0)
        testmse = metrics.mean_absolute_error(test_y, pred_y)
        testr2 = metrics.r2_score(test_y, pred_y)
        
        avgr2_test += testr2
        avgmse_test += testmse
    
        pred_y = model.predict(train_x, verbose=0)
        trainmse = metrics.mean_absolute_error(train_y, pred_y)
        trainr2 = metrics.r2_score(train_y, pred_y)

        avgr2_train += trainr2
        avgmse_train += trainmse

        num += 1.0

        if verbose:
            print("%5.2f , %10.6f , %10.6f , %10.6f , %10.6f"%(t, testmse, testr2, \
                                                        trainmse,  trainr2))
        if modelfname != "":
            print("%5.2f , %10.6f , %10.6f , %10.6f , %10.6f"%(t, testmse, testr2, \
                                                        trainmse,  trainr2), file=ofp)
            
    if modelfname != "":
        ofp.close()

    return avgr2_train/num, avgmse_train/num, avgr2_test/num, avgmse_test/num

#######################################################################

def build_v_split (x_s, y_s, vset, modelshape, batch_size, epochs, \
                lossfun, optimizer, activation, \
                vmap_toreal, modelfname="", verbose=False):

    ofp = None
    if modelfname != "":
        ofp = open(modelfname, "w")

    avgr2_test = 0.0
    avgr2_train = 0.0
    avgmse_test = 0.0
    avgmse_train = 0.0

    num = 0.0
    basename = "" 
    if modelfname != "":
        basename = modelfname.split(".csv")[0]

    #early_stopping = keras.callbacks.EarlyStopping(
    #    monitor='val_loss',
    #    patience=10,
    #    min_delta=0.00001,
    #    mode='min',
    #    verbose=1
    #)

    thefirst = True
    if verbose:
        print (" v Removed , Test MSE , Test R2 , Train MSE , Train R2", flush=True)
    if modelfname != "":
        print (" v Removed , Test MSE , Test R2 , Train MSE , Train R2", file=ofp, flush=True)
    for v in vset:

        if FIXEDSEED:
            # to fix seed
            np.random.seed(42)
            tf.random.set_seed(42)
            random.seed(42)

        train_x, test_x, train_y, test_y = cm.test_train_split (0, [v], x_s, y_s)

        if thefirst:
            model = cm.buildmodel(modelshape, lossf=lossfun, optimizerf=optimizer, \
                                    activationf=activation)
            history = model.fit(train_x, train_y, epochs=10,\
                batch_size=batch_size, verbose=0)
            thefirst = False

        model = cm.buildmodel(modelshape, lossf=lossfun, optimizerf=optimizer, \
                                    activationf=activation)
        history = model.fit(train_x, train_y, epochs=epochs,  batch_size=batch_size, \
            verbose=0)
        with open(basename + "_" + \
                  str(vmap_toreal[v]) + "_training_history.pkl", 'wb') as f:
            pickle.dump(history.history, f)

        pred_y = model.predict(test_x, verbose=0)
        testfile = basename + "_" + str(vmap_toreal[v]) + "_test.csv"
        with open(testfile, "a") as f:
            for ix, xval in enumerate(test_x):
                for xx in xval:
                    print("%10.5e, "%xx, end="", file=f)
                print("%10.5e"%(pred_y[ix]), file=f)
        try:
            testmse = metrics.mean_absolute_error(test_y, pred_y)
            testr2 = metrics.r2_score(test_y, pred_y)
        except:
            testmse = float('inf')
            testr2 = 0.0

        avgr2_test += testr2
        avgmse_test += testmse

        pred_y = model.predict(train_x, verbose=0)
        trainfile = basename + "_" + str(vmap_toreal[v]) + "_train.csv"
        with open(trainfile, "a") as f:
            for ix, xval in enumerate(train_x):
                for xx in xval:
                    print("%10.5e, "%xx, end="", file=f)
                print("%10.5e"%(pred_y[ix]), file=f)
        try:
            trainmse = metrics.mean_absolute_error(train_y, pred_y)
            trainr2 = metrics.r2_score(train_y, pred_y)
        except:
            trainmse = float('inf')
            trainr2 = 0.0

        avgr2_train += trainr2
        avgmse_train += trainmse

        num += 1.0
        
        if verbose:
            print("%5d , %10.6f , %10.6f , %10.6f , %10.6f"%(vmap_toreal[v], testmse, testr2, \
                                                        trainmse,  trainr2), flush=True)
        
        if modelfname != "":
            print("%5d , %10.6f , %10.6f , %10.6f , %10.6f"%(vmap_toreal[v], testmse, testr2, \
                                                        trainmse,  trainr2), file=ofp, flush=True)
    
    if modelfname != "":
        ofp.close()

    return avgr2_train/num, avgmse_train/num, avgr2_test/num, avgmse_test/num

#######################################################################

def build_vsets_split (x_s, y_s, vlist, modelshape, batch_size, epochs, \
                    lossfun, optimizer, activation, \
                    vmap_toreal, modelfname="", verbose=False):

    ofp = None
    if modelfname != "":
        ofp = open(modelfname, "w")

    basename = "" 
    if modelfname != "":
        basename = modelfname.split(".csv")[0]

    avgr2_test = 0.0
    avgr2_train = 0.0
    avgmse_test = 0.0
    avgmse_train = 0.0

    num = 0.0

    thefirst = True
    if verbose:
        print (" vset Removed , Test MSE , Test R2 , Train MSE , Train R2", flush=True)
    if modelfname != "":
        print (" vset Removed , Test MSE , Test R2 , Train MSE , Train R2", file=ofp, flush=True)

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

    #print(len(vlist))
    #for v in vset_torm:
    #    print(len(v), v)

    for v in vset_torm:

        if FIXEDSEED:
            # to fix seed
            np.random.seed(42)
            tf.random.set_seed(42)
            random.seed(42)

        train_x, test_x, train_y, test_y = cm.test_train_split (0, v, x_s, y_s)

        if thefirst:
            model = cm.buildmodel(modelshape, lossf=lossfun, optimizerf=optimizer, \
                                    activationf=activation)
            history = model.fit(train_x, train_y, epochs=10,  batch_size=batch_size, \
                verbose=0)
            thefirst = False

        model = cm.buildmodel(modelshape, lossf=lossfun, optimizerf=optimizer, \
                                    activationf=activation)
        history = model.fit(train_x, train_y, epochs=epochs,  batch_size=batch_size, \
            verbose=0)
        valuetoprint = ""
        for val in v:
            valuetoprint += str(vmap_toreal[val]) + "_"
        with open(basename + "_" + \
                  valuetoprint + "training_history.pkl", 'wb') as f:
            pickle.dump(history.history, f)

        pred_y = model.predict(test_x, verbose=0)
        try:
            testmse = metrics.mean_absolute_error(test_y, pred_y)
            testr2 = metrics.r2_score(test_y, pred_y)
        except:
            testmse = float('inf')
            testr2 = 0.0

        avgr2_test += testr2
        avgmse_test += testmse

        pred_y = model.predict(train_x, verbose=0)
        try:
            trainmse = metrics.mean_absolute_error(train_y, pred_y)
            trainr2 = metrics.r2_score(train_y, pred_y)
        except:
            trainmse = float('inf')
            trainr2 = 0.0

        avgr2_train += trainr2
        avgmse_train += trainmse

        num += 1.0
        
        if verbose:
            print("%s , %10.6f , %10.6f , %10.6f , %10.6f"%(valuetoprint, testmse, testr2, \
                                                        trainmse,  trainr2), flush=True)
        if modelfname != "":
            print("%s , %10.6f , %10.6f , %10.6f , %10.6f"%(valuetoprint, testmse, testr2, \
                                                        trainmse,  trainr2), file=ofp, flush=True)
    
    if modelfname != "":
        ofp.close()


    return avgr2_train/num, avgmse_train/num, avgr2_test/num, avgmse_test/num

#######################################################################

if __name__ == "__main__":

    # so all the models should start from the same point
    np.random.seed(812)
    keras.utils.set_random_seed(812)
    tf.config.experimental.enable_op_determinism()

    filename = "N2H2_3D.xlsx"
    testtovisualize = False
    wselected = 0
    columselected = 1

    df = pd.read_excel(filename)

    x = df[['v', 'w', 'T(K)']].values
    y = df[['k(cm^3/s)']].values

    scalerx = MinMaxScaler()
    scalerx.fit(x)
    x_s = scalerx.transform(x)

    vmap_toreal = {}

    for i, vn in enumerate(x_s[:,0]):
        vmap_toreal[vn] = x[i,0]

    print("V map: ")
    for a in vmap_toreal:
        print("%4.2f --> %3d"%(a, vmap_toreal[a]))

    vset = set(x_s[:,0])
    wset = set(x_s[:,1])
    tset = set(x_s[:,2])
    vlist = list(vset)

    scalery = MinMaxScaler()
    scalery.fit(y)
    y_s = scalery.transform(y)

    if testtovisualize:

        if FIXEDSEED:
            # to fix seed
            np.random.seed(42)
            tf.random.set_seed(42)
            random.seed(42)

        train_x, test_x, train_y, test_y = \
            cm.test_train_split (columselected, [wselected], x_s, y_s)
        print("Train shape: ", train_x.shape, "Test shape: ", test_x.shape)

        toplotx = []
        toploty = []
        for i in range(test_x.shape[0]):
            v = test_x[i,0]
            w = test_x[i,1]
            T = test_x[i,2]
            k = test_y[i]

            ix = scalerx.inverse_transform(np.asarray([v, w, T]).reshape(1, -1))
            iy = scalery.inverse_transform(np.asarray([k]))

            toplotx.append([int(ix[0,0]), int(ix[0,1]),int(ix[0,2])])
            toploty.append(iy)

            #print("%4d %4d %5d %10.5e"%(int(ix[0,0]), int(ix[0,1]),int(ix[0,2]), iy[0]))

        #cm.plotfull3dcurve (1, np.asarray(toplotx), np.asarray(toploty))


    modelshape_s = [
        [256, 256, 256, 256, 256, 256],
        [256, 256, 256],
        [128, 128, 128]]
    batch_size_s = [12, 25, 100, 256]
    epochs_s = [1000]
    lossfuns = ['mse', 'mae']
    optimizers = ['adam']
    activations = ['relu']

    # run a grid search
    totalnum = len(modelshape_s)*\
        len(batch_size_s)*len(epochs_s)*\
            len(lossfuns)*len(optimizers)*len(activations)
    print("Total number of models to run: ", totalnum)
    modelnum = 0
    for modelshape in modelshape_s:
        for batch_size in batch_size_s:
            for epochs  in epochs_s:
                for lossfun in lossfuns:
                    for optimizer in optimizers:
                        for activation in activations:
                            modelnum += 1
                            
                            r2test_v_split = 0.0
                            msetest_v_split = 0.0
                            r2train_v_split = 0.0 
                            msetrain_v_split = 0.0
                            
                            r2test_vsets_split = 0.0
                            msetest_vsets_split = 0.0
                            r2train_vsets_split = 0.0 
                            msetrain_vsets_split = 0.0

                            timestart = time.time()

                            r2train_v_split, msetrain_v_split, r2test_v_split, msetest_v_split = \
                                build_v_split (x_s, y_s, vset, modelshape, batch_size, epochs, \
                                            lossfun, optimizer, activation, \
                                            vmap_toreal, modelfname="vsplitmodel_"+str(modelnum)+".csv")
                            
                            r2train_vsets_split, msetrain_vsets_split, \
                                r2test_vsets_split, msetest_vsets_split = \
                                build_vsets_split (x_s, y_s, vlist, modelshape, batch_size, epochs, \
                                                lossfun, optimizer, activation, \
                                                vmap_toreal, modelfname="vsetsplitmodel_"+str(modelnum)+".csv")
                            
                            print("v split , Model metrics %3d , %10.5f , %10.5f , %10.5f , %10.5f"%( \
                                modelnum, r2test_v_split, msetest_v_split, \
                                r2train_v_split, msetrain_v_split), flush=True)
                            print("vsets split , Model metrics %3d , %10.5f , %10.5f , %10.5f , %10.5f"%( \
                                modelnum, r2test_vsets_split, msetest_vsets_split, \
                                r2train_vsets_split, msetrain_vsets_split), flush=True)
                            print("Model shapes  %3d , %s , %5d , %5d , %s , %s , %s "%( \
                                 modelnum, str(modelshape), batch_size, epochs, \
                                    lossfun, optimizer, activation), flush=True)
                            
                            timeend = time.time()
                            print("Total seconds taken: ", timeend-timestart, flush=True)