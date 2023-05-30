import random 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

import commonmodules as cm

FIXEDSEED = True

#######################################################################

def build_perc_split (modelshape, batch_size, epochs, \
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

#######################################################################

def build_v_split (vset, modelshape, batch_size, epochs, \
                   modelfname="", verbose=False):

    ofp = None
    if modelfname != "":
        ofp = open(modelfname, "w")

    avgr2_test = 0.0
    avgr2_train = 0.0
    avgmse_test = 0.0
    avgmse_train = 0.0

    num = 0.0

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
            print("%5.2f , %10.6f , %10.6f , %10.6f , %10.6f"%(v, testmse, testr2, \
                                                        trainmse,  trainr2), flush=True)
        
        if modelfname != "":
            print("%5.2f , %10.6f , %10.6f , %10.6f , %10.6f"%(v, testmse, testr2, \
                                                        trainmse,  trainr2), file=ofp, flush=True)
    
    if modelfname != "":
        ofp.close()

    return avgr2_train/num, avgmse_train/num, avgr2_test/num, avgmse_test/num

#######################################################################

def build_vsets_split (vlist, modelshape, batch_size, epochs, \
                   modelfname="", verbose=False):

    ofp = None
    if modelfname != "":
        ofp = open(modelfname, "w")

    avgr2_test = 0.0
    avgr2_train = 0.0
    avgmse_test = 0.0
    avgmse_train = 0.0

    num = 0.0

    thefirst = True
    if verbose:
        print (" v Removed , Test MSE , Test R2 , Train MSE , Train R2", flush=True)
    if modelfname != "":
        print (" v Removed , Test MSE , Test R2 , Train MSE , Train R2", file=ofp, flush=True)

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
    #    print(len(v))
    #exit(1)

    for v in vset_torm:

        if FIXEDSEED:
            # to fix seed
            np.random.seed(42)
            tf.random.set_seed(42)
            random.seed(42)

        train_x, test_x, train_y, test_y = cm.test_train_split (0, v, x_s, y_s)

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
            print("%5.2f , %10.6f , %10.6f , %10.6f , %10.6f"%(v, testmse, testr2, \
                                                        trainmse,  trainr2), flush=True)
        
        if modelfname != "":
            print("%5.2f , %10.6f , %10.6f , %10.6f , %10.6f"%(v, testmse, testr2, \
                                                        trainmse,  trainr2), file=ofp, flush=True)
    
    if modelfname != "":
        ofp.close()


    return avgr2_train/num, avgmse_train/num, avgr2_test/num, avgmse_test/num

#######################################################################

def build_w_split (wset, modelshape, batch_size, epochs, \
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

def build_t_split (tset, modelshape, batch_size, epochs, \
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

if __name__ == "__main__":

    filename = "N2H2_VVdata_3variables.xlsx"
    testtovisualize = False
    wselected = 0
    columselected = 1

    df = pd.read_excel(filename)

    x = df[['v', 'w', 'T(K)']].values
    y = df[['k(cm^3/s)']].values

    scalerx = MinMaxScaler()
    scalerx.fit(x)
    x_s = scalerx.transform(x)

    vset = set(x_s[:,0])
    wset = set(x_s[:,1])
    tset = set(x_s[:,2])
    vlist = list(x_s[:,0])

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

        cm.plotfull3dcurve (1, np.asarray(toplotx), np.asarray(toploty))


    modelshape_s = [[ 8,  8,  8,  8, 8],
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
                    [128, 128, 128],
                    [ 8,  8], 
                    [16, 16],
                    [32, 32],
                    [64, 64], 
                    [128, 128]]
   
    epochs = 20
    batch_size_s = [10, 25, 50, 100]

    modelnum = 0

    for modelshape in modelshape_s:
        for batch_size in batch_size_s:
            #modelfname = "perc.csv"
            
            r2test = 0.0
            msetest = 0.0
            r2train = 0.0 
            msetrain = 0.0

            """
            avgr2_train, avgmse_train, avgr2_test, avgmse_test = \
                build_perc_split (modelshape, batch_size, epochs)
            
            r2test += avgr2_test
            msetest += avgmse_test
            r2train += avgr2_train
            msetrain += avgmse_train
            
            print("  Perc, %10.5f , %10.5f , %10.5f , %10.5f"%(avgmse_train, avgr2_train, \
                                                  avgmse_test,  avgr2_test))
            """
            
            """
            avgr2_train, avgmse_train, avgr2_test, avgmse_test = \
                build_v_split (vset, modelshape, batch_size, epochs)
            
            r2test += avgr2_test
            msetest += avgmse_test
            r2train += avgr2_train
            msetrain += avgmse_train
            
            print("    vSplit, %10.5f , %10.5f , %10.5f , %10.5f"%(avgmse_train, avgr2_train, \
                                                  avgmse_test,  avgr2_test))
            """
            
            avgr2_train, avgmse_train, avgr2_test, avgmse_test = \
                build_vsets_split (vlist, modelshape, batch_size, epochs)
            
            r2test += avgr2_test
            msetest += avgmse_test
            r2train += avgr2_train
            msetrain += avgmse_train
            
            print("vsetsSplit, %10.5f , %10.5f , %10.5f , %10.5f"%(avgmse_train, avgr2_train, \
                                                  avgmse_test,  avgr2_test), flush=True)
            """
            avgr2_train, avgmse_train, avgr2_test, avgmse_test = \
                build_w_split (wset, modelshape, batch_size, epochs)
            
            r2test += avgr2_test
            msetest += avgmse_test
            r2train += avgr2_train
            msetrain += avgmse_train
            
            print("wSplit, %10.5f , %10.5f , %10.5f , %10.5f"%(avgmse_train, avgr2_train, \
                                                  avgmse_test,  avgr2_test))
            
            avgr2_train, avgmse_train, avgr2_test, avgmse_test = \
                build_t_split (tset, modelshape, batch_size, epochs)
            
            r2test += avgr2_test
            msetest += avgmse_test
            r2train += avgr2_train
            msetrain += avgmse_train
            
            print("TSplit, %10.5f , %10.5f , %10.5f , %10.5f"%(avgmse_train, avgr2_train, \
                                                  avgmse_test,  avgr2_test))
            """

            modelnum += 1
            r2test = r2test/2.0
            msetest = msetest/2.0
            r2train = r2train/2.0
            msetrain = msetrain/2.0

            print("Model metrics %3d , %10.5f , %10.5f , %10.5f , %10.5f"%( \
                 modelnum, r2test, msetest, r2train, msetrain), flush=True)
            print("Model shapes  %3d , %s , %5d "%( \
                 modelnum, str(modelshape), batch_size), flush=True)
