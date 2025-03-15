
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf

#import keras
from tensorflow import keras

import keras.optimizers as tko
import keras.activations as tka
import keras.losses as tkl
from keras.layers import Input, Dense
from keras.models import Model

from sklearn.model_selection import train_test_split
from sklearn import metrics

import commonmodules as cm

#import fittingviaNN_3D_CLI as f3d
from sklearn.preprocessing import MinMaxScaler
import pickle

#######################################################################

class SaveModelEpoch(keras.callbacks.Callback):
    def __init__(self, filepath):
        super(SaveModelEpoch, self).__init__()
        self.filepath = filepath

    def on_epoch_end(self, epoch, logs=None):
        epoch_str = '{:04d}'.format(epoch + 1)  # Format epoch number
        filepath = self.filepath.format(epoch=epoch_str)
        self.model.save(filepath)
        #print(f"\nSaved model at epoch {epoch + 1} to {filepath}")
    
#######################################################################

def build_v_split (x_s, y_s, scalery, scalerx, alsolog10, vset, \
                modelshape, batch_size, epochs, \
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
    #    monitor='mse',
    #    patience=10,
    #    min_delta=0.00001,
    #    mode='min',
    #    verbose=1
    #)

    if verbose:
        print (" v Removed , Test MSE , Test R2 , Train MSE , Train R2", flush=True)
    if modelfname != "":
        print (" v Removed , Test MSE , Test R2 , Train MSE , Train R2", file=ofp, flush=True)
    for v in vset:
        train_x, test_x, train_y, test_y = cm.test_train_split (0, [v], x_s, y_s)

        model = cm.buildmodel(modelshape, lossf=lossfun, optimizerf=optimizer, \
                                    activationf=activation)
        
        filepath = 'model_epoch_{epoch}.keras' 
        save_model_callback = SaveModelEpoch(filepath)

        history = model.fit(train_x, train_y, \
                            epochs=epochs,  \
                            batch_size=batch_size, \
                            callbacks=[save_model_callback], \
                            verbose=0)
        # plot training history
        #plt.plot(history.history[lossfun], label='train')   
        #plt.legend()
        #plt.show()
        # print the min of the training loss and epoch
        minmse = min(history.history[lossfun])
        minepoch = np.argmin(history.history[lossfun])
        #print("Test v: ", vmap_toreal[v])
        #print("  min training loss: ", minmse)
        #print("  epoch of min training loss: ", minepoch)
        epoch_str = '{:04d}'.format(minepoch + 1)  # Format epoch number
        filename = filepath.format(epoch=epoch_str)
        # load the model with the min training loss
        model = keras.models.load_model(filename)
        # remove all the files
        for i in range(epochs):
            try:
                epoch_str = '{:04d}'.format(i + 1)  # Format epoch number
                filename = filepath.format(epoch=epoch_str)
                os.remove(filename)
            except:
                print("error in removing file: ", filepath.format(epoch=i)) 
            
        pred_y = model.predict(test_x, verbose=0)
        original_test_y = scalery.inverse_transform(test_y)
        original_pred_y = scalery.inverse_transform(pred_y)
        if alsolog10:
            original_test_y = 10**original_test_y
            original_pred_y = 10**original_pred_y
        testfile = basename + "_" + str(vmap_toreal[v]) + "_test.csv"
        with open(testfile, "a") as f:
            for ix, xval in enumerate(test_x):
                xval_orig = scalerx.inverse_transform(xval.reshape(1,-1))
                for xx in xval_orig[0]:
                    print("%10.5e, "%xx, end="", file=f)
                print("%10.8e , %10.8e"%(original_pred_y[ix][0], \
                                         original_test_y[ix][0]), file=f)
        try:
            testmse = metrics.mean_absolute_error(original_test_y, original_pred_y)
            testr2 = metrics.r2_score(original_test_y, original_pred_y)
        except:
            testmse = float('inf')
            testr2 = 0.0

        avgr2_test += testr2
        avgmse_test += testmse

        pred_y = model.predict(train_x, verbose=0)
        original_train_y = scalery.inverse_transform(train_y)
        original_pred_y = scalery.inverse_transform(pred_y)
        #print("original_train_y: ", original_train_y.shape)
        #print("original_pred_y: ", original_pred_y.shape)
        if alsolog10:
            original_train_y = 10**original_train_y
            original_pred_y = 10**original_pred_y
        trainfile = basename + "_" + str(vmap_toreal[v]) + "_train.csv"
        with open(trainfile, "a") as f:
            for ix, xval in enumerate(train_x):
                xval_orig = scalerx.inverse_transform(xval.reshape(1,-1))
                for xx in xval_orig[0]:
                    print("%10.5e, "%xx, end="", file=f)
                print("%10.8e , %10.8e"%(original_pred_y[ix][0], 
                                         original_train_y[ix][0]), file=f)
        try:
            trainmse = metrics.mean_absolute_error(original_train_y, original_pred_y)
            trainr2 = metrics.r2_score(original_train_y, original_pred_y)
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

def build_vsets_split (x_s, y_s, scalery, scalerx, alsolog10, vlist, modelshape, batch_size, epochs, \
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
        #plt.plot(history.history[lossfun], label='train')
        #plt.legend()
        #plt.show()
        #print("Test Vset: ", valuetoprint)
        #print("  min training loss: ", min(history.history[lossfun]))
        #print("  epoch of min training loss: ", np.argmin(history.history[lossfun]))

        pred_y = model.predict(test_x, verbose=0)
        original_test_y = scalery.inverse_transform(test_y)
        original_pred_y = scalery.inverse_transform(pred_y)
        if alsolog10:
            original_test_y = 10**original_test_y
            original_pred_y = 10**original_pred_y
        try:
            testmse = metrics.mean_absolute_error(original_test_y, original_pred_y)
            testr2 = metrics.r2_score(original_test_y, original_pred_y)
        except:
            testmse = float('inf')
            testr2 = 0.0

        testfile = basename + "_" + valuetoprint + "_test.csv"
        with open(testfile, "a") as f:
            for ix, xval in enumerate(test_x):
                xval_orig = scalerx.inverse_transform(xval.reshape(1,-1))
                for xx in xval_orig[0]:
                    print("%10.5e, "%xx, end="", file=f)
                print("%10.8e , %10.8e"%(original_pred_y[ix][0], 
                                         original_test_y[ix][0]), file=f)

        avgr2_test += testr2
        avgmse_test += testmse

        pred_y = model.predict(train_x, verbose=0)
        original_train_y = scalery.inverse_transform(train_y)
        original_pred_y = scalery.inverse_transform(pred_y)
        if alsolog10:
            original_train_y = 10**original_train_y
            original_pred_y = 10**original_pred_y
        try:
            trainmse = metrics.mean_absolute_error(original_train_y, original_pred_y)
            trainr2 = metrics.r2_score(original_train_y, original_pred_y)
        except:
            trainmse = float('inf')
            trainr2 = 0.0
        trainfile = basename + "_" + valuetoprint + "_test.csv"
        with open(trainfile, "a") as f:
            for ix, xval in enumerate(train_x):
                xval_orig = scalerx.inverse_transform(xval.reshape(1,-1))
                for xx in xval_orig[0]:
                    print("%10.5e, "%xx, end="", file=f)
                print("%10.8e, %10.8e"%(original_pred_y[ix][0], 
                                original_train_y[ix][0]), file=f)

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

    filename = "N2H2_VT_process.xlsx"
    df = pd.read_excel(filename, sheet_name="dv=1")
    debug = False

    x = df[['v', 'dE', 'cE']].values
    #y = df[['k(cm^3/s)']].values
    y = np.log10(df[['cS']].values)
    scalery = MinMaxScaler()
    scalery.fit(y)
    y_s = scalery.transform(y)

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
    vlist = list(vset)

    import time

    modelshape_s = [
            [256, 256, 256], 
            [256, 256, 256, 256, 256, 256]]
    batch_size_s = [50, 256]
    epochs_s = [1000]
    lossfuns = ['mse']
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
                                build_v_split (x_s, y_s, scalery, scalerx, False, vset, modelshape, \
                                            batch_size, epochs, \
                                            lossfun, optimizer, activation, \
                                            vmap_toreal, modelfname="vsplitmodel_"+str(modelnum)+".csv")

                            r2train_vsets_split, msetrain_vsets_split, \
                                r2test_vsets_split, msetest_vsets_split = \
                                build_vsets_split (x_s, y_s, scalery, scalerx, False, vlist, modelshape, \
                                                batch_size, epochs, \
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