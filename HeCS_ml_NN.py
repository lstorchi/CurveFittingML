import os
import math

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import sklearn.gaussian_process as gp
import sklearn.preprocessing as skp
import sklearn.gaussian_process as gp
from sklearn.gaussian_process.kernels import RationalQuadratic, \
    Matern, RBF, ConstantKernel, DotProduct
import commonmodules as cm
from fittingviaNN_process_two_2D_cS import SaveModelEpoch
import tensorflow as tf

#import keras
from tensorflow import keras

import keras.optimizers as tko
import keras.activations as tka
import keras.losses as tkl
from keras.layers import Input, Dense
from keras.models import Model

if __name__ == "__main__":
    filename = "HeCS.xlsx"
    df = pd.read_excel(filename, sheet_name="dv1")
    #print(df.head())
    #for cname in df.columns:
    #    print(cname, df[cname].dtype)
    x = df[['v', 'T']].values
    y = df['RC'].values
    X = np.array(x)
    Y = np.array(y)
    temp_values = np.unique(X[:, 1])
    vib_values = np.unique(X[:, 0])

    print("X shape :", X.shape)
    print("Y shape :", Y.shape)
    vib_values_uniq = np.unique(X[:, 0])
    print("Vibration values:", vib_values_uniq)
    vibtest = vib_values_uniq[int(len(vib_values_uniq)/2)]
    print("Testing with vibration value:", vibtest)
    train_xy = []
    train_z = []
    test_xy = []
    test_z = []
    for i in range(X.shape[0]):
        if X[i][0] == vibtest:
            test_xy.append(X[i])
            test_z.append(Y[i])
        else:
            train_xy.append(X[i])
            train_z.append(Y[i])
    train_xy = np.array(train_xy)
    train_z = np.array(train_z)
    test_xy = np.array(test_xy)
    test_z = np.array(test_z)
    print("Train X shape:", train_xy.shape)
    print("Train Y shape:", train_z.shape)
    print("Test X shape:", test_xy.shape)
    print("Test Y shape:", test_z.shape)        

    # log scale for Y
    train_z = np.log(train_z)
    test_z = np.log(test_z)
    # scale values to [0, 1]
    scalerxy = skp.MinMaxScaler()
    train_xy = scalerxy.fit_transform(train_xy)
    test_xy = scalerxy.transform(test_xy)
    # same for z
    scalerz = skp.MinMaxScaler()
    train_z = scalerz.fit_transform(train_z.reshape(-1, 1))
    test_z = scalerz.transform(test_z.reshape(-1, 1))

    modelshape_s = [
            [64, 64, 64],
            [128, 128, 128, 128], 
            [256, 256, 256], 
            [256, 256, 256, 256, 256, 256]]
    batch_size_s = [10, 50, 256]
    epochs_s = [1000]
    lossfuns = ['mse']
    optimizers = ['adam']
    activations = ['relu']

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
                            inshape = train_xy.shape[1]

                            model = cm.buildmodel(modelshape, inputshape=inshape, \
                                    lossf=lossfun, optimizerf=optimizer, \
                                    activationf=activation)

                            filepath = 'model_epoch_{epoch}.keras' 
                            save_model_callback = SaveModelEpoch(filepath)

                            history = model.fit(train_xy, train_z, \
                                epochs=epochs,  \
                                batch_size=batch_size, \
                                callbacks=[save_model_callback], \
                                verbose=0)
                            minmse = min(history.history[lossfun])
                            minepoch = np.argmin(history.history[lossfun])
                            epoch_str = '{:04d}'.format(minepoch + 1)  # Format epoch number
                            filename = filepath.format(epoch=epoch_str)
                            model = keras.models.load_model(filename)

                            try:
                                epoch_str = '{:04d}'.format(i + 1)  # Format epoch number
                                filename = filepath.format(epoch=epoch_str)
                                os.remove(filename)
                            except:
                                print("error in removing file: ", filepath.format(epoch=i))

                            pred_z = model.predict(train_xy)
                            pred_z_sb = scalerz.inverse_transform(pred_z.reshape(-1, 1))
                            train_z_sb = scalerz.inverse_transform(train_z.reshape(-1, 1))
                            trainmse = np.mean((pred_z_sb - train_z_sb) ** 2)
                            print("Train MSE:", trainmse)
                            train_xy_sb = scalerxy.inverse_transform(train_xy.reshape(-1, 2))
                            fp = open(f"train_predictions_model_{modelnum}.txt", "w")
                            for i in range(len(train_z_sb)):
                                fp.write(f"{train_xy_sb[i][0]} , {train_xy_sb[i][1]}")
                                fp.write(f"{train_z_sb[i][0]} , {pred_z_sb[i][0]}\n")
                            fp.close()

                            pred_z = model.predict(test_xy)
                            pred_z_sb = scalerz.inverse_transform(pred_z.reshape(-1, 1))
                            test_z_sb = scalerz.inverse_transform(test_z.reshape(-1, 1))
                            mse = np.mean((pred_z_sb - test_z_sb) ** 2)
                            print("Test MSE:", mse)
                            fp = open(f"test_predictions_model_{modelnum}.txt", "w")
                            for i in range(len(test_z_sb)):
                                fp.write(f"{test_xy[i][0]} , {test_xy[i][1]}")
                                fp.write(f"{test_z_sb[i][0]} , {pred_z_sb[i][0]}\n")
                            fp.close()
