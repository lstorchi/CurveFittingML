import sys

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import sklearn.gaussian_process as gp
from mpl_toolkits.mplot3d import Axes3D

import commonmodules as cm

#################################################################################################

if __name__ == "__main__":

    filename = "dv1.xlsx"
    testvib = [11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 37, 38, 39]
    if (len(sys.argv) == 3):
        filename = sys.argv[1]
        testvib = []
        sline = sys.argv[2].split(",")
        for n in sline:
            testvib.append(int(n))
    elif (len(sys.argv) == 2):
        filename = sys.argv[1]

    basename = filename.split(".")[0]

    ofp = open(basename+"_results_NN1.csv", "w")

    df, vib_values, temp_values, minvalue, maxvalue = cm.filterinitialset (filename)
    
    maxt = max(temp_values)
    mint = min(temp_values)

    minv = min(vib_values)
    maxv = max(vib_values)

    epochs = 50
    batch_size = 50

    train_xy, train_z, test_xy = cm.get_train_full (temp_values, vib_values, \
        df, testvib)
        
    model = cm.build_model_NN_1()
    history = model.fit(train_xy, train_z, epochs=epochs,  batch_size=batch_size, \
               verbose=1)

    z_pred = model.predict(test_xy)

    for vslct in testvib :
        print("V = ", vslct)
        print("T , V, Predicted Value", file=ofp)
        for i in range(z_pred.shape[0]):
            t = test_xy[i,0]
            t = int(t*(maxt - mint)+mint)
            v = test_xy[i,1]
            v = int(v*(maxv - minv)+minv)
            if (v == vslct):
                zpred = z_pred[i]
        
                denorm_zpred = (zpred*(maxvalue - minvalue))+minvalue
    
                print("%10.2f , %10.2f , %10.7e "%(t, v, denorm_zpred), file=ofp)
    
    ofp.close()