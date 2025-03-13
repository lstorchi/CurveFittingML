import random 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics

import commonmodules as cm

#######################################################################

def build_v_split (vset, modelshape, \
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
        print (" v Removed , Test MSE , Test R2 , Train MSE , Train R2", flush=True)
    if modelfname != "":
        print (" v Removed , Test MSE , Test R2 , Train MSE , Train R2", file=ofp, flush=True)
    for v in vset:

        train_x, test_x, train_y, test_y = cm.test_train_split (0, [v], x_s, y_s)


        model = cm.buildmodel_RF(modelshape)
        model.fit(train_x, train_y[:,0])

        pred_y = model.predict(test_x)
        testmse = metrics.mean_absolute_error(test_y, pred_y)
        testr2 = metrics.r2_score(test_y, pred_y)

        avgr2_test += testr2
        avgmse_test += testmse

        pred_y = model.predict(train_x)
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

def build_vsets_split (vlist, modelshape, \
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

        model = cm.buildmodel_RF(modelshape)
        model.fit(train_x, train_y[:,0])

        pred_y = model.predict(test_x)
        testmse = metrics.mean_absolute_error(test_y, pred_y)
        testr2 = metrics.r2_score(test_y, pred_y)

        avgr2_test += testr2
        avgmse_test += testmse

        pred_y = model.predict(train_x)
        trainmse = metrics.mean_absolute_error(train_y, pred_y)
        trainr2 = metrics.r2_score(train_y, pred_y)

        avgr2_train += trainr2
        avgmse_train += trainmse

        num += 1.0
        
        if verbose:
            print("%s , %10.6f , %10.6f , %10.6f , %10.6f"%(v, testmse, testr2, \
                                                        trainmse,  trainr2), flush=True)
        
        if modelfname != "":
            print("%s , %10.6f , %10.6f , %10.6f , %10.6f"%(v, testmse, testr2, \
                                                        trainmse,  trainr2), file=ofp, flush=True)
    
    if modelfname != "":
        ofp.close()


    return avgr2_train/num, avgmse_train/num, avgr2_test/num, avgmse_test/num

#######################################################################

if __name__ == "__main__":

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

    vset = set(x_s[:,0])
    wset = set(x_s[:,1])
    tset = set(x_s[:,2])
    vlist = list(vset)

    scalery = MinMaxScaler()
    scalery.fit(y)
    y_s = scalery.transform(y)

    if testtovisualize:
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

    modelnum = 0

    n_estimators = [50, 100, 300, 500, 800, 1200]
    max_depth = [None, 5, 8, 15, 25, 30]
    min_samples_split = [2, 5, 10, 15, 100]
    min_samples_leaf = [10, 20, 50, 100, 200] 
    random_state = [42]
    max_features = ['log2', 'sqrt']
    bootstrap = [True]

    hyperF = {"n_estimators" : n_estimators, 
            "max_depth" : max_depth, 
            "min_samples_split" : min_samples_split, 
            "min_samples_leaf" : min_samples_leaf, 
            "random_state" : random_state, 
            "bootstrap" : bootstrap,
            "max_features" : max_features}

    for a in hyperF["n_estimators"]:
        for b in  hyperF["max_depth"]:
            for c in  hyperF["min_samples_split"]:
                for d in  hyperF["min_samples_leaf"]:
                    for e in  hyperF["random_state"]:
                        for f in  hyperF["bootstrap"]:
                            for g in  hyperF["max_features"]:
                                modelshape = {
                                    "n_estimators" : a,
                                    "max_depth" : b,
                                    "min_samples_split" : c,
                                    "min_samples_leaf" : d,
                                    "random_state" : e,
                                    "bootstrap" : f,
                                    "max_features" : g
                                }    

                                modelnum += 1
              
                                r2test_v_split = 0.0
                                msetest_v_split = 0.0
                                r2train_v_split = 0.0 
                                msetrain_v_split = 0.0
              
                                r2test_vsets_split = 0.0
                                msetest_vsets_split = 0.0
                                r2train_vsets_split = 0.0 
                                msetrain_vsets_split = 0.0
              
                                r2train_v_split, msetrain_v_split, r2test_v_split, msetest_v_split = \
                                    build_v_split (vset, modelshape, \
                                                   modelfname="vsplitmodel_"+str(modelnum)+".csv")
              
                                r2train_vsets_split, msetrain_vsets_split, \
                                    r2test_vsets_split, msetest_vsets_split = \
                                    build_vsets_split (vlist, modelshape, \
                                                       modelfname="vsetsplitmodel_"+str(modelnum)+".csv")
                                
                                print("v split , Model metrics %3d , %10.5f , %10.5f , %10.5f , %10.5f"%( \
                                    modelnum, r2test_v_split, msetest_v_split, \
                                    r2train_v_split, msetrain_v_split), flush=True)
                                print("vsets split , Model metrics %3d , %10.5f , %10.5f , %10.5f , %10.5f"%( \
                                    modelnum, r2test_vsets_split, msetest_vsets_split, \
                                    r2train_vsets_split, msetrain_vsets_split), flush=True)
                                print("Model shapes  %3d , %s "%(\
                                    modelnum, str(modelshape).replace(","," ")), flush=True)
