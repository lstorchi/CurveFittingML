import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import sklearn.gaussian_process as gp
from sklearn.gaussian_process.kernels import RationalQuadratic, \
    Matern, RBF, ConstantKernel, DotProduct

import commonmodules as cm

from sklearn import metrics
import time

if __name__ == "__main__":

    filename = "N2H2_VT_process.xlsx"
    sheetname = "dv=1"
    if len(sys.argv) == 3:
        filename = sys.argv[1]
        sheetname = sys.argv[2]
    elif len(sys.argv) == 2:
        sheetname = sys.argv[1]

    df = pd.read_excel(filename, sheet_name=sheetname)
    debug = False

    x = df[['v', 'cE']].values
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

    if debug:
        for i, ys in enumerate(y_s):
            print(ys, y[i])
        for i, xs in enumerate(x_s):
            print(xs, x[i])
     
    for nu in [0.5]:
        for scale in [0.8]:
            ofp = open("vremoved_GP_"+str(nu)+".csv", "w")
        
            avgr2test = 0.0
            avgmsetest = 0.0
            avgr2train = 0.0
            avgmsetrain = 0.0
            print (" v Removed , Test MSE , Test R2 , Train MSE , Train R2")
            print (" v Removed , Test MSE , Test R2 , Train MSE , Train R2", file=ofp)
        
            starttime = time.time()
            for v in vset:
                
                train_x, test_x, train_y, test_y = cm.test_train_split (0, [v], x_s, y_s)
        
                test_y_sb = scalery.inverse_transform(test_y)
        
                kernel = scale * Matern(length_scale=scale, nu=nu)
                model = gp.GaussianProcessRegressor(kernel=kernel, \
                    n_restarts_optimizer=50, \
                    normalize_y=False)
                model.fit(train_x, train_y)
        
                test_x_sb = scalerx.inverse_transform(test_x)
                pred_y = model.predict(test_x)
                pred_y_sb = scalery.inverse_transform(pred_y.reshape(-1,1))
                test_y_sb = scalery.inverse_transform(test_y)
        
                with open("vremoved_GP_"+\
                          str(nu)+"_"+ str(scale) + "_" +\
                          str(vmap_toreal[v])+"_test.csv", "w") as ofptest:
                    for ix, x_sb in enumerate(test_x_sb):
                        for xx in x_sb:
                            print("%15.19e , "%(xx), end="", file=ofptest)
                        print("%15.19e , %15.19e "%(test_y_sb[ix], pred_y_sb[ix]),
                                file=ofptest, flush=True)
        
                testmse = metrics.mean_absolute_error(test_y_sb, pred_y_sb)
                testr2 = metrics.r2_score(test_y_sb, pred_y_sb)
                
                pred_y = model.predict(train_x)
                pred_y_sb = scalery.inverse_transform(pred_y.reshape(-1,1))
                train_y_sb = scalery.inverse_transform(train_y.reshape(-1,1))
                train_x_sb = scalerx.inverse_transform(train_x)
                with open("vremoved_GP_"+\
                            str(nu)+ "_" + str(scale) + "_" + \
                            str(vmap_toreal[v])+"_train.csv", "w") as ofptrain:
                    for ix, x_sb in enumerate(train_x_sb):
                        for xx in x_sb:
                            print("%15.19e , "%(xx), end="", file=ofptrain)
                        print("%15.19e , %15.19e "%(train_y_sb[ix], pred_y_sb[ix]),
                                file=ofptrain, flush=True)
                trainmse = metrics.mean_absolute_error(train_y_sb, pred_y_sb)
                trainr2 = metrics.r2_score(train_y_sb, pred_y_sb)
                
                print("%3d , %10.6e , %10.6f , %10.6e , %10.6f"%(vmap_toreal[v], testmse, testr2, \
                                                                   trainmse,  trainr2), flush=True)
                
                print("%3d , %10.6e , %10.6f , %10.6e , %10.6f"%(vmap_toreal[v], testmse, testr2, \
                                                                   trainmse,  trainr2), file=ofp, flush=True)
                avgmsetest += testmse
                avgr2test += testr2
                avgmsetrain += trainmse
                avgr2train += trainr2
            ofp.close()
            endtime = time.time()
            print("Time taken for nu = %4.2f is %10.6f"%(nu, endtime-starttime))
            print("v split ", nu, avgmsetest/len(vset), avgr2test/len(vset), \
                  avgmsetrain/len(vset), avgr2train/len(vset))
        
            vlist = list(vset)
        
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
        
            avgr2test = 0.0
            avgmsetest = 0.0
            avgr2train = 0.0
            avgmsetrain = 0.0
        
            ofp = open("vsetremoved_GP_"+str(nu)+".csv", "w")
            starttime = time.time()
            print (" vset Removed , Test MSE , Test R2 , Train MSE , Train R2", flush=True)
            print (" vset Removed , Test MSE , Test R2 , Train MSE , Train R2", file=ofp, \
                flush=True)
            for setid, v in enumerate(vset_torm):
                print("SetID: ", setid)
                for vsetvalue in v:
                    print(vmap_toreal[vsetvalue], " ", end="")
                print(flush=True)
            
                train_x, test_x, train_y, test_y = cm.test_train_split (0, v, x_s, y_s)
            
                v_sp = []
                for val in v:
                    v_sp.append(vmap_toreal[val]) 
                
                kernel = scale * Matern(length_scale=scale, nu=nu)
                model = gp.GaussianProcessRegressor(kernel=kernel, \
                    n_restarts_optimizer=50, \
                    normalize_y=False)
                model.fit(train_x, train_y)

                pred_y = model.predict(test_x)
                test_x_sb = scalerx.inverse_transform(test_x)
                pred_y_sb = scalery.inverse_transform(pred_y.reshape(-1, 1))
                test_y_sb = scalery.inverse_transform(test_y.reshape(-1, 1))
                with open("vsetremoved_GP_set"+str(setid+1)+\
                            "_" + str(nu) + "_" + str(scale) + \
                            "_test.csv", "w") as ofptest:
                    for ix, x_sb in enumerate(test_x_sb):
                        for xx in x_sb:
                            print("%15.19e , "%(xx), end="", file=ofptest)
                        print("%15.19e , %15.19e "%(test_y_sb[ix], pred_y_sb[ix]),
                                file=ofptest, flush=True)
                testmse = metrics.mean_absolute_error(test_y_sb, pred_y_sb)
                testr2 = metrics.r2_score(test_y_sb, pred_y_sb)
            
                ofptrain = open("vsetremoved_GP_set"+str(setid+1)+\
                    "_" + str(nu) + str(scale) + "_train.csv", "w")
                pred_y = model.predict(train_x)
                pred_y_sb = scalery.inverse_transform(pred_y.reshape(-1, 1))
                train_y_sb = scalery.inverse_transform(train_y.reshape(-1, 1))
                train_x_sb = scalerx.inverse_transform(train_x)
                with open("vsetremoved_GP_set"+str(setid+1)+\
                            "_" + str(nu) + "_" + \
                                str(scale) + "_train.csv", "w") as ofptrain:
                    for i, xt in enumerate(train_x_sb):
                        for xx in xt:
                            print("%15.19e , "%(xx), end="", file=ofptrain)
                        print("%15.19e , %15.19e "%(train_y_sb[i], pred_y_sb[i]),
                                file=ofptrain, flush=True)
                        
                trainmse = metrics.mean_absolute_error(train_y_sb, pred_y_sb)
                trainr2 = metrics.r2_score(train_y_sb, pred_y_sb)
                    
                print("%s , %10.6e , %10.6f , %10.6e , %10.6f"%(str(v_sp).replace(",",";"), testmse, testr2, \
                                                                    trainmse,  trainr2), flush=True)
                print("%s , %10.6e , %10.6f , %10.6e , %10.6f"%(str(v_sp).replace(",",";"), testmse, testr2, \
                                                                    trainmse,  trainr2), file=ofp, flush=True)
                
                avgmsetest += testmse
                avgr2test += testr2
                avgmsetrain += trainmse
                avgr2train += trainr2
        
            ofp.close()
            endtime =  time.time()
            print("Time taken for nu = %4.2f is %10.6f"%(nu, endtime-starttime))
            print("vset split ", nu, avgmsetest/len(vset_torm), avgr2test/len(vset_torm), \
                  avgmsetrain/len(vset_torm), avgr2train/len(vset_torm))