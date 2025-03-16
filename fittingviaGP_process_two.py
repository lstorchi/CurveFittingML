import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

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

    x = df[['v', 'dE', 'cE']].values
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

    if debug:
        for i, ys in enumerate(y_s):
            print(ys, y[i])
        for i, xs in enumerate(x_s):
            print(xs, x[i])
     
    for nu in [1.0, 2.5]:
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
            
            model = cm.build_model_GP_3D (train_x, train_y, nuval=nu)
            
            ofptest = open("vremoved_GP_"+\
                           str(nu)+"_"+ \
                           str(vmap_toreal[v])+"_test.csv", "w")
            print (" v , dE , cE , y , y_pred ", file=ofptest)
            print ("Test Shape : ", test_x.shape)
            test_x_sp = scalerx.inverse_transform(test_x)
            pred_y = model.predict(test_x)
            print ("Pred y Shape ", pred_y.shape)
            print ("Test y Shape ", test_y.shape)
            pred_y_sb = scalery.inverse_transform(pred_y.reshape(-1,1))
            test_y_sb = scalery.inverse_transform(test_y.reshape(-1,1))
            for i, yt in enumerate(test_y_sb):
                 print (" %3d , %3d , %6d , %10.8e , %10.8e  "%(test_x_sp[i,0], 
                                                     test_x_sp[i,1],
                                                     test_x_sp[i,2],
                                                     yt,
                                                     pred_y_sb[i]), file=ofptest, flush=True)
            #plt.scatter(test_y_sb, pred_y_sb)
            #plt.show()
            testmse = metrics.mean_absolute_error(test_y_sb, pred_y_sb)
            testr2 = metrics.r2_score(test_y_sb, pred_y_sb)
            ofptest.close()
            
            ofptrain = open("vremoved_GP_"+\
                str(nu)+"_"+ \
                str(vmap_toreal[v])+"_train.csv", "w")
            print (" v , dE , cE , y , y_pred  ", file=ofptrain)
            pred_y = model.predict(train_x)
            #print ("Train y Shape ", train_y.shape)
            #print ("Pred y Shape ", pred_y.shape)
            pred_y_sb = scalery.inverse_transform(pred_y.reshape(-1,1))
            train_y_sb = scalery.inverse_transform(train_y.reshape(-1,1))
            train_x_sp = scalerx.inverse_transform(train_x)
            for i, yt in enumerate(train_y_sb):
                 print (" %3d , %3d , %6d , %10.8e , %10.8e  "%(train_x_sp[i,0], 
                                                     train_x_sp[i,1],
                                                     train_x_sp[i,2],
                                                     yt,
                                                     pred_y_sb[i]), file=ofptrain, flush=True)
            #plt.scatter(train_y_sb, pred_y_sb)
            #plt.show()
            trainmse = metrics.mean_absolute_error(train_y_sb, pred_y_sb)
            trainr2 = metrics.r2_score(train_y_sb, pred_y_sb)
            ofptrain.close()
            
            print("%3d , %10.6e , %10.6f , %10.6e , %10.6f"%(vmap_toreal[v], testmse, testr2, \
                                                               trainmse,  trainr2))
            
            print("%3d , %10.6e , %10.6f , %10.6e , %10.6f"%(vmap_toreal[v], testmse, testr2, \
                                                               trainmse,  trainr2), file=ofp)
            avgmsetest += testmse
            avgr2test += testr2
            avgmsetrain += trainmse
            avgr2train += trainr2
        ofp.close()
        endtime = time.time()
        print("Time taken for nu = %4.2f is %10.6f"%(nu, endtime-starttime))
        print(nu, avgmsetest/len(vset), avgr2test/len(vset), avgmsetrain/len(vset), avgr2train/len(vset))

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
            
            model = cm.build_model_GP_3D (train_x, train_y, nuval=nu)
        
            ofptest = open("vsetremoved_GP_set"+str(setid+1)+\
                "_" + str(nu) + "_test.csv", "w")
            print (" v , dE , cE , y , y_pred", file=ofptest, flush=True)
            pred_y = model.predict(test_x)
            test_x_sp = scalerx.inverse_transform(test_x)
            pred_y_sb = scalery.inverse_transform(pred_y.reshape(-1, 1))
            test_y_sb = scalery.inverse_transform(test_y.reshape(-1, 1))
            for i, yt in enumerate(test_y_sb):
                print (" %3d , %3d , %6d , %10.8e , %10.8e  "%(test_x_sp[i,0], 
                                                     test_x_sp[i,1],
                                                     test_x_sp[i,2],
                                                     yt,
                                                     pred_y_sb[i]), file=ofptest, flush=True)
            #plt.scatter(test_y_sb, pred_y_sb)
            #plt.show()
            testmse = metrics.mean_absolute_error(test_y_sb, pred_y_sb)
            testr2 = metrics.r2_score(test_y_sb, pred_y_sb)
            ofptest.close()
        
            ofptrain = open("vsetremoved_GP_set"+str(setid+1)+\
                "_" + str(nu) + "_train.csv", "w")
            print (" v , dE , cE , y , y_pred  ", file=ofptrain)
            pred_y = model.predict(train_x)
            pred_y_sb = scalery.inverse_transform(pred_y.reshape(-1, 1))
            train_y_sb = scalery.inverse_transform(train_y.reshape(-1, 1))
            train_x_sp = scalerx.inverse_transform(train_x)
            for i, yt in enumerate(train_y_sb):
                 print (" %3d , %3d , %6d , %10.8e , %10.8e  "%(train_x_sp[i,0], 
                                                     train_x_sp[i,1],
                                                     train_x_sp[i,2],
                                                     yt,
                                                     pred_y_sb[i]), file=ofptrain, flush=True)
            #plt.scatter(train_y_sb, pred_y_sb)
            #plt.show()
            trainmse = metrics.mean_absolute_error(train_y_sb, pred_y_sb)
            trainr2 = metrics.r2_score(train_y_sb, pred_y_sb)
            ofptrain.close()
                
            print("%s , %10.6e , %10.6f , %10.6e , %10.6f"%(str(v_sp).replace(",",";"), testmse, testr2, \
                                                                trainmse,  trainr2), flush=True)
            print("%s , %10.6e , %10.6f , %10.6e , %10.6f"%(str(v_sp).replace(",",";"), testmse, testr2, \
                                                                trainmse,  trainr2), file=ofp, flush=True)
            
        ofp.close()
        endtime =  time.time()
        print("Time taken for nu = %4.2f is %10.6f"%(nu, endtime-starttime))