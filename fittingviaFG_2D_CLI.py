import numpy as np
import pandas as pd

from tensorflow import keras
import tensorflow as tf

from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

import commonmodules as cm
import fotmulagenerator as fg
from generalutil import *
import sys

###############################################################################

if __name__ == "__main__":

    #cE = Collision Energy
    #dE = Delta E 
    #cS = Cross Section
    dumppredictions = True

    # first file
    filename = "N2H2_2D_VT_process.xlsx"
    xkey, ykey, x_s, y_s, scalerx, scalery, x1map_toreal, f1set, f1list = \
        read_excel_file_and_norm (filename)
    basicdescriptors = ["v", "cE"]

    # second file
    #filename = "N2H2_2D_VT_process_using_T.xlsx"
    #xkey, ykey, x_s, y_s, scalerx, scalery, x1map_toreal, f1set, f1list = \
    #    read_excel_file_and_norm_tfile (filename)
    #basicdescriptors = ["v", "T"]

    print (" xK , split , FormulaGen , avg TrainMSE , avg TrainR2,  avg TestMSE ,avg TestR2 ")

    formulatypes = ["gen1"]

    totvalues = len(xkey) * len(formulatypes)

    counter = 0

    for xk in xkey:
        if len(xk.split("_")) != 3:
            print("Error: xk.split('_') != 3")
            exit(1)

        yk = xk.split("_")[0]
        f1 = xk.split("_")[1]
        f2 = xk.split("_")[2]

        for ft in formulatypes:

            counter += 1
            print("vsplit: ", counter, "/", totvalues, flush=True, file=sys.stderr)
            
            testmses  = []
            testr2s   = []
            trainmses = []
            trainr2s  = []
            
            for x1 in f1set[xk]:
                removedx = x1map_toreal[xk][x1]
                train_x, test_x, train_y, test_y = cm.test_train_split (0, [x1], \
                                                                        x_s[xk], y_s[yk])
            
                model = fg.buildmodel(ft, basicdescriptors)
                model.fit(train_x, train_y)
                
                fp = None
                fpplot = None
                if dumppredictions:
                    fp = open("model_"+str(counter)+"_"+\
                            "xk_"+xk+"_"+\
                            "rm_"+str(removedx)+"_"+\
                            "predictions.csv", "w")

                    print ("Set,"+f1+","+f2+",y,pred_y", file=fp)
                    
                    fpplot = open("model_"+str(counter)+"_"+\
                            "xk_"+xk+"_"+\
                            "rm_"+str(removedx)+"_"+\
                            "toplot.csv", "w")

                    print (f1+" "+f2+" y", file=fpplot)        
            
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

                        print (x1[0], x1[1], pred_y_sb[i][0], file=fpplot)
            
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

                        print (x1[0], x1[1], train_y_sb[i][0], file=fpplot)

                trainmse = metrics.mean_absolute_error(train_y_sb, pred_y_sb)
                trainr2 = metrics.r2_score(train_y_sb, pred_y_sb)
                trainmses.append(trainmse)
                trainr2s.append(trainr2)

                if dumppredictions:
                    fp.close()
                    fpplot.close()

                printProgressBar (len(testmses), len(f1set[xk]), \
                                   prefix = 'Progress:', \
                                    suffix = 'Complete', length = 50)
            
            
            print (xk, " , vsplit , ", ft , ";" , \
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

        for ft in formulatypes: 
                                     
            testmses  = []
            testr2s   = []
            trainmses = []
            trainr2s  = []

            counter += 1
            print("vsetsplit: ", counter, "/", totvalues, flush=True, file=sys.stderr)

            for vset in vsettorm:

                removedx = ""
                fp = None
                fpplot = None
                if dumppredictions:
                    for v in vset:
                        removedx += str(x1map_toreal[xk][v])+ \
                                    "_"

                    fp = open("model_"+str(counter)+"_"+\
                            "xk_"+xk+"_"+\
                            "rmset_"+str(removedx)+"_"+\
                            "predictions.csv", "w")
                    
                    print ("Set,"+f1+","+f2+",y,pred_y", file=fp)

                    fpplot = open("model_"+str(counter)+"_"+\
                            "xk_"+xk+"_"+\
                            "rmset_"+str(removedx)+"_"+\
                            "toplot.csv", "w")
                    
                    print (f1+" "+f2+" y", file=fpplot)
            

                train_x, test_x, train_y, test_y = cm.test_train_split (0, vset, \
                                                                        x_s[xk], y_s[yk])
            
                model = fg.buildmodel(ft, basicdescriptors)
                history = model.fit(train_x, train_y)
            
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
                        
                        print (x1[0], x1[1], pred_y_sb[i][0], file=fpplot)
            
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
                        
                        print (x1[0], x1[1], train_y_sb[i][0], file=fpplot)


                trainmse = metrics.mean_absolute_error(train_y_sb, pred_y_sb)
                trainr2 = metrics.r2_score(train_y_sb, pred_y_sb)
                trainmses.append(trainmse)
                trainr2s.append(trainr2)

                if dumppredictions:
                    fp.close()
                    fpplot.close()  

                printProgressBar (len(testmses), len(vsettorm),\
                                   prefix = 'Progress:',\
                                      suffix = 'Complete', length = 50)

            print (xk, " , vsetsplit , ", ft, \
                   " , ", np.average(trainmses), \
                   " , ", np.average(trainr2s), \
                   " , ", np.average(testmses), \
                   " , ", np.average(testr2s), flush=True)
