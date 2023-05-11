import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import commonmodules as cm

from sklearn import metrics

if __name__ == "__main__":

    filename = "N2H2_VVdata_3variables.xlsx"
    df = pd.read_excel(filename)
    debug = False

    x = df[['v', 'w', 'T(K)']].values
    y = df[['k(cm^3/s)']].values

    scalerx = MinMaxScaler()
    scalerx.fit(x)
    x_s = scalerx.transform(x)

    vset = set(x_s[:,0])
    wset = set(x_s[:,1])
    tset = set(x_s[:,2])

    scalery = MinMaxScaler()
    scalery.fit(y)
    y_s = scalery.transform(y)

    if debug:
        for i, ys in enumerate(y_s):
            print(ys, y[i])
        for i, xs in enumerate(x_s):
            print(xs, x[i])


    ofp = open("perc_GP.csv", "w")

    print (" Perc. Split , Test MSE , Test R2 , Train MSE , Train R2")
    print (" Perc. Split , Test MSE , Test R2 , Train MSE , Train R2", file=ofp)
    for perc in [0.05, 0.10, 0.25, 0.30, 0.50]:
        train_x, test_x, train_y, test_y = train_test_split(x_s, y_s, \
                        test_size=perc, random_state=42)

        modelshape = [64, 64, 64]
        epochs = 20
        batch_size = 50

        model = cm.build_model_GP_3D (train_x, train_y)
        pred_y = model.predict(test_x)
        #to scale back y
        #pred_y_sb = scalery.inverse_transform(pred_y)
        #y_sb = scalery.inverse_transform(test_y)
        #plt.scatter(y_sb, pred_y_sb)
        #plt.show()
        testmse = metrics.mean_absolute_error(test_y, pred_y)
        testr2 = metrics.r2_score(test_y, pred_y)

        pred_y = model.predict(train_x, verbose=0)
        trainmse = metrics.mean_absolute_error(train_y, pred_y)
        trainr2 = metrics.r2_score(train_y, pred_y)

        print("%5.2f , %10.6f , %10.6f , %10.6f , %10.6f"%(perc, testmse, testr2, \
                                                        trainmse,  trainr2))
        print("%5.2f , %10.6f , %10.6f , %10.6f , %10.6f"%(perc, testmse, testr2, \
                                                        trainmse,  trainr2), file=ofp)
        
    ofp.close()


    ofp = open("vremoved_GP.csv", "w")

    thefirst = True
    print (" v Removed , Test MSE , Test R2 , Train MSE , Train R2")
    print (" v Removed , Test MSE , Test R2 , Train MSE , Train R2", file=ofp)
    for v in vset:
        train_x, test_x, train_y, test_y = cm.test_train_split (0, [v], x_s, y_s)

        modelshape = [64, 64, 64]
        epochs = 20
        batch_size = 50

        if thefirst:
            model = cm.buildmodel(modelshape)
            history = model.fit(train_x, train_y, epochs=epochs,  batch_size=batch_size, \
                verbose=0)
            thefirst = False

        model = cm.buildmodel(modelshape)
        history = model.fit(train_x, train_y, epochs=epochs,  batch_size=batch_size, \
            verbose=0)

        pred_y = model.predict(test_x)
        #to scale back y
        #pred_y_sb = scalery.inverse_transform(pred_y)
        #y_sb = scalery.inverse_transform(test_y)
        #plt.scatter(y_sb, pred_y_sb)
        #plt.show()
        testmse = metrics.mean_absolute_error(test_y, pred_y)
        testr2 = metrics.r2_score(test_y, pred_y)

        pred_y = model.predict(train_x, verbose=0)
        trainmse = metrics.mean_absolute_error(train_y, pred_y)
        trainr2 = metrics.r2_score(train_y, pred_y)

        print("%5.2f , %10.6f , %10.6f , %10.6f , %10.6f"%(v, testmse, testr2, \
                                                        trainmse,  trainr2))
        
        print("%5.2f , %10.6f , %10.6f , %10.6f , %10.6f"%(v, testmse, testr2, \
                                                        trainmse,  trainr2), file=ofp)
        
    ofp.close()

    ofp = open("wremoved_GP.csv", "w")

    thefirst = True

    print (" w Removed , Test MSE , Test R2 , Train MSE , Train R2")
    print (" w Removed , Test MSE , Test R2 , Train MSE , Train R2", file=ofp)
    for w in wset:
        train_x, test_x, train_y, test_y = cm.test_train_split (1, [w], x_s, y_s)

        modelshape = [64, 64, 64]
        epochs = 20
        batch_size = 50

        if thefirst:
            model = cm.buildmodel(modelshape)
            history = model.fit(train_x, train_y, epochs=epochs,  batch_size=batch_size, \
                verbose=0)
            thefirst = False

        model = cm.buildmodel(modelshape)
        history = model.fit(train_x, train_y, epochs=epochs,  batch_size=batch_size, \
            verbose=0)

        pred_y = model.predict(test_x)
        #to scale back y
        #pred_y_sb = scalery.inverse_transform(pred_y)
        #y_sb = scalery.inverse_transform(test_y)
        #plt.scatter(y_sb, pred_y_sb)
        #plt.show()
        testmse = metrics.mean_absolute_error(test_y, pred_y)
        testr2 = metrics.r2_score(test_y, pred_y)

        pred_y = model.predict(train_x, verbose=0)
        trainmse = metrics.mean_absolute_error(train_y, pred_y)
        trainr2 = metrics.r2_score(train_y, pred_y)

        print("%5.2f , %10.6f , %10.6f , %10.6f , %10.6f"%(w, testmse, testr2, \
                                                        trainmse,  trainr2))
        
        print("%5.2f , %10.6f , %10.6f , %10.6f , %10.6f"%(w, testmse, testr2, \
                                                        trainmse,  trainr2), file=ofp)
        
    ofp.close()

    ofp = open("tremoved_GP.csv", "w")

    thefirst = True

    print (" T Removed , Test MSE , Test R2 , Train MSE , Train R2")
    print (" T Removed , Test MSE , Test R2 , Train MSE , Train R2", file=ofp)
    for t in tset:
        train_x, test_x, train_y, test_y = cm.test_train_split (2, [t], x_s, y_s)

        modelshape = [64, 64, 64]
        epochs = 20
        batch_size = 50

        if thefirst:
            model = cm.buildmodel(modelshape)
            history = model.fit(train_x, train_y, epochs=epochs,  batch_size=batch_size, \
                verbose=0)
            thefirst = False

        model = cm.buildmodel(modelshape)
        history = model.fit(train_x, train_y, epochs=epochs,  batch_size=batch_size, \
            verbose=0)

        pred_y = model.predict(test_x)
        #to scale back y
        #pred_y_sb = scalery.inverse_transform(pred_y)
        #y_sb = scalery.inverse_transform(test_y)
        #plt.scatter(y_sb, pred_y_sb)
        #plt.show()
        testmse = metrics.mean_absolute_error(test_y, pred_y)
        testr2 = metrics.r2_score(test_y, pred_y)

        pred_y = model.predict(train_x, verbose=0)
        trainmse = metrics.mean_absolute_error(train_y, pred_y)
        trainr2 = metrics.r2_score(train_y, pred_y)

        print("%5.2f , %10.6f , %10.6f , %10.6f , %10.6f"%(t, testmse, testr2, \
                                                        trainmse,  trainr2))
        print("%5.2f , %10.6f , %10.6f , %10.6f , %10.6f"%(t, testmse, testr2, \
                                                        trainmse,  trainr2), file=ofp)
        
    ofp.close()