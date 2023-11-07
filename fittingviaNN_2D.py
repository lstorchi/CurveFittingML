import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import commonmodules as cm

###############################################################################

def read_excel_file_and_norm (filename, debug=False):

    excf = pd.ExcelFile(filename)

    if debug:
        print(excf.sheet_names)

    df1 = pd.read_excel(excf, "dv=1")
    df2 = pd.read_excel(excf, "dv=2")
    df3 = pd.read_excel(excf, "dv=3")

    if debug:
        print(df1.columns)
        print(df2.columns)
        print(df3.columns)

    x = {}
    x_s = {}
    y = {} 
    y_s = {}
    scalerx = {}
    scalery = {}
    x1map_toreal = {}
    f1set = {}

    x["1_v_cE"] = df1[['v', 'cE']].values
    x["1_dE_cE"] = df1[['dE', 'cE']].values
    y["1"] = np.log10(df1[["cS"]].values)

    x["2_v_cE"] = df2[['v', 'cE']].values
    x["2_dE_cE"] = df2[['dE', 'cE']].values
    y["2"] = np.log10(df2[["cS"]].values)

    x["3_v_cE"] = df3[['v', 'cE']].values
    x["3_dE_cE"] = df3[['dE', 'cE']].values
    y["3"] = np.log10(df3[["cS"]].values)

    xkey = ["1_v_cE", "1_dE_cE", \
            "2_v_cE", "2_dE_cE", \
            "3_v_cE", "3_dE_cE"]

    ykey = ["1", "2", "3"]

    for k in xkey:
        scalerx[k] = MinMaxScaler()
        scalerx[k].fit(x[k])
        x_s[k] = scalerx[k].transform(x[k])

        x1map = {}

        for i, vn in enumerate(x_s[k][:,0]):
            x1map[vn] = x[k][i,0]

        x1map_toreal[k] = x1map
    
        f1set[k] = set(x_s[k][:,0])

        if debug:
            for i, xs in enumerate(x_s[k]):
                print(xs, x[k][i])

    for k in ykey:
        scalery[k] = MinMaxScaler()
        scalery[k].fit(y[k])
        y_s[k] = scalery[k].transform(y[k])

        if debug:
            for i, ys in enumerate(y_s[k]):
                print(ys, y[k][i]) 

    return xkey, ykey, x_s, y_s, scalerx, scalery, x1map_toreal, f1set

###############################################################################

if __name__ == "__main__":

    #cE = Collision Energy
    #dE = Delta E 
    #cS = Cross Section

    filename = "N2H2_2D.xlsx"

    xkey, ykey, x_s, y_s, scalerx, scalery, x1map_toreal, f1set = \
        read_excel_file_and_norm (filename)

