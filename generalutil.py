###############################################################################

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd, file = sys.stderr)
    # Print New Line on Complete
    if iteration == total: 
        print(file = sys.stderr)

###############################################################################

def build_vsettorm (vlist):

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

    return vset_torm

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
    f1list = {}

    useonlyv = False

    x["1_v_cE"] = df1[['v', 'cE']].values
    if not useonlyv:
        x["1_dE_cE"] = df1[['dE', 'cE']].values
    y["1"] = np.log10(df1[["cS"]].values)

    x["2_v_cE"] = df2[['v', 'cE']].values
    if not useonlyv:
        x["2_dE_cE"] = df2[['dE', 'cE']].values
    y["2"] = np.log10(df2[["cS"]].values)

    x["3_v_cE"] = df3[['v', 'cE']].values
    if not useonlyv:
        x["3_dE_cE"] = df3[['dE', 'cE']].values
    y["3"] = np.log10(df3[["cS"]].values)

    xkey = ["1_v_cE", "1_dE_cE", \
            "2_v_cE", "2_dE_cE", \
            "3_v_cE", "3_dE_cE"]

    if useonlyv:
        xkey = ["1_v_cE", \
                "2_v_cE", \
                "3_v_cE", ]

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
        lista = list(set(x_s[k][:,0]))
        lista.sort(reverse=False)
        f1list[k] = lista

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

    return xkey, ykey, x_s, y_s, scalerx, scalery, x1map_toreal, f1set, f1list

###############################################################################

def read_excel_file_and_norm_tfile (filename, debug=False):

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
    f1list = {}

    x["1_v_T"] = df1[['v', 'T']].values
    y["1"] = np.log10(df1[["RateC"]].values)

    x["2_v_T"] = df2[['v', 'T']].values
    y["2"] = np.log10(df2[["RateC"]].values)

    x["3_v_T"] = df3[['v', 'T']].values
    y["3"] = np.log10(df3[["RateC"]].values)

    xkey = ["1_v_T", \
            "2_v_T", \
            "3_v_T"]

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
        lista = list(set(x_s[k][:,0]))
        lista.sort(reverse=False)
        f1list[k] = lista

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

    return xkey, ykey, x_s, y_s, scalerx, scalery, x1map_toreal, f1set, f1list

###############################################################################
