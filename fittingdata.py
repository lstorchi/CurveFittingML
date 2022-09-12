import sys

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

filename = ""

if len(sys.argv) != 2:
    print("Usage: %s filename", sys.argv[0])
    exit(1)
else:
    filename = sys.argv[1]

df = pd.read_excel(filename)

for v in df.columns[1:]:
    T = float(v)
    print(T)
    print(df[v].values)

vib= []
for v in df["vibrational level v\Temperature(K)"].values:
    vib.append(float(v))

print(vib)