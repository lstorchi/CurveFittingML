import os
import math

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from scipy.signal import convolve2d

#################################################################

def evaluate_surface_smoothness(xy_coords, z_values, 
                                grid_resolution=100j):
    """
    Evaluates the smoothness of a surface defined by scattered 3D points.

    The function first interpolates the scattered (X, Y, Z) data onto a regular
    grid. It then calculates the Laplacian of the surface, which is a measure of
    its curvature. The mean of the absolute Laplacian values is returned as the
    smoothness metric. A smaller value indicates a smoother surface.

    Args:
        xy_coords (np.ndarray): A NumPy array of shape (N, 2) containing the
                                [X, Y] coordinates of the N data points.
        z_values (np.ndarray): A 1D NumPy array of shape (N,) containing the
                               Z-values corresponding to each [X, Y] coordinate.
        grid_resolution (complex): The number of points for the grid in each
                                   dimension. e.g., 100j creates a 100x100 grid.

    Returns:
        float: The smoothness metric (mean absolute Laplacian). Returns np.nan
               if interpolation fails.
    """
    if xy_coords.shape[0] != len(z_values):
        raise ValueError("The number of XY coordinates must match the number of Z values.")
    
    # 1. Create a regular grid to interpolate onto
    x_min, y_min = xy_coords.min(axis=0)
    x_max, y_max = xy_coords.max(axis=0)
    
    grid_x, grid_y = np.mgrid[x_min:x_max:grid_resolution, y_min:y_max:grid_resolution]
    
    # 2. Interpolate the scattered Z data onto the regular grid
    # 'cubic' interpolation is good for smooth surfaces, 'linear' is more robust
    try:
        grid_z = griddata(xy_coords, z_values, (grid_x, grid_y), method='cubic')
    except Exception:
        # Fallback to linear if cubic fails (e.g., not enough points)
        grid_z = griddata(xy_coords, z_values, (grid_x, grid_y), method='linear')

    # Handle cases where interpolation returns NaNs (e.g., outside convex hull)
    if np.isnan(grid_z).all():
        print("Warning: Interpolation resulted in a grid of NaNs. Cannot calculate smoothness.")
        return np.nan
        
    # Replace NaNs with the mean of the valid data for a stable calculation
    mean_z = np.nanmean(grid_z)
    grid_z[np.isnan(grid_z)] = mean_z

    # 3. Calculate the Laplacian using a kernel convolution
    # The Laplacian kernel approximates the second derivative (curvature)
    laplacian_kernel = np.array([
        [0,  1, 0],
        [1, -4, 1],
        [0,  1, 0]
    ])

    # Apply the convolution to get the Laplacian at each grid point
    laplacian_grid = convolve2d(grid_z, laplacian_kernel, mode='same', boundary='symm')
    
    # 4. Calculate the smoothness metric
    # The mean of the absolute Laplacian values. A higher value means less smooth.
    smoothness_metric = np.mean(np.abs(laplacian_grid))
    
    return smoothness_metric

#################################################################

if __name__ == "__main__":

    # dv1 best is 5
    filename1 = "dv1/train_predictions_model_5.txt"
    filename2 = "dv1/test_predictions_model_5.txt"
    df1 = pd.read_csv(filename1)
    df2 = pd.read_csv(filename2)
    # add header
    x1 = df1[['v', 'T']].values
    y1 = df1['RC'].values
    X1 = np.array(x1)
    Y1 = np.array(y1)
    x2 = df2[['v', 'T']].values
    y2 = df2['RC'].values
    X2 = np.array(x2)
    Y2 = np.array(y2)

    temp_values1 = np.unique(X1[:, 1])
    vib_values1 = np.unique(X1[:, 0])
    temp_values2 = np.unique(X2[:, 1])
    vib_values2 = np.unique(X2[:, 0])

    print("Training v values: ", vib_values1)
    print("Testing v values: ", vib_values2)

    if temp_values1.shape != temp_values2.shape:
        for i in range(len(temp_values1)):
            if temp_values1[i] != temp_values2[i]:
                print("ERROR Temperature values are different between training and testing datasets.")

    # merge x1 and x2 and y1 and y2
    X = np.vstack((X1, X2))
    Y = np.hstack((Y1, Y2)) 
    if temp_values1.shape == temp_values2.shape:
        for i in range(len(temp_values1)):
            if temp_values1[i] != temp_values2[i]:
                print("ERROR Temperature values are different between training and testing datasets.")
                exit()
    else: 
        print("Temperature values are consistent between training and testing datasets.")
        exit()

    # merge x1 and x2 and y1 and y2
    X = np.vstack((X1, X2))
    Y = np.hstack((Y1, Y2))

    # Evaluate the smoothness of the surface
    smoothness = evaluate_surface_smoothness(np.column_stack((X[:, 0], X[:, 1])), Y)
    print("Surface smoothness (lower is smoother):", smoothness)

    # create a surface and plot it
    xi = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
    yi = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100)
    Z = griddata((X[:, 0], X[:, 1]), Y, (xi[None, :], yi[:, None]), method='cubic')
    fig = plt.figure(figsize=(20, 15))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xi[None, :], yi[:, None], Z, cmap='viridis')
    ax.set_xlabel('Vibration (v)')
    ax.set_ylabel('Temperature (T)')
    ax.set_zlabel('RC')
    ax.set_title('Surface Plot of RC')
    plt.savefig("HeCS_surface_plot.png")

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    xdim1 = len(temp_values1)
    ydim1 = len(vib_values1)
    xdim2 = len(temp_values2)
    ydim2 = len(vib_values2)
    for xidx in range(xdim1):
        t = temp_values1[xidx]
        for yidx in range(ydim1):
            v =  vib_values1[yidx]
            x = float(t)
            y = float(v)
            if df1[(df1['v'] == v) & (df1['T'] == t)]['RC'].empty:
                print(f"Warning: No data for v={v}, T={t}")

                if df1[(df1['v'] == v)].empty:
                    print(f"Warning: No data for v={v}")
                if df1[(df1['T'] == t)].empty:
                    print(f"Warning: No data for T={t}")

                continue
            z = df1[(df1['v'] == v) & (df1['T'] == t)]['RC'].values[0]
            ax.scatter(x, y, z, marker="o", color="b")

    for xidx in range(xdim2):
        t = temp_values2[xidx]
        for yidx in range(ydim2):
            v =  vib_values2[yidx]
            x = float(t)
            y = float(v)
            if df2[(df2['v'] == v) & (df2['T'] == t)]['RC'].empty:
                print(f"Warning: No data for v={v}, T={t}")

                if df2[(df2['v'] == v)].empty:
                    print(f"Warning: No data for v={v}")
                if df2[(df2['T'] == t)].empty:
                    print(f"Warning: No data for T={t}")

                continue
            z = df2[(df2['v'] == v) & (df2['T'] == t)]['RC'].values[0]
            ax.scatter(x, y, z, marker="o", color="r")  

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.gcf().set_size_inches(20, 15)

    plt.savefig("HeCS_scatter_plot.png")
