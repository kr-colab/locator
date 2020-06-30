## IMPORT LIBRARIES

import argparse
import scipy as sp
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib import cm
from math import sin, cos, sqrt, atan2, radians
from sklearn.neighbors import KernelDensity as kd
import shapefile
import seaborn as sns
from shapely import geometry
import pyproj
import sys
import zarr

## ADD ARGUMENTS

parser = argparse.ArgumentParser(description = "Plot summary of a set of locator predictions.")
parser.add_argument('--infile', help = "path to folder with .predlocs files")
parser.add_argument('--sample_data', help = "path to sample_data file (should be WGS1984 x / y if map=TRUE.")
parser.add_argument('--out', help = "path to output (will be appended with _typeofplot.pdf)")
parser.add_argument('--width', default = 10, type = float, help = "width in inches of the output map. default = 5")
parser.add_argument('--sample', default = None, type = str, help = "sample ID to plot. if no argument is provided, a random sample will be plotted")
parser.add_argument('--error', default = False, type = bool, help = "calculate error and plot summary? requires known locations for all samples. T / F. default = F")
parser.add_argument('--legend_position', default = "bottom", help = "legend position for summary plots if --error is True. Options: 'bottom', 'right'. default = bottom")
parser.add_argument('--map', default = "T", type = str, help = "plot  basemap? default = T")
parser.add_argument('--haploid', default = False, type = bool, help = "set to TRUE if predictions are from locator_phased.py. Predictions will be plotted for each haploid chromosome separately. default: FALSE.")

args=parser.parse_args()

infile = args.infile
sample_data = args.sample_data
out = args.out
width = args.width
error = args.error
sample = args.sample
usemap = args.map
haploid = args.haploid
legend_position = args.legend_position

## KERNEL DENSITY

def kdepred (xcoords, ycoords):
    try:
        coords = list(zip(xcoords, ycoords))
        density = kd(kernel='gaussian', bandwidth=0.2).fit(coords) ## bandwidth
        e = density.score_samples(coords)
        max_index = int((np.argwhere(e == np.amax(e)).tolist())[0][0])
        kd_x = xcoords[max_index]
        kd_y = ycoords[max_index]
    except Exception:
        kd_x = np.mean(xcoords)
        kd_y = np.mean(ycoords)
    return(kd_x, kd_y)
    
## CENTROID

def centroid(xcoords, ycoords):
    arr = np.array(list(zip(xcoords, ycoords)))
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length
    
## LOAD, COMPILE DATA

print('loading data')
if 'predlocs.txt' in (x for x in os.listdir(infile)):
    aeg = pd.DataFrame(infile)
else:
    files = [x for x in os.listdir(infile)]
    files = [x for x in files if 'predlocs' in x]
    path = infile + '/' + files[0]
    aeg = pd.read_csv(path)
    for f in files:
        path = infile + '/' + f
        a = pd.read_csv(path)
        aeg = aeg.append(a, ignore_index = True)

locs = pd.read_csv(sample_data, sep = '\t')
samples = aeg.sampleID.unique()
aeg = pd.merge(aeg, locs, on='sampleID')
aeg = aeg.rename(columns={'0': 'xpred', '1': 'ypred'})

## CALCULATE ERROR

if error != False:
    print('calculating error')
    #bp = {}
    ids = np.empty(len(samples), dtype = 'object')
    kd_xy = np.empty(len(samples))
    kd_y = np.empty(len(samples))
    gc_x = np.empty(len(samples))
    gc_y = np.empty(len(samples))
    bp = {'sampleID': ids, 'kd_x': kd_x, 'kd_y': kd_y, 'gc_x': gc_x, 'gc_y': gc_y}
    count = 0
    for item in samples:
        seriesObj = aeg.apply(lambda x: True if x['sampleID'] == item else False , axis=1)
        #print(seriesObj)
        #numOfRows = len(seriesObj[seriesObj == True].index)
        xcoords = np.empty(len(seriesObj[seriesObj == True].index))
        ycoords = np.empty(len(seriesObj[seriesObj == True].index))
        ticker = 0
        for index, row in (aeg.loc[aeg['sampleID'] == item]).iterrows():
            xcoords[ticker] = row['xpred']
            ycoords[ticker] = row['ypred']
            ticker += 1
        k = kdepred(xcoords, ycoords)
        kd_x[count] = k[0]
        kd_y[count] = k[1]
        g = centroid(xcoords, ycoords)
        gc_x[count] = g[0]
        gc_y[count] = g[1]
        ids[count] = item
        count += 1
    bp.update({'sampleID': ids, 'kd_x': kd_x, 'kd_y': kd_y, 'gc_x': gc_x, 'gc_y': gc_y})
    bp = pd.DataFrame(bp)
    bp.to_csv((out + '_centroids.txt'), index = False, sep = '\t')
    aeg = pd.merge(aeg, bp, on='sampleID')

    plocs = list(zip(aeg['kd_x'], aeg['kd_y']))
    tlocs = list(zip(aeg['x'], aeg['y']))
    dists = []
    R = 6373.0
    for n in range(len(plocs)):
        xpred = radians(plocs[n][0])
        ypred = radians(plocs[n][1])
        x = radians(tlocs[n][0])
        y = radians(tlocs[n][1])
        dlon = xpred - x
        dlat = ypred - y
        a = sin(dlat / 2)**2 + cos(y) * cos(ypred) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        dists.append(R * c)
    aeg['dists'] = dists
    print('mean centroid error = ', np.mean(dists))
    print('median centroid error = ', np.median(dists))
    print('90% CI for centroid error ', np.quantile(dists, 0.05), ' ', np.quantile(dists, 0.95))

## PLOT

mapp = zarr.open('zarr/map.zarr', mode = 'r')
def plot(sample, locs):
    print("plotting")
    print(sample)
    xpred = []
    ypred = []
    truex = []
    truey = []
    trainx = []
    trainy = []
    for index, row in aeg.loc[aeg['sampleID'] == sample].iterrows():
        xpred.append(row['xpred'])
        ypred.append(row['ypred'])
        truex = row['x']
        truey = row['y']
    for index, row in locs.iterrows():
        trainx.append(row['x'])
        trainy.append(row['y'])
    xscale = (np.max(xpred)) - (np.min(xpred))
    yscale = (np.max(ypred)) - (np.min(ypred))
    xmin = ((np.min(xpred))-5)
    xmax = ((np.max(xpred))+5)
    ymin = ((np.min(ypred))-5)
    ymax = ((np.max(ypred))+5)
    ys = ymax - ymin
    xs = xmax - xmin
    ratio = ys/xs
    fig, ax = plt.subplots()
    ax.patch.set_facecolor('#fefdfc')
    if usemap == 'T':
        for group in mapp:
            for item in mapp[group]:
                x = mapp[group][item][0]
                y = mapp[group][item][1]
                ax.plot(x, y, '#fefdfc')
                ax.fill(x, y, '#C5CBCB')
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    ax1 = fig.add_subplot()
    cmap = sns.cubehelix_palette(8, start=.5, rot=-.75, dark = 1, light = .3, as_cmap = True)
    sns.kdeplot(xpred, ypred, n_levels=1000, cmap = cmap, shade = True, shade_lowest = False, ax = ax1, alpha=0.7)
    ax1.scatter(xpred, ypred, s = 10, color = '#080A0C', alpha = .9)
    ax1.scatter(trainx, trainy, s = 25, color = '#DAFFD6', edgecolor = '#080A0C', linewidth = .5)
    ax1.scatter(truex, truey, s = 40, color = '#E42535', edgecolor = '#080A0C', linewidth = .5)
    ax1.set_xlim(xmin,xmax)
    ax1.set_ylim(ymin,ymax)
    for x in [ax, ax1]:
        x.tick_params(color='w', labelcolor='k')
        for spine in x.spines.values():
            spine.set_edgecolor('w')
    ax1.spines['bottom'].set_color('k')
    ax1.spines['left'].set_color('k')
    ax1.patch.set_alpha(0)
    height = (width*ratio)
    fig.set_size_inches(width, (height))
    ax.set_aspect('equal')
    ax1.set_aspect('equal')
    true = mpl.lines.Line2D([], [], color='#E42535', marker='o',
                          markersize=5, label='Sample Location')
    pred = mpl.lines.Line2D([], [], color='#080A0C', marker='o',
                          markersize=5, label='Predicted locations')
    train = mpl.lines.Line2D([], [], color='#DAFFD6', marker='o',
                          markersize=5, label='Training locations')
    ##TODO: add legend stuff
    plt.savefig(out + '_plot_map.png')
    
if sample == None:
    sample = samples[np.random.randint(len(samples))]
    plot(sample, locs)
elif sample != None:
    sample = sample
    plot(sample, locs)
