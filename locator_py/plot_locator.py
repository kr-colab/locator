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
<<<<<<< HEAD
import scipy.interpolate
=======
import seaborn as sns
>>>>>>> e3e1dfa35179895845c8ac1fc9aabaa133daf128
import sys
import zarr
from sklearn.neighbors import KernelDensity

## ADD ARGUMENTS

parser = argparse.ArgumentParser(description = "Plot summary of a set of locator predictions.")
parser.add_argument('--infile', help = "path to folder with .predlocs files")
parser.add_argument('--sample_data', help = "path to sample_data file (should be WGS1984 x / y if map=TRUE.")
parser.add_argument('--out', help = "path to output (will be appended with _typeofplot.pdf)")
parser.add_argument('--width', default = 10, type = float, help = "width in inches of the output map. default = 5")
parser.add_argument('--sample', default = None, type = str, help = "sample ID to plot. if no argument is provided, a random sample will be plotted")
parser.add_argument('--error', default = False, type = bool, help = "calculate error and plot summary? requires known locations for all samples. True / False. default = False")
#parser.add_argument('--legend_position', default = "bottom", help = "legend position for summary plots if --error is True. Options: 'bottom', 'right'. default = bottom")
parser.add_argument('--map', default = True, type = str, help = "plot  basemap? default = True")
#parser.add_argument('--haploid', default = False, type = bool, help = "set to TRUE if predictions are from locator_phased.py. Predictions will be plotted for each haploid chromosome separately. default: FALSE.")
parser.add_argument('--centroid_method', default = 'kd', type = str, help = "Method for summarizing window/bootstrap predictions. Options 'gc' (take the centroid of window predictions with rgeos::gCentroid() ) or 'kd' (take the location of maximum density after kernal density estimation with mass::kde( )). default: kd")



args=parser.parse_args()

infile = args.infile
sample_data = args.sample_data
out = args.out
width = args.width
error = args.error
sample = args.sample
usemap = args.map
#haploid = args.haploid
#legend_position = args.legend_position
centroid_method = args.centroid_method

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

## CLOSEST FLOAT

def get_closest(floats, value):
    difference = (np.abs(floats - value))
    closest = difference.argmin()
    return closest
    
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
aeg = pd.merge(aeg, locs, on='sampleID')
aeg = aeg.rename(columns={'0': 'xpred', '1': 'ypred'})
samples = aeg.sampleID.unique()

## CALCULATE ERROR

if error != False:
    print('calculating error')
    ids = np.empty(len(samples), dtype = 'object')
    if centroid_method == 'kd':
        kd_x = np.empty(len(samples))
        kd_y = np.empty(len(samples))
        bp = {'sampleID': ids, 'kd_x': kd_x, 'kd_y': kd_y}
        count = 0
        for item in samples:
            seriesObj = aeg.apply(lambda x: True if x['sampleID'] == item else False , axis=1)
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
            ids[count] = item
            count += 1
        bp.update({'sampleID': ids, 'kd_x': kd_x, 'kd_y': kd_y})
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
        
    elif centroid_method == 'gc':
        gc_x = np.empty(len(samples))
        gc_y = np.empty(len(samples))
        bp = {'sampleID': ids, 'gc_x': gc_x, 'gc_y': gc_y}
        count = 0
        for item in samples:
            seriesObj = aeg.apply(lambda x: True if x['sampleID'] == item else False , axis=1)
            xcoords = np.empty(len(seriesObj[seriesObj == True].index))
            ycoords = np.empty(len(seriesObj[seriesObj == True].index))
            ticker = 0
            for index, row in (aeg.loc[aeg['sampleID'] == item]).iterrows():
                xcoords[ticker] = row['xpred']
                ycoords[ticker] = row['ypred']
                ticker += 1
            g = centroid(xcoords, ycoords)
            gc_x[count] = g[0]
            gc_y[count] = g[1]
            ids[count] = item
            count += 1
        bp.update({'sampleID': ids, 'gc_x': gc_x, 'gc_y': gc_y})
        bp = pd.DataFrame(bp)
        bp.to_csv((out + '_centroids.txt'), index = False, sep = '\t')
        aeg = pd.merge(aeg, bp, on='sampleID')
        plocs = list(zip(aeg['gc_x'], aeg['gc_y']))
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

mapp = zarr.open('map.zarr', mode = 'r')
def plot(sample, locs):
    print("plotting")
    print(sample)
    
    data = (aeg[aeg['sampleID'].str.contains(sample)]).reset_index(drop = True)

    kde = KernelDensity(bandwidth=0.04, metric='haversine', kernel='gaussian', algorithm='ball_tree')
    Xtrain = np.vstack([data.ypred, data.xpred]).T
    Xtrain = np.radians(Xtrain)

    xgrid = np.linspace(min(data.xpred) - 10, max(data.xpred) + 10, (int((max(data.xpred) - min(data.xpred))) * 10))
    ygrid = np.linspace(min(data.ypred) - 10, max(data.ypred) + 10, (int((max(data.ypred) - min(data.ypred))) * 10))
    Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)

    kde.fit(Xtrain)

    xy = np.vstack([Ygrid.ravel(), Xgrid.ravel()]).T
    xy = np.radians(xy)

    Z = np.exp(kde.score_samples(xy))

    zed = np.sort(Z)
    c1 = np.cumsum(zed)
    y_interp = scipy.interpolate.interp1d(c1, zed)

    ex = np.linspace(min(zed), max(zed), len(np.ndarray.flatten(Xgrid + Ygrid)))
    why = y_interp(zed)
    vals = list(zip(ex, why))
    quants = np.quantile(why, [.05, .5, .9])

    levels = []

    for q in quants:
        close = get_closest(why, q)
        levels.append(vals[close][0])
    
    Z = Z.reshape(Xgrid.shape)   

    fig,ax=plt.subplots()
    colors = ['#000000', '#1e90ff', '#ff0000']
    texts = ['Predicted locations', 'Training locations', 'Actual location']
    patches = [ plt.plot([],[], marker="o", ms=10, ls="", mec=None, color=colors[i], 
    label="{:s}".format(texts[i]) )[0]  for i in range(len(texts)) ]
    legend = plt.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=3, facecolor="w", edgecolor = None, numpoints=1 )
    ax.set_aspect('equal')
    ax.set_xlim((min(data.xpred) - 10), (max(data.xpred)+10))
    ax.set_ylim((min(data.ypred) - 10), (max(data.ypred)+10))
    ax.patch.set_facecolor('#ffffff')
    if usemap == True:
        itm = []
        for group in mapp:
            if group == 'Mozambique':
                for item in mapp[group]:
                    if item == 'Mozambique_2':
                        itm.append((mapp[group][item][0], mapp[group][item][1]))
                ax.plot(itm[0][0][0:(len(itm[0][0]) - 11)], itm[0][1][0:(len(itm[0][1]) - 11)], '#ffffff', lw = .15)
                ax.fill(itm[0][0][0:(len(itm[0][0]) - 11)], itm[0][1][0:(len(itm[0][1]) - 11)], '#b0b0b0')
            elif group != 'Mozambique':
                for item in mapp[group]:
                    x = mapp[group][item][0]
                    y = mapp[group][item][1]
                    ax.plot(x, y, '#ffffff', lw = .15)
                    ax.fill(x, y, '#b0b0b0')
    ax1 = fig.add_subplot()
    ax1.set_xlim((min(data.xpred) - 10), (max(data.xpred)+10))
    ax1.set_ylim((min(data.ypred) - 10), (max(data.ypred)+10))
    ax1.set_aspect('equal')
    contour = ax1.contour(Xgrid, Ygrid, Z, levels = levels, colors = 'k')

    fmt = {}
    strs = ['0.95', '0.5', '0.1']
    for l, s in zip(contour.levels, strs):
        fmt[l] = s


    ax1.clabel(contour, contour.levels, inline=True, fmt=fmt, fontsize=10)
    ax1.scatter(data.xpred, data.ypred, s = 1, color = '#000000')
    ax1.scatter(locs.x, locs.y, s = 10, color = '#1e90ff') 
    ax1.scatter(data.x[0], data.y[0], s = 40, color = '#FF0000')
    ax1.patch.set_alpha(0)
    title = fig.suptitle(sample, fontweight = 'bold')
    plt.savefig(out + '_plot_map.png', bbox_extra_artists = (title, legend), bbox_inches = 'tight')
    
if sample == None:
    sample = samples[np.random.randint(len(samples))]
    plot(sample, locs)
elif sample != None:
    sample = sample
    plot(sample, locs)