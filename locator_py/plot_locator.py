import argparse, scipy as sp, pandas as pd, os, numpy as np, matplotlib as mpl, zarr
from matplotlib import pyplot as plt
from math import sin, cos, sqrt, atan2, radians
from sklearn.neighbors import KernelDensity
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

parser=argparse.ArgumentParser(description='Plot summary of a set of locator predictions.')
parser.add_argument('--infile',help='path to folder with .predlocs files')
parser.add_argument('--sample_data',help='path to sample_data file (should be WGS1984 x/y if map=True')
parser.add_argument('--out',help='path to output (will be appended with _typeofplot.pdf')
parser.add_argument('--width',default=10,type=float,help='width in inches of the output plot. default=5')
parser.add_argument('--height',default=8,type=float,help='height in inches of the output plot. default=4')
parser.add_argument('--samples',default=None,nargs='+',help='samples IDs to plot, separated by spaces. e.g. sample1 sample2 sample3. default=None')
parser.add_argument('--nsamples',default=9,type=int,help='if no --samples argument is provided, --nsamples random samples will be plotted. default=9')
parser.add_argument('--ncol',default=3,type=int,help='number of columns for multipanel plots (should evenly divide --nsamples, otherwise nsamples will supersede). default=3')
parser.add_argument('--error',default=True,action='store_true',help='calculate error and plot summary? requires known locations for all samples. T/F. default=False')
parser.add_argument('--plot',default=False,action='store_true',help='make plots of predictions? default=True')
parser.add_argument('--basemap',default=False,action='store_true',help='plot basemap? default=False')
parser.add_argument('--longlat',default=False,action='store_true',help='set to True if coordinates are x and y in decimal degrees to print error in kilometers. default=False')
parser.add_argument('--silence',default=False,action='store_true',help='no terminal output. T/F. default=True')
parser.add_argument('--training_samples',default=None,help='path to training metadata file for plotting training locations, if provided. default=None')
args=parser.parse_args()

if args.basemap:
    mapp=zarr.open('map.zarr',mode='r')

def kdepred(xcoords,ycoords): # kernel density
    try:
        coords=list(zip(xcoords,ycoords))
        density=KernelDensity(kernel='gaussian',bandwidth=0.2).fit(coords) # bandwidth
        e=density.score_samples(coords)
        max_index=int((np.argwhere(e==np.amax(e)).tolist())[0][0])
        kx=xcoords[max_index]
        ky=ycoords[max_index]
    except Exception:
        print('oops')
        kx=np.mean(xcoords)
        ky=np.mean(ycoords)
    return kx,ky

def centroid(xcoords,ycoords): # geographic centroid
    coords=np.array(list(zip(xcoords,ycoords)))
    length=coords.shape[0]
    sum_x=np.sum(coords[:,0])
    sum_y=np.sum(coords[:,1])
    return sum_x/length, sum_y/length

def distance_km(xpred,ypred,x,y): # distance in km if longlat==True
    dlon=xpred-x
    dlat=ypred-y
    a=sin(dlat/2)**2+cos(y)*cos(ypred)*sin(dlon/2)**2
    c=2*atan2(sqrt(a),sqrt(1-a))
    return 6373.0*c

def distance(xpred,ypred,x,y): # hypotenuse distance otherwise
    xc=xpred-x
    yc=ypred-y
    return np.hypot(xc,yc)

def get_closest(floats,value): # closest float
    diff=(np.abs(floats-value))
    return diff.argmin()

# load data
if not args.silence:
    print('loading data')
if 'predlocs.txt' in args.infile:
    aeg=pd.DataFrame(infile)
else:
    files=[x for x in os.listdir(args.infile)]
    files=[x for x in files if 'predlocs' in x]
    aeg=pd.read_csv(args.infile+'/'+files[0])
    for i in range(len(files[1:])):
        a=pd.read_csv(args.infile+'/'+files[i])
        aeg=aeg.append(a,ignore_index=True)
aeg=aeg.reset_index(drop=True)
aeg=aeg.rename(columns={'x':'xpred','y':'ypred'})
locs=pd.read_csv(args.sample_data,sep='\t')
aeg=pd.merge(aeg,locs,on='sampleID')
samples=aeg.sampleID.unique()
if args.samples:
    samples=args.samples
else:
    samples=np.random.choice(samples,args.nsamples,replace=False)

# calculate error
if args.error:
    if not args.silence:
        print('calculating error')
    # get error for centroids and max kernel density locations
    bp={}
    kd_x=[]
    kd_y=[]
    gc_x=[]
    gc_y=[]
    loc_x=[]
    loc_y=[]
    kd_dists=[]
    gc_dists=[]
    ids=aeg.sampleID.unique()
    for item in ids:
        xcoords=aeg.loc[aeg['sampleID']==item,'xpred'].tolist()
        ycoords=aeg.loc[aeg['sampleID']==item,'ypred'].tolist()
    # actual location
        x=aeg.loc[aeg['sampleID']==item]['x'].tolist()
        y=aeg.loc[aeg['sampleID']==item]['y'].tolist()
        loc_x.append(x[0])
        loc_y.append(y[0])
    # kd centroids
        k=kdepred(xcoords,ycoords)
        kd_x.append(k[0])
        kd_y.append(k[1])
        if args.longlat:
            kd_dists.append(distance_km(k[0],k[1],x[0],y[0]))
        else:
            kd_dists.append(distance(k[0],k[1],x[0],y[0]))
     # gc centroids
        g=centroid(xcoords,ycoords)
        gc_x.append(g[0])
        gc_y.append(g[1])
        if args.longlat:
            gc_dists.append(distance_km(g[0],g[1],x[0],y[0]))
        else:
            gc_dists.append(distance(g[0],g[1],x[0],y[0]))
    # save to tsv
    bp.update({'sampleID':ids,'x':loc_x,'y':loc_y,'kd_x':kd_x,'kd_y':kd_y,'gc_x':gc_x,'gc_y':gc_y})
    bp=pd.DataFrame(bp)
    bp.to_csv(args.out+'_centroids.txt',index=False,sep='\t')
    # print results
    if not args.silence:
        print('mean kernel peak error = ' + str(np.mean(kd_dists)))
        print('median kernel peak error = ' + str(np.median(kd_dists)))
        print('90% CI for kernel peak error = ' + str(np.quantile(kd_dists,0.05))+' '+str(np.quantile(kd_dists,0.95)))
        print('mean centroid error = ' + str(np.mean(gc_dists)))
        print('median centroid error = ' + str(np.median(gc_dists)))
        print('90% CI for centroid error = ' + str(np.quantile(gc_dists,0.05))+' '+str(np.quantile(gc_dists,0.95)))
if not args.plot:
    quit()
# plot
if args.plot:
    if args.training_samples: # get training samples
        training_samples=pd.read_csv(args.training_samples,sep='\t')
        ts_x=(training_samples['x'][~np.isnan(training_samples['x'].to_numpy())])
        ts_y=(training_samples['y'][~np.isnan(training_samples['y'].to_numpy())])
        print(ts_x,ts_y)
    if not args.silence:
        print('plotting')
    if args.samples:
        s=len(args.samples)
    else:
        s=args.nsamples
    r=s/args.ncol
    if r/r != 1: # if args.samples doesn't fit into args.columns, plot all samples provided and leave the rest empty
        s=args.nsamples+(args.nsamples%args.ncol)
        r=s/args.ncol
    r=int(r)

    ax_ratio=(args.width/args.ncol)/(args.height/r)
    fig,axes=plt.subplots(nrows=r,ncols=args.ncol)

    count=0
    for ax in axes.flatten():
        sample=samples[count]
        count+=1
        # kernel density
        data=(aeg[aeg['sampleID'].str.contains(sample)]).reset_index(drop=True)
        kde=KernelDensity(bandwidth=0.04,metric='haversine',kernel='gaussian',algorithm='ball_tree')
        Xtrain=np.vstack([data.ypred,data.xpred]).T
        Xtrain=np.radians(Xtrain)
        xgrid=np.linspace(min(data.xpred)-10,max(data.xpred)+10,(int((max(data.xpred)-min(data.xpred)))*10))
        ygrid=np.linspace(min(data.ypred)-10,max(data.ypred)+10,(int((max(data.ypred)-min(data.ypred)))*10))
        Xgrid,Ygrid=np.meshgrid(xgrid,ygrid)
        kde.fit(Xtrain)
        xy=np.vstack([Ygrid.ravel(),Xgrid.ravel()]).T
        xy=np.radians(xy)
        Z=np.exp(kde.score_samples(xy))
        zed=np.sort(Z)
        c1=np.cumsum(zed)
        y_interp=sp.interpolate.interp1d(c1,zed)
        ex=np.linspace(min(zed),max(zed),len(np.ndarray.flatten(Xgrid+Ygrid)))
        why=y_interp(zed)
        vals=list(zip(ex,why))
        quants=np.quantile(why,[.05,.5,.9])
        levels=[]
        for q in quants:
            close=get_closest(why,q)
            if vals[close][0] not in levels:
                levels.append(vals[close][0])
        Z=Z.reshape(Xgrid.shape)
        # plot params
        if ((max(data.xpred)+10)-(min(data.xpred)-10))>((max(data.ypred)+10)-(min(data.ypred)-10)):
            xmin,xmax=min(data.xpred)-10,max(data.xpred)+10
            ax.set_xlim(xmin,xmax)
            width=(max(data.xpred)+10)-(min(data.xpred)-10)
            height=width/ax_ratio
            center=np.mean([max(data.ypred)+10,min(data.ypred)-10])
            ymin,ymax=center-(height/2),center+height/2
            ax.set_ylim(ymin,ymax)
            ax.set_aspect('equal')
        elif ((max(data.ypred)+10)-(min(data.ypred)-10))>((max(data.xpred)+10)-(min(data.xpred)-10)):
            ymin,ymax=min(data.ypred)-10,max(data.ypred)+10
            ax.set_ylim(ymin,ymax)
            height=(max(data.ypred)+10)-(min(data.ypred)-10)
            width=ax_ratio*height
            center=np.mean([max(data.xpred)+10,min(data.xpred)-10])
            xmin,xmax=center-(width/2),center+(width/2)
            ax.set_xlim(xmin,xmax)
            ax.set_aspect('equal')
        ax.patch.set_facecolor('#ffffff')
        # plot map
        plot_ax=ax.inset_axes([0,0,1,1])
        plot_ax.set_xlim(xmin,xmax)
        plot_ax.set_ylim(ymin,ymax)
        if args.basemap:
            itm=[]
            for group in mapp:
                for item in mapp[group]:
                    if any(xmin<xcoord<xmax for xcoord in mapp[group][item][0]) or any(ymin<ycoord<ymax for ycoord in mapp[group][item][1]):
                        ax.plot(mapp[group][item][0],mapp[group][item][1],'#ffffff',lw=.15)
                        ax.fill(mapp[group][item][0],mapp[group][item][1],'#b0b0b0')
        [s.set_visible(False) for s in ax.spines.values()]
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # plot locs
        contour=plot_ax.contour(Xgrid,Ygrid,Z,levels=levels,colors='k')
        fmt={}
        strs=['0.95', '0.5', '0.1']
        for l,s in zip(contour.levels,strs):
            fmt[l]=s
        plot_ax.clabel(contour,contour.levels,inline=True,fmt=fmt,fontsize='small')
        plot_ax.scatter(data.xpred,data.ypred,s=1,color='#000000',label='Predicted Locations')
        plot_ax.scatter(data.x[0],data.y[0],s=40,color='#FF0000',label='Sample Location')
        if args.training_samples:
            plot_ax.scatter(ts_x,ts_y,s=10,color='#1e90ff',label='Training Locations')
        plot_ax.patch.set_alpha(0)
        plot_ax.set_title(sample,fontsize='small')
        handles,labels=plot_ax.get_legend_handles_labels()
    loc='lower center'
    col=len(handles)
    #    fig.subplots_adjust(bottom=.01)
    lgt=fig.legend(handles,labels,loc=loc,ncol=col,fontsize='small')
    fig.set_size_inches(args.width,args.height)
    plt.savefig(args.out+'.pdf',format='pdf',bbox_inches='tight')
