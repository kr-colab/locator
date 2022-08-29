#estimating sample locations from genotype matrices
import allel, re, os, matplotlib, sys, zarr, time, subprocess, copy
import numpy as np, pandas as pd, tensorflow as tf
from scipy import spatial
from tqdm import tqdm
from matplotlib import pyplot as plt
import argparse
import json
from tensorflow.keras import backend as K
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

parser=argparse.ArgumentParser()
parser.add_argument("--vcf",help="VCF with SNPs for all samples.")
parser.add_argument("--zarr", help="zarr file of SNPs for all samples.")
parser.add_argument("--matrix",help="tab-delimited matrix of minor allele counts with first column named 'sampleID'.\
                                     E.g., \
                                     \
                                     sampleID\tsite1\tsite2\t...\n \
                                     msp1\t0\t1\t...\n \
                                     msp2\t2\t0\t...\n ")
parser.add_argument("--sample_data",
                    help="tab-delimited text file with columns\
                         'sampleID \t x \t y'.\
                          SampleIDs must exactly match those in the \
                          VCF. X and Y values for \
                          samples without known locations should \
                          be NA.")
parser.add_argument("--train_split",default=0.9,type=float,
                    help="0-1, proportion of samples to use for training. \
                          default: 0.9 ")
parser.add_argument("--windows",default=False,action="store_true",
                    help="Run windowed analysis over a single chromosome (requires zarr input).")
parser.add_argument("--window_start",default=0,help="default: 0")
parser.add_argument("--window_stop",default=None,help="default: max snp position")
parser.add_argument("--window_size",default=5e5,help="default: 500000")
parser.add_argument("--bootstrap",default=False,action="store_true",
                    help="Run bootstrap replicates by retraining on bootstrapped data.")
parser.add_argument("--jacknife",default=False,action="store_true",
                    help="Run jacknife uncertainty estimate on a trained network. \
                    NOTE: we recommend this only as a fast heuristic -- use the bootstrap \
                    option or run windowed analyses for final results.")
parser.add_argument("--jacknife_prop",default=0.05,type=float,
                    help="proportion of SNPs to remove for jacknife resampling.\
                    default: 0.05")
parser.add_argument("--nboots",default=50,type=int,
                    help="number of bootstrap replicates to run.\
                    default: 50")
parser.add_argument("--batch_size",default=32,type=int,
                    help="default: 32")
parser.add_argument("--max_epochs",default=5000,type=int,
                    help="default: 5000")
parser.add_argument("--patience",type=int,default=100,
                    help="n epochs to run the optimizer after last \
                          improvement in validation loss. \
                          default: 100")
parser.add_argument("--min_mac",default=2,type=int,
                    help="minimum minor allele count.\
                          default: 2.")
parser.add_argument("--max_SNPs",default=None,type=int,
                    help="randomly select max_SNPs variants to use in the analysis \
                    default: None.")
parser.add_argument("--impute_missing",default=False,action="store_true",
                    help='default: True (if False, all alleles at missing sites are ancestral)')
parser.add_argument("--dropout_prop",default=0.25,type=float,
                     help="proportion of weights to zero at the dropout layer. \
                           default: 0.25")
parser.add_argument("--nlayers",default=10,type=int,
                    help="number of layers in the network. \
                        default: 10")
parser.add_argument("--width",default=256,type=int,
                    help="number of units per layer in the network\
                    default:256")
parser.add_argument("--out",help="file name stem for output")
parser.add_argument("--seed",default=None,type=int,
                    help="random seed for train/test splits and SNP subsetting.")
parser.add_argument('--tfseed', default=None, type=int,
                    help='random seed for TensorFlow initialization.')
parser.add_argument("--gpu_number",default=None,type=str)
parser.add_argument('--plot_history',default=True,type=bool,
                    help="plot training history? \
                    default: True")
parser.add_argument('--gnuplot',default=False,action="store_true",
                    help="print acii plot of training history to stdout? \
                    default: False")
parser.add_argument('--keep_weights',default=False,action="store_true",
                    help='keep model weights after training? \
                    default: False.')
parser.add_argument('--load_params',default=None,type=str,
                    help='Path to a _params.json file to load parameters from a previous run.\
                          Parameters from the json file will supersede all parameters provided \
                          via command line.')
parser.add_argument('--keras_verbose',default=1,type=int,
                    help='verbose argument passed to keras in model training. \
                    0 = silent. 1 = progress bars for minibatches. 2 = show epochs. \
                    Yes, 1 is more verbose than 2. Blame keras. \
                    default: 1. ')
parser.add_argument('--weight_samples',choices=[None, 'tsv', 'histogram', 'kernel density'],default=None,
                    help='Weight samples according to spatial density? \
                            "tsv" = manually assign sample weights. must provide --sample_weights argument. \
                            "histogram" = calculate weights using a 2D histogram. optional: \
                                provide --bins argument to define number of x and y bins used. \
                            "kernel density" = calculate weights using a gaussian kernel. optional: \
                                provide --bandwidth argument to define KDE bandwidth, \
                                        --lam argument to scale assigned weights.')
parser.add_argument('--sample_weights',default=None,
                    help='path to TSV of sample weights to use during training. \
                         columns = ["sampleID", "sample_weight"]')
parser.add_argument('--bins', default=None, nargs=2, type=int,
                    help='number of bins to use for histogram weight calculations. \
                            first argument is x bin count, second is y bin count')
parser.add_argument('--lam', default=1, type=float, help='factor to scale kernel density weights by:\
                    sample_weights = sample_weights ^ lam.')
parser.add_argument('--bandwidth', default=None, type=float, help='bandwidth for fitting kernel density estimate to landscape. Default is found using GridSearchCV.')

args=parser.parse_args()

#set seed and gpu
if args.seed is not None:
    np.random.seed(args.seed)
if args.tfseed is not None:
    tf.random.set_seed(args.tfseed)
if args.gpu_number is not None:
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_number

#load old run parameters
if args.load_params is not None:
    with open(args.predict_from_weights+"_params", 'r') as f:
        args.__dict__ = json.load(f)
    f.close()

#store run params
with open(args.out+'_params.json', 'w') as f:
    json.dump(args.__dict__, f, indent=2)
f.close()

def load_genotypes():
    if args.zarr is not None:
        print("reading zarr")
        callset = zarr.open_group(args.zarr, mode='r')
        gt = callset['calldata/GT']
        genotypes = allel.GenotypeArray(gt[:])
        samples = callset['samples'][:]
        positions = callset['variants/POS']
    elif args.vcf is not None:
        print("reading VCF")
        vcf=allel.read_vcf(args.vcf,log=sys.stderr)
        genotypes=allel.GenotypeArray(vcf['calldata/GT'])
        samples=vcf['samples']
    elif args.matrix is not None:
        gmat=pd.read_csv(args.matrix,sep="\t")
        samples=np.array(gmat['sampleID'])
        gmat=gmat.drop(labels="sampleID",axis=1)
        gmat=np.array(gmat,dtype="int8")
        for i in range(gmat.shape[0]): #kludge to get haplotypes for reading in to allel.
            h1=[];h2=[]
            for j in range(gmat.shape[1]):
                count=gmat[i,j]
                if count==0:
                    h1.append(0)
                    h2.append(0)
                elif count==1:
                    h1.append(1)
                    h2.append(0)
                elif count==2:
                    h1.append(1)
                    h2.append(1)
            if i==0:
                hmat=h1
                hmat=np.vstack((hmat,h2))
            else:
                hmat=np.vstack((hmat,h1))
                hmat=np.vstack((hmat,h2))
        genotypes=allel.HaplotypeArray(np.transpose(hmat)).to_genotypes(ploidy=2)
    return genotypes,samples

def sort_samples(samples):
    sample_data=pd.read_csv(args.sample_data,sep="\t")
    sample_data['sampleID2']=sample_data['sampleID']
    sample_data.set_index('sampleID',inplace=True)
    samples = samples.astype('str')
    sample_data=sample_data.reindex(np.array(samples)) #sort loc table so samples are in same order as vcf samples
    if not all([sample_data['sampleID2'][x]==samples[x] for x in range(len(samples))]): #check that all sample names are present
        print("WARNING: not all genotype samples are present in the metadata.\n \
                running on samples with available metadata...")
        sample_data = sample_data.loc[~pd.isna(sample_data.sampleID2)]
    locs=np.array(sample_data[["x","y"]])
    print("loaded "+str(np.shape(genotypes))+" genotypes\n\n")
    return(sample_data,locs)

def make_kd_weights(trainlocs, lam, bandwidth):
    if bandwidth:
        bw = bandwidth
    else:
    # use gridsearch to ID best bandwidth size
        bandwidths = np.linspace(0.1, 10, 1000)
        grid = GridSearchCV(KernelDensity(kernel='gaussian'), {'bandwidth':bandwidths})
        grid.fit(trainlocs)
        bw = grid.best_params_['bandwidth']
    
    # fit kernel
    kde = KernelDensity(bandwidth=bw, kernel='gaussian')
    kde.fit(trainlocs)

    # calculate weights
    weights = kde.score_samples(trainlocs)
    weights = 1.0 / np.exp(weights)
    weights /= min(weights)

    weights = np.power(weights, lam)

    weights /= sum(weights)

    return weights

def make_histogram_weights(trainlocs, bins):
    if bins:
        bincount = bins
    else:
        bincount = [10, 10] # default Numpy behavior

    # make 2D histogram
    H, xedges, yedges = np.histogram2d(trainlocs[:,0], trainlocs[:, 1], bins=bincount)
    # sort trainlocs into bins
    xbin = np.digitize(trainlocs[:, 0], xedges[1:], right=True)
    ybin = np.digitize(trainlocs[:, 1], yedges[1:], right=True)

    # assign sample weights
    weights = np.empty(len(trainlocs), dtype='float')
    for i in range(len(trainlocs)):
        weights[i] = 1/(H[xbin[i]][ybin[i]])
    weights /= min(weights)

    return weights

def load_sample_weights(weightpath, trainsamps):
 
    weightdf = pd.read_csv(weightpath, sep='\t')
    weightdf.set_index('sampleID', inplace=True)
    
    weights = np.empty(len(trainsamps), dtype='float')
    for i in range(len(trainsamps)):
        w = weightdf.loc[trainsamps[i], 'sample_weight']
        if type(w) == pd.core.series.Series:
            weights[i] = w[0]
        else:
            weights[i] = w 
    return np.array(weights)

#replace missing sites with binomial(2,mean_allele_frequency)
def replace_md(genotypes):
    print("imputing missing data")
    dc=genotypes.count_alleles()[:,1]
    ac=genotypes.to_allele_counts()[:,:,1]
    missingness=genotypes.is_missing()
    ninds=np.array([np.sum(x) for x in ~missingness])
    af=np.array([dc[x]/(2*ninds[x]) for x in range(len(ninds))])
    for i in tqdm(range(np.shape(ac)[0])):
        for j in range(np.shape(ac)[1]):
            if(missingness[i,j]):
                ac[i,j]=np.random.binomial(2,af[i])
    return ac

def filter_snps(genotypes):
    print("filtering SNPs")
    tmp=genotypes.count_alleles()
    biallel=tmp.is_biallelic()
    genotypes=genotypes[biallel,:,:]
    if not args.min_mac==1:
        derived_counts=genotypes.count_alleles()[:,1]
        ac_filter=[x >= args.min_mac for x in derived_counts]
        genotypes=genotypes[ac_filter,:,:]
    if args.impute_missing:
        ac=replace_md(genotypes)
    else:
        ac=genotypes.to_allele_counts()[:,:,1]
    if not args.max_SNPs==None:
        ac=ac[np.random.choice(range(ac.shape[0]),args.max_SNPs,replace=False),:]
    print("running on "+str(len(ac))+" genotypes after filtering\n\n\n")
    return ac

def normalize_locs(locs):
    meanlong=np.nanmean(locs[:,0])
    sdlong=np.nanstd(locs[:,0])
    meanlat=np.nanmean(locs[:,1])
    sdlat=np.nanstd(locs[:,1])
    locs=np.array([[(x[0]-meanlong)/sdlong,(x[1]-meanlat)/sdlat] for x in locs])
    return meanlong,sdlong,meanlat,sdlat,locs

def split_train_test(ac,locs):
    train=np.argwhere(~np.isnan(locs[:,0]))
    train=np.array([x[0] for x in train])
    pred=np.array([x for x in range(len(locs)) if not x in train])
    test=np.random.choice(train,
                          round((1-args.train_split)*len(train)),
                          replace=False)
    train=np.array([x for x in train if x not in test])
    traingen=np.transpose(ac[:,train])
    trainlocs=locs[train]
    testgen=np.transpose(ac[:,test])
    testlocs=locs[test]
    predgen=np.transpose(ac[:,pred])
    return train,test,traingen,testgen,trainlocs,testlocs,pred,predgen

def load_network(traingen,dropout_prop):
    from tensorflow.keras import backend as K
    def euclidean_distance_loss(y_true, y_pred):
        return K.sqrt(K.sum(K.square(y_pred - y_true),axis=-1))
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.BatchNormalization(input_shape=(traingen.shape[1],)))
    for i in range(int(np.floor(args.nlayers/2))):
        model.add(tf.keras.layers.Dense(args.width,activation="elu"))
    model.add(tf.keras.layers.Dropout(args.dropout_prop))
    for i in range(int(np.ceil(args.nlayers/2))):
        model.add(tf.keras.layers.Dense(args.width,activation="elu"))
    model.add(tf.keras.layers.Dense(2))
    model.add(tf.keras.layers.Dense(2))
    model.compile(optimizer="Adam",
                  loss=euclidean_distance_loss)
    return model

def load_callbacks(boot):
    if args.bootstrap or args.jacknife:
        checkpointer=tf.keras.callbacks.ModelCheckpoint(
                      filepath=args.out+"_boot"+str(boot)+"_weights.hdf5",
                      verbose=args.keras_verbose,
                      save_best_only=True,
                      save_weights_only=True,
                      monitor="val_loss",
                      period=1)
    else:
        checkpointer=tf.keras.callbacks.ModelCheckpoint(
                      filepath=args.out+"_weights.hdf5",
                      verbose=args.keras_verbose,
                      save_best_only=True,
                      save_weights_only=True,
                      monitor="val_loss",
                      period=1)
    earlystop=tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                            min_delta=0,
                                            patience=args.patience)
    reducelr=tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                               factor=0.5,
                                               patience=int(args.patience/6),
                                               verbose=args.keras_verbose,
                                               mode='auto',
                                               min_delta=0,
                                               cooldown=0,
                                               min_lr=0)
    return checkpointer,earlystop,reducelr

def train_network(model,traingen,testgen,trainlocs,testlocs,sample_weights):
    history = model.fit(traingen, trainlocs,
                        epochs=args.max_epochs,
                        batch_size=args.batch_size,
                        sample_weight=sample_weights,
                        shuffle=True,
                        verbose=args.keras_verbose,
                        validation_data=(testgen,testlocs),
                        callbacks=[checkpointer,earlystop,reducelr])
    if args.bootstrap or args.jacknife:
        model.load_weights(args.out+"_boot"+str(boot)+"_weights.hdf5")
    else:
        model.load_weights(args.out+"_weights.hdf5")
    return history,model

def predict_locs(model,predgen,sdlong,meanlong,sdlat,meanlat,testlocs,pred,samples,testgen,verbose=True):
    if verbose==True:
        print("predicting locations...")
    prediction=model.predict(predgen)
    prediction=np.array([[x[0]*sdlong+meanlong,x[1]*sdlat+meanlat] for x in prediction])
    predout=pd.DataFrame(prediction)
    predout.columns=['x','y']
    predout['sampleID']=samples[pred]
    if args.bootstrap or args.jacknife:
        predout.to_csv(args.out+"_boot"+str(boot)+"_predlocs.txt",index=False)
        testlocs2=np.array([[x[0]*sdlong+meanlong,x[1]*sdlat+meanlat] for x in testlocs])
    elif args.windows:
        predout.to_csv(args.out+"_"+str(i)+"-"+str(i+size-1)+"_predlocs.txt",index=False) # this is dumb
        testlocs2=np.array([[x[0]*sdlong+meanlong,x[1]*sdlat+meanlat] for x in testlocs])
    else:
        predout.to_csv(args.out+"_predlocs.txt",index=False)
        testlocs2=np.array([[x[0]*sdlong+meanlong,x[1]*sdlat+meanlat] for x in testlocs])
    p2=model.predict(testgen) #print validation loss to screen
    p2=np.array([[x[0]*sdlong+meanlong,x[1]*sdlat+meanlat] for x in p2])
    r2_long=np.corrcoef(p2[:,0],testlocs2[:,0])[0][1]**2
    r2_lat=np.corrcoef(p2[:,1],testlocs2[:,1])[0][1]**2
    mean_dist=np.mean([spatial.distance.euclidean(p2[x,:],testlocs2[x,:]) for x in range(len(p2))])
    median_dist=np.median([spatial.distance.euclidean(p2[x,:],testlocs2[x,:]) for x in range(len(p2))])
    dists=[spatial.distance.euclidean(p2[x,:],testlocs2[x,:]) for x in range(len(p2))]
    if verbose==True:
        print("R2(x)="+str(r2_long)+"\nR2(y)="+str(r2_lat)+"\n"
               +"mean validation error "+str(mean_dist)+"\n"
               +"median validation error "+str(median_dist)+"\n")
    hist=pd.DataFrame(history.history)
    hist.to_csv(args.out+"_history.txt",sep="\t",index=False)
    return(dists)

def plot_history(history,dists,gnuplot):
    if args.plot_history:
        plt.switch_backend('agg')
        fig = plt.figure(figsize=(4,1.5),dpi=200)
        plt.rcParams.update({'font.size': 7})
        ax1=fig.add_axes([0,0,0.4,1])
        ax1.plot(history.history['val_loss'][3:],"-",color="black",lw=0.5)
        ax1.set_xlabel("Validation Loss")
        ax2=fig.add_axes([0.55,0,0.4,1])
        ax2.plot(history.history['loss'][3:],"-",color="black",lw=0.5)
        ax2.set_xlabel("Training Loss")
        fig.savefig(args.out+"_fitplot.pdf",bbox_inches='tight')
        if gnuplot:
            gp.plot(np.array(history.history['val_loss'][3:]),
                    unset='grid',
                    terminal='dumb 60 20',
                    #set= 'logscale y',
                    title='Validation Loss by Epoch')
            gp.plot((np.array(dists),
                     dict(histogram = 'freq',binwidth=np.std(dists)/5)),
                    unset='grid',
                    terminal='dumb 60 20',
                    title='Test Error')


### windows ###
if args.windows:
    callset = zarr.open_group(args.zarr, mode='r')
    gt = allel.GenotypeDaskArray(callset['calldata/GT'])
    samples = callset['samples'][:].astype('str')
    positions = np.array(callset['variants/POS'])
    
    # cut down genotypes to only the included samples
    metadata = pd.read_csv(args.sample_data, sep='\t')
    samples_list = list(samples)
    samples_callset_index = [samples_list.index(s) for s in metadata['sampleID']]
    samples = np.array([samples[s] for s in samples_callset_index])
    metadata['callset_index'] = samples_callset_index
    indexes = metadata.callset_index.values.sort()

    start=int(args.window_start)
    if args.window_stop==None:
        stop=np.max(positions)
    else:
        stop=int(args.window_stop)
    size=int(args.window_size)
    for i in np.arange(start,stop,size):
        mask=np.logical_and(positions >= i,positions < i+size)
        a=np.min(np.argwhere(mask))
        b=np.max(np.argwhere(mask))
        print(a,b)
        genotypes=allel.GenotypeArray(gt[a:b])
        genotypes=genotypes.take(samples_callset_index, axis=1)
        sample_data,locs=sort_samples(samples)
        unnormedlocs=locs # save un-normalized locs for sample weighting
        meanlong,sdlong,meanlat,sdlat,locs=normalize_locs(locs)
        ac=filter_snps(genotypes)
        checkpointer,earlystop,reducelr=load_callbacks("FULL")
        train,test,traingen,testgen,trainlocs,testlocs,pred,predgen=split_train_test(ac,locs)

        if args.weight_samples:
            if args.weight_samples == 'tsv':
                sample_weights = load_sample_weights(args.sample_weights, samples[train])
            elif args.weight_samples == 'histogram':
                sample_weights = make_histogram_weights(unnormedlocs[train], args.bins)
                wdf = pd.DataFrame({'sampleID':samples[train], 
                                    'sample_weight':sample_weights, 
                                    'x':unnormedlocs[train][:,0], 
                                    'y':unnormedlocs[train][:,1]})
                wdf.to_csv(args.out+'_sample_weights.txt', sep='\t')

            elif args.weight_samples == 'kernel density':
                sample_weights = make_kd_weights(unnormedlocs[train], args.lam, args.bandwidth)
                wdf = pd.DataFrame({'sampleID':samples[train], 
                                    'sample_weight':sample_weights, 
                                    'x':unnormedlocs[train][:,0], 
                                    'y':unnormedlocs[train][:,1]})
                wdf.to_csv(args.out+'_sample_weights.txt', sep='\t')
        else:
            sample_weights = None
        model=load_network(traingen,args.dropout_prop)
        t1=time.time()
        history,model=train_network(model,traingen,testgen,trainlocs,testlocs,sample_weights)
        dists=predict_locs(model,predgen,sdlong,meanlong,sdlat,meanlat,testlocs,pred,samples,testgen)
        plot_history(history,dists,args.gnuplot)
        if not args.keep_weights:
            subprocess.run("rm "+args.out+"_weights.hdf5",shell=True)
        t2=time.time()
        elapsed=t2-t1
        print("run time "+str(elapsed/60)+" minutes")
else:
    if not args.bootstrap and not args.jacknife:
        boot=None
        genotypes,samples=load_genotypes()
        sample_data,locs=sort_samples(samples)
        unnormedlocs=locs # save un-normalized locs for sample weighting
        meanlong,sdlong,meanlat,sdlat,locs=normalize_locs(locs)
        ac=filter_snps(genotypes)
        checkpointer,earlystop,reducelr=load_callbacks("FULL")
        train,test,traingen,testgen,trainlocs,testlocs,pred,predgen=split_train_test(ac,locs)
        if args.weight_samples:
            if args.weight_samples == 'tsv':
                sample_weights = load_sample_weights(args.sample_weights, samples[train])
            elif args.weight_samples == 'histogram':
                sample_weights = make_histogram_weights(unnormedlocs[train], args.bins)
                wdf = pd.DataFrame({'sampleID':samples[train], 
                                    'sample_weight':sample_weights, 
                                    'x':unnormedlocs[train][:,0], 
                                    'y':unnormedlocs[train][:,1]})
                wdf.to_csv(args.out+'_sample_weights.txt', sep='\t')
            elif args.weight_samples == 'kernel density':
                sample_weights = make_kd_weights(unnormedlocs[train], args.lam, args.bandwidth)
                wdf = pd.DataFrame({'sampleID':samples[train], 
                                    'sample_weight':sample_weights, 
                                    'x':unnormedlocs[train][:,0], 
                                    'y':unnormedlocs[train][:,1]})
                wdf.to_csv(args.out+'_sample_weights.txt', sep='\t')
        else:
            sample_weights = None
        model=load_network(traingen,args.dropout_prop)
        start=time.time()
        history,model=train_network(model,traingen,testgen,trainlocs,testlocs,sample_weights)
        dists=predict_locs(model,predgen,sdlong,meanlong,sdlat,meanlat,testlocs,pred,samples,testgen)
        plot_history(history,dists,args.gnuplot)
        if not args.keep_weights:
            subprocess.run("rm "+args.out+"_weights.hdf5",shell=True)
        end=time.time()
        elapsed=end-start
        print("run time "+str(elapsed/60)+" minutes")
    elif args.bootstrap:
        boot="FULL"
        genotypes,samples=load_genotypes()
        sample_data,locs=sort_samples(samples)
        unnormedlocs=locs
        meanlong,sdlong,meanlat,sdlat,locs=normalize_locs(locs)
        ac=filter_snps(genotypes)
        checkpointer,earlystop,reducelr=load_callbacks("FULL")
        train,test,traingen,testgen,trainlocs,testlocs,pred,predgen=split_train_test(ac,locs)
        if args.weight_samples:
            if args.weight_samples == 'tsv':
                sample_weights = load_sample_weights(args.sample_weights, samples[train])
            elif args.weight_samples == 'histogram':
                sample_weights = make_histogram_weights(unnormedlocs[train], args.bins)
                wdf = pd.DataFrame({'sampleID':samples[train], 
                                    'sample_weight':sample_weights, 
                                    'x':unnormedlocs[train][:,0], 
                                    'y':unnormedlocs[train][:,1]})
                wdf.to_csv(args.out+'_sample_weights.txt', sep='\t')
            elif args.weight_samples == 'kernel density':
                sample_weights = make_kd_weights(unnormedlocs[train], args.lam, args.bandwidth)
                wdf = pd.DataFrame({'sampleID':samples[train], 
                                    'sample_weight':sample_weights, 
                                    'x':unnormedlocs[train][:,0], 
                                    'y':unnormedlocs[train][:,1]})
                wdf.to_csv(args.out+'_sample_weights.txt', sep='\t')
        else:
            sample_weights = None
        model=load_network(traingen,args.dropout_prop)
        start=time.time()
        history,model=train_network(model,traingen,testgen,trainlocs,testlocs,sample_weights)
        dists=predict_locs(model,predgen,sdlong,meanlong,sdlat,meanlat,testlocs,pred,samples,testgen)
        plot_history(history,dists,args.gnuplot)
        if not args.keep_weights:
            subprocess.run("rm "+args.out+"_bootFULL_weights.hdf5",shell=True)
        end=time.time()
        elapsed=end-start
        print("run time "+str(elapsed/60)+" minutes")
        for boot in range(args.nboots):
            np.random.seed(np.random.choice(range(int(1e6)),1))
            checkpointer,earlystop,reducelr=load_callbacks(boot)
            print("starting bootstrap "+str(boot))
            traingen2=copy.deepcopy(traingen)
            testgen2=copy.deepcopy(testgen)
            predgen2=copy.deepcopy(predgen)
            site_order=np.random.choice(traingen2.shape[1],traingen2.shape[1],replace=True)
            traingen2=traingen2[:,site_order]
            testgen2=testgen2[:,site_order]
            predgen2=predgen2[:,site_order]
            model=load_network(traingen2,args.dropout_prop)
            start=time.time()
            history,model=train_network(model,traingen2,testgen2,trainlocs,testlocs,sample_weights)
            dists=predict_locs(model,predgen2,sdlong,meanlong,sdlat,meanlat,testlocs,pred,samples,testgen2)
            plot_history(history,dists,args.gnuplot)
            if not args.keep_weights:
                subprocess.run("rm "+args.out+"_boot"+str(boot)+"_weights.hdf5",shell=True)
            end=time.time()
            elapsed=end-start
            K.clear_session()
            print("run time "+str(elapsed/60)+" minutes\n\n")
    elif args.jacknife:
        boot="FULL"
        genotypes,samples=load_genotypes()
        sample_data,locs=sort_samples(samples)
        meanlong,sdlong,meanlat,sdlat,locs=normalize_locs(locs)
        unnormedlocs=locs
        ac=filter_snps(genotypes)
        checkpointer,earlystop,reducelr=load_callbacks(boot)
        train,test,traingen,testgen,trainlocs,testlocs,pred,predgen=split_train_test(ac,locs)
        if args.weight_samples:
            if args.weight_samples == 'tsv':
                sample_weights = load_sample_weights(args.sample_weights, samples[train])
            elif args.weight_samples == 'histogram':
                sample_weights = make_histogram_weights(unnormedlocs[train], args.bins)
                wdf = pd.DataFrame({'sampleID':samples[train], 
                                    'sample_weight':sample_weights, 
                                    'x':unnormedlocs[train][:,0], 
                                    'y':unnormedlocs[train][:,1]})
                wdf.to_csv(args.out+'_sample_weights.txt', sep='\t')
            elif args.weight_samples == 'kernel density':
                sample_weights = make_kd_weights(unnormedlocs[train], args.lam, args.bandwidth)
                wdf = pd.DataFrame({'sampleID':samples[train], 
                                    'sample_weight':sample_weights, 
                                    'x':unnormedlocs[train][:,0], 
                                    'y':unnormedlocs[train][:,1]})
                wdf.to_csv(args.out+'_sample_weights.txt', sep='\t')
        else:
            sample_weights = None       
        model=load_network(traingen,args.dropout_prop)
        start=time.time()
        history,model=train_network(model,traingen,testgen,trainlocs,testlocs,sample_weights)
        dists=predict_locs(model,predgen,sdlong,meanlong,sdlat,meanlat,testlocs,pred,samples,testgen)
        plot_history(history,dists,args.gnuplot)
        end=time.time()
        elapsed=end-start
        print("run time "+str(elapsed/60)+" minutes")
        print("starting jacknife resampling")
        af=[]
        for i in tqdm(range(ac.shape[0])):
            af.append(sum(ac[i,:])/(ac.shape[1]*2))
        af=np.array(af)
        for boot in tqdm(range(args.nboots)):
            checkpointer,earlystop,reducelr=load_callbacks(boot)
            pg=copy.deepcopy(predgen) #this asshole
            sites_to_remove=np.random.choice(pg.shape[1],int(pg.shape[1]*args.jacknife_prop),replace=False) #treat X% of sites as missing data
            for i in sites_to_remove:
                pg[:,i]=np.random.binomial(2,af[i],pg.shape[0])
                #pg[:,i]=af[i]
            dists=predict_locs(model,pg,sdlong,meanlong,sdlat,meanlat,testlocs,pred,samples,testgen,verbose=False) #TODO: check testgen behavior for printing R2 to screen with jacknife in predict mode
        if not args.keep_weights:
            subprocess.run("rm "+args.out+"_bootFULL_weights.hdf5",shell=True)

#ag1000g.phase1.ar3.pass.2L.0-5e6.zarr
###debugging params
# args=argparse.Namespace(vcf=None,#"/Users/cj/locator/data/test_genotypes.vcf.gz",
#                         matrix=None,#"/Users/cj/locator/data/test_genotypes.vcf.gz",
#                         zarr="/Users/cj/locator/data/test_genotypes.zarr",
#                         sample_data="/Users/cj/locator/data/test_sample_data.txt",
#                         train_split=0.9,
#                         windows=True,
#                         window_start=0,
#                         window_stop=None,
#                         window_size=2e5,
#                         seed=12345,
#                         boot=False,
#                         load_params=None,
#                         nboots=100,
#                         nlayers=8,
#                         jacknife=False,
#                         width=256,
#                         batch_size=32,
#                         max_epochs=5000,
#                         bootstrap=False,
#                         patience=20,
#                         impute_missing=True,
#                         max_SNPs=None,
#                         min_mac=2,
#                         gnuplot=True,
#                         out="/Users/cj/Desktop/test",
#                         plot_history='True',
#                         dropout_prop=0.25,
#                         gpu_number="0")
