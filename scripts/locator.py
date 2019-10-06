#estimating sample locations from genotype matrices
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    import allel, re, os, keras, matplotlib, sys, zarr, time, subprocess, copy
    import numpy as np, pandas as pd, tensorflow as tf
    from scipy import spatial
    from tqdm import tqdm
    from matplotlib import pyplot as plt
    import argparse
    import gnuplotlib as gp

parser=argparse.ArgumentParser()
parser.add_argument("--vcf",help="VCF with SNPs for all samples.")
parser.add_argument("--zarr", help="zarr file of SNPs for all samples.")
parser.add_argument("--sample_data",
                    help="tab-delimited text file with columns\
                         'sampleID \t x \t y'.\
                          SampleIDs must exactly match those in the \
                          VCF. X and Y values for \
                          samples without known locations should \
                          be NA. If a column named 'test' \
                          is included, samples with test==True will be \
                          used as the test set.")
parser.add_argument("--mode",default="cv",
                    help="'cv' splits the sample by train_split \
                          and predicts on the test set. \
                          'predict' extracts samples with non-NaN \
                          coordinates, splits those by train_split \
                          for training and model evaluation, and returns \
                          predictions for samples with NaN coordinates.")
parser.add_argument("--train_split",default=0.9,type=float,
                    help="0-1, proportion of samples to use for training. \
                          default: 0.9 ")
parser.add_argument("--bootstrap",default="False",type=str,
                    help="Run bootstrap replicates by retraining on bootstrapped data. True/False.\
                    default: False")
parser.add_argument("--jacknife",default="False",type=str,
                    help="Run jacknife uncertainty estimate on a trained network. \
                    NOTE: we recommend this only as a fast heuristic -- use the bootstrap \
                    option or run windowed analyses for final results.")
parser.add_argument("--jacknife_prop",default=0.05,type=float,
                    help="proportion of SNPs to remove for jacknife resampling")
parser.add_argument("--nboots",default=100,type=int,
                    help="number of bootstrap replicates to run.\
                    default: 50")
parser.add_argument("--batch_size",default=32,type=int,
                    help="default: 32")
parser.add_argument("--max_epochs",default=5000,type=int,
                    help="default: 5000")
parser.add_argument("--patience",type=int,default=100,
                    help="n epochs to run the optimizer after last \
                          improvement in test loss before stopping. \
                          default: 100")
parser.add_argument("--min_mac",default=2,type=int,
                    help="minimum minor allele count.\
                          default: 2.")
parser.add_argument("--max_SNPs",default=None,type=int,
                    help="randomly select max_SNPs variants to use in the analysis \
                    default: None.")
parser.add_argument("--impute_missing",default="True",type=str,
                    help='default: True (if False, all alleles at missing sites are ancestral)')
parser.add_argument("--dropout_prop",default=0.25,type=float,
                    help="proportion of weights to drop at the dropout layer. \
                          default: 0.25")
parser.add_argument("--nlayers",default=10,type=int,
                    help="if model=='dense', number of fully-connected \
                    layers in the network. \
                    default: 10")
parser.add_argument("--width",default=256,type=int,
                    help="if model==dense, width of layers in the network\
                    default:256")
parser.add_argument("--out",help="file name stem for output")
parser.add_argument("--seed",default=None,type=int,
                    help="random seed used for train/test splits and max_SNPs.")
parser.add_argument("--gpu_number",default=None,type=str)
parser.add_argument("--n_predictions",default=1,type=int,
                    help="if >1, number of predictions to generate \
                          for uncertainty estimation via droupout. \
                          default: 1. Note  (this output is not supported by\
                          plot_locator.R).")
parser.add_argument('--plot_history',default=True,type=bool,
                    help="plot training history? \
                    default: True")
parser.add_argument('--keep_weights',default='True',type=str,
                    help='keep model weights after training? \
                    default: True.')
#parser.add_argument('--predict_from_weights',default='False',type=str,
#                    help='load model weights and predict on all samples')
args=parser.parse_args()

if not args.seed==None:
    np.random.seed(args.seed)
if not args.gpu_number==None:
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_number

#load genotype matrices
def load_genotypes():
    if args.zarr is not None:
        print("reading zarr")
        callset = zarr.open_group(args.zarr, mode='r')
        gt = callset['calldata/GT']
        genotypes = allel.GenotypeArray(gt[:])
        samples = callset['samples'][:]
    else:
        print("reading VCF")
        vcf=allel.read_vcf(args.vcf,log=sys.stderr)
        genotypes=allel.GenotypeArray(vcf['calldata/GT'])
        samples=vcf['samples']
    return genotypes,samples

#sort sample data
def sort_samples(samples):
    sample_data=pd.read_csv(args.sample_data,sep="\t")
    sample_data['sampleID2']=sample_data['sampleID']
    sample_data.set_index('sampleID',inplace=True)
    sample_data=sample_data.reindex(np.array(samples)) #sort loc table so samples are in same order as vcf samples
    if not all([sample_data['sampleID2'][x]==samples[x] for x in range(len(samples))]): #check that all sample names are present
        print("sample ordering failed! Check that sample IDs match the VCF.")
        sys.exit()
    locs=np.array(sample_data[["longitude","latitude"]])
    print("loaded "+str(np.shape(genotypes))+" genotypes\n\n")
    return(sample_data,locs)


#replace missing sites with binomial(2,mean_allele_frequency)
def replace_md(genotypes,impute):
    if impute in [True,"TRUE","True","T","true"]:
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
    else:
       missingness=genotypes.is_missing()
       ac=genotypes.to_allele_counts()[:,:,1]
       ac[missingness]=-1
    return ac

#SNP filters
def filter_snps(genotypes):
    print("filtering SNPs")
    if not args.min_mac==1:
        derived_counts=genotypes.count_alleles()[:,1]
        ac_filter=[x >= args.min_mac for x in derived_counts] #drop SNPs with minor allele < min_mac
        genotypes=genotypes[ac_filter,:,:]
    if args.impute_missing in ['TRUE','true','True',"T","t",True]:
        ac=replace_md(genotypes,impute=True)
    else:
        ac=genotypes.to_allele_counts()[:,:,1]
    if not args.max_SNPs==None:
        ac=ac[np.random.choice(range(ac.shape[0]),args.max_SNPs,replace=False),:]
    print("running on "+str(len(ac))+" genotypes after filtering\n\n\n")
    return ac

#normalize coordinates
def normalize_locs(locs):
    meanlong=np.nanmean(locs[:,0])
    sdlong=np.nanstd(locs[:,0])
    meanlat=np.nanmean(locs[:,1])
    sdlat=np.nanstd(locs[:,1])
    locs=np.array([[(x[0]-meanlong)/sdlong,(x[1]-meanlat)/sdlat] for x in locs])
    return meanlong,sdlong,meanlat,sdlat,locs

def split_train_test(ac,locs):
    #split training, testing, and prediction sets
    if args.mode=="predict": #TODO: add manual test splits to pred mode
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
    elif args.mode=="cv": #cross-validation mode
        if 'test' in sample_data.keys():
            train=np.argwhere(sample_data['test']==False)
            test=np.argwhere(sample_data['test']==True)
            train=np.array([x[0] for x in train])
            test=np.array([x[0] for x in test])
        else:
            train=np.random.choice(range(len(locs)),
                                   round(args.train_split*len(locs)),
                                   replace=False)
            test=np.array([x for x in range(len(locs)) if not x in train])
        pred=test
        traingen=np.transpose(ac[:,train])
        trainlocs=locs[train]
        testgen=np.transpose(ac[:,test])
        testlocs=locs[test]
        predgen=testgen
    return train,test,traingen,testgen,trainlocs,testlocs,pred,predgen

def load_network(traingen,dropout_prop):
    from keras.models import Sequential
    from keras import layers
    from keras.layers.core import Lambda
    from keras import backend as K
    import keras
    def euclidean_distance_loss(y_true, y_pred):
        return K.sqrt(K.sum(K.square(y_pred - y_true),axis=-1))
    model = Sequential()
    model.add(layers.BatchNormalization(input_shape=(traingen.shape[1],)))
    for i in range(int(np.floor(args.nlayers/2))):
        model.add(layers.Dense(args.width,activation="elu"))
    if args.dropout_prop > 0:
        if args.n_predictions > 1:
            model.add(Lambda(lambda x: K.dropout(x, level=args.dropout_prop)))
        else:
            model.add(layers.Dropout(args.dropout_prop))
    for i in range(int(np.ceil(args.nlayers/2))):
        model.add(layers.Dense(args.width,activation="elu"))
    model.add(layers.Dense(2))
    model.add(layers.Dense(2))
    model.compile(optimizer="Adam",
                  loss=euclidean_distance_loss)
    return model

#fit model and choose best weights
def load_callbacks(boot):
    if args.bootstrap in ['True','true','TRUE','t','T'] or args.jacknife in ['True','true','TRUE','t','T']:
        checkpointer=keras.callbacks.ModelCheckpoint(
                      filepath=args.out+"_boot"+str(boot)+"_weights.hdf5",
                      verbose=1,
                      save_best_only=True,
                      save_weights_only=True,
                      monitor="val_loss",
                      period=1)
    else:
        checkpointer=keras.callbacks.ModelCheckpoint(
                      filepath=args.out+"_weights.hdf5",
                      verbose=1,
                      save_best_only=True,
                      save_weights_only=True,
                      monitor="val_loss",
                      period=1)
    earlystop=keras.callbacks.EarlyStopping(monitor="val_loss",
                                            min_delta=0,
                                            patience=args.patience)
    reducelr=keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                               factor=0.5,
                                               patience=int(args.patience/6),
                                               verbose=1,
                                               mode='auto',
                                               min_delta=0,
                                               cooldown=0,
                                               min_lr=0)
    return checkpointer,earlystop,reducelr

def train_network(model,traingen,testgen,trainlocs,testlocs):
    history = model.fit(traingen, trainlocs,
                        epochs=args.max_epochs,
                        batch_size=args.batch_size,
                        shuffle=True,
                        verbose=1,
                        validation_data=(testgen,testlocs),
                        callbacks=[checkpointer,earlystop,reducelr])
    if args.bootstrap in ['True','true','TRUE','T','t'] or args.jacknife in ['True','true','TRUE','T','t']:
        model.load_weights(args.out+"_boot"+str(boot)+"_weights.hdf5")
    else:
        model.load_weights(args.out+"_weights.hdf5")
    return history,model

#predict and plot
def predict_locs(model,predgen,sdlong,meanlong,sdlat,meanlat,testlocs,pred,samples,verbose=True):
    import keras
    if verbose==True:
        print("predicting locations...")
    for i in tqdm(range(args.n_predictions)):
        prediction=model.predict(predgen)
        prediction=np.array([[x[0]*sdlong+meanlong,x[1]*sdlat+meanlat] for x in prediction])
        if i==0:
            predictions=prediction
        else:
            predictions=np.concatenate((predictions,prediction),axis=0)
    predout=pd.DataFrame(predictions)
    s2=[samples[pred] for x in range(args.n_predictions)]
    predout['sampleID']=[x for y in s2 for x in y]
    s3=[np.repeat(x,prediction.shape[0]) for x in range(args.n_predictions)]
    predout['prediction']=[x for y in s3 for x in y]
    if args.bootstrap in ['TRUE','True','true','T','t'] or args.jacknife in ['TRUE','True','true','T','t']:
        predout.to_csv(args.out+"_boot"+str(boot)+"_predlocs.txt",index=False)
        testlocs2=np.array([[x[0]*sdlong+meanlong,x[1]*sdlat+meanlat] for x in testlocs])
    else:
        predout.to_csv(args.out+"_predlocs.txt",index=False)
        testlocs2=np.array([[x[0]*sdlong+meanlong,x[1]*sdlat+meanlat] for x in testlocs])
    #print correlation coefficient for longitude
    if args.mode=="cv":
        r2_long=np.corrcoef(prediction[:,0],testlocs2[:,0])[0][1]**2
        r2_lat=np.corrcoef(prediction[:,1],testlocs2[:,1])[0][1]**2
        mean_dist=np.mean([spatial.distance.euclidean(prediction[x,:],testlocs2[x,:]) for x in range(len(prediction))])
        median_dist=np.median([spatial.distance.euclidean(prediction[x,:],testlocs2[x,:]) for x in range(len(prediction))])
        dists=[spatial.distance.euclidean(prediction[x,:],testlocs2[x,:]) for x in range(len(prediction))]
        if verbose==True:
            print("R2(longitude)="+str(r2_long)+"\nR2(latitude)="+str(r2_lat)+"\n"
                   +"mean error "+str(mean_dist)+"\n"
                   +"median error "+str(median_dist)+"\n")
    elif args.mode=="predict":
        p2=model.predict(testgen)
        p2=np.array([[x[0]*sdlong+meanlong,x[1]*sdlat+meanlat] for x in p2])
        r2_long=np.corrcoef(p2[:,0],testlocs2[:,0])[0][1]**2
        r2_lat=np.corrcoef(p2[:,1],testlocs2[:,1])[0][1]**2
        mean_dist=np.mean([spatial.distance.euclidean(p2[x,:],testlocs2[x,:]) for x in range(len(p2))])
        median_dist=np.median([spatial.distance.euclidean(p2[x,:],testlocs2[x,:]) for x in range(len(p2))])
        dists=[spatial.distance.euclidean(p2[x,:],testlocs2[x,:]) for x in range(len(p2))]
        if verbose==True:
            print("R2(longitude)="+str(r2_long)+"\nR2(latitude)="+str(r2_lat)+"\n"
                   +"mean error "+str(mean_dist)+"\n"
                   +"median error "+str(median_dist)+"\n")
    hist=pd.DataFrame(history.history)
    hist.to_csv(args.out+"_history.txt",sep="\t",index=False) #TODO: add if/else for bootstraps?
    #keras.backend.clear_session()
    return(dists)

def plot_history(history,dists):
    if args.plot_history:
        plt.switch_backend('agg')
        fig = plt.figure(figsize=(4,1.5),dpi=200)
        plt.rcParams.update({'font.size': 7})
        ax1=fig.add_axes([0,0,0.4,1])
        ax1.plot(history.history['val_loss'][3:],"-",color="black",lw=0.5)
        ax1.set_xlabel("Validation Loss")
        #ax1.set_yscale("log")
        #
        ax2=fig.add_axes([0.55,0,0.4,1])
        ax2.plot(history.history['loss'][3:],"-",color="black",lw=0.5)
        ax2.set_xlabel("Training Loss")
        #ax2.set_yscale("log")
        #
        fig.savefig(args.out+"_fitplot.pdf",bbox_inches='tight')
        #sys.tracebacklimit = 0 #gp.plot throws an error when printing to stdout from command line
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

#######################################################################
dropout_prop=args.dropout_prop
if args.bootstrap in ['False','FALSE','F','false','f'] and args.jacknife in ['False','FALSE','F','false','f']:
    boot=None
    genotypes,samples=load_genotypes()
    sample_data,locs=sort_samples(samples)
    meanlong,sdlong,meanlat,sdlat,locs=normalize_locs(locs)
    ac=filter_snps(genotypes)
    checkpointer,earlystop,reducelr=load_callbacks("FULL")
    train,test,traingen,testgen,trainlocs,testlocs,pred,predgen=split_train_test(ac,locs)
    model=load_network(traingen,args.dropout_prop)
    start=time.time()
    history,model=train_network(model,traingen,testgen,trainlocs,testlocs)
    dists=predict_locs(model,predgen,sdlong,meanlong,sdlat,meanlat,testlocs,pred,samples)
    plot_history(history,dists)
    if args.keep_weights in ['False','F','FALSE','f','false']:
        subprocess.run("rm "+args.out+"_weights.hdf5",shell=True)
    end=time.time()
    elapsed=end-start
    print("run time "+str(elapsed/60)+" minutes")
elif args.bootstrap in ['True','TRUE','T','true','t'] and args.jacknife in ['False','FALSE','F','false','f']:
    boot="FULL"
    genotypes,samples=load_genotypes()
    sample_data,locs=sort_samples(samples)
    meanlong,sdlong,meanlat,sdlat,locs=normalize_locs(locs)
    ac=filter_snps(genotypes)
    checkpointer,earlystop,reducelr=load_callbacks("FULL")
    train,test,traingen,testgen,trainlocs,testlocs,pred,predgen=split_train_test(ac,locs)
    model=load_network(traingen,args.dropout_prop)
    start=time.time()
    history,model=train_network(model,traingen,testgen,trainlocs,testlocs)
    dists=predict_locs(model,predgen,sdlong,meanlong,sdlat,meanlat,testlocs,pred,samples)
    plot_history(history,dists)
    if args.keep_weights in ['False','F','FALSE','f','false']:
        subprocess.run("rm "+args.out+"_bootFULL_weights.hdf5",shell=True)
    end=time.time()
    elapsed=end-start
    print("run time "+str(elapsed/60)+" minutes")
    for boot in range(args.nboots):
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
        history,model=train_network(model,traingen2,testgen2,trainlocs,testlocs)
        dists=predict_locs(model,predgen2,sdlong,meanlong,sdlat,meanlat,testlocs,pred,samples)
        plot_history(history,dists)
        if args.keep_weights in ['False','F','FALSE','f','false']:
            subprocess.run("rm "+args.out+"_boot"+str(boot)+"_weights.hdf5",shell=True)
        end=time.time()
        elapsed=end-start
        print("run time "+str(elapsed/60)+" minutes\n\n")
elif args.jacknife in ['True','TRUE','T','true','t']:
    boot="FULL"
    genotypes,samples=load_genotypes()
    sample_data,locs=sort_samples(samples)
    meanlong,sdlong,meanlat,sdlat,locs=normalize_locs(locs)
    ac=filter_snps(genotypes)
    checkpointer,earlystop,reducelr=load_callbacks(boot)
    train,test,traingen,testgen,trainlocs,testlocs,pred,predgen=split_train_test(ac,locs)
    model=load_network(traingen,args.dropout_prop)
    start=time.time()
    history,model=train_network(model,traingen,testgen,trainlocs,testlocs)
    dists=predict_locs(model,predgen,sdlong,meanlong,sdlat,meanlat,testlocs,pred,samples)
    plot_history(history,dists)
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
            pg[:,i]=af[i]
        dists=predict_locs(model,pg,sdlong,meanlong,sdlat,meanlat,testlocs,pred,samples,verbose=False)
    if args.keep_weights in ['False','F','FALSE','f','false']:
        subprocess.run("rm "+args.out+"_bootFULL_weights.hdf5",shell=True)

#
# #debugging params
# args=argparse.Namespace(vcf="/Users/cj/locator/data/ag1000g/ag1000g2L_1e6_to_2.5e6.vcf.gz",
#                         sample_data="/Users/cj/locator/data/ag1000g/anopheles_samples_sp.txt",
#                         train_split=0.8,
#                         zarr=None,
#                         boot=False,
#                         nboots=100,
#                         nlayers=10,
#                         jacknife="True",
#                         width=256,
#                         batch_size=128,
#                         max_epochs=5000,
#                         patience=20,
#                         impute_missing=True,
#                         max_SNPs=1000,
#                         min_mac=2,
#                         out="anopheles_2L_1e6-2.5e6",
#                         model="dense",
#                         outdir="/Users/cj/locator/out/",
#                         mode="cv",
#                         plot_history='True',
#                         locality_split=True,
#                         dropout_prop=0.5,
#                         gpu_number="0",
#                         n_predictions=1)
