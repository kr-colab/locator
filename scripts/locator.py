#estimating sample locations from genotype matrices
import allel, re, os, keras, geopy, matplotlib, sys
import numpy as np, pandas as pd, tensorflow as tf
from tqdm import tqdm
from matplotlib import pyplot as plt
import argparse
#os.environ["CUDA_VISIBLE_DEVICES"]="0" #set to "" to run on CPU
#config = tensorflow.ConfigProto(device_count={'CPU': 60})
#sess = tensorflow.Session(config=config)
# config = tf.ConfigProto()
# config.intra_op_parallelism_threads = 44
# config.inter_op_parallelism_threads = 44
# tf.Session(config=config)

parser=argparse.ArgumentParser()
parser.add_argument("--vcf",help="VCF with SNPs for all samples.")
parser.add_argument("--sample_data",help="tab-delimited text file with columns\
                                         'sampleID \t longitude \t latitude'.\
                                         SampleIDs must exactly match those in the \
                                         training VCF. Longitude and latitude for \
                                         samples without known geographic origin should \
                                         be NA. By default, locations will be predicted \
                                         for all samples without locations. If the \
                                         train_split parameter is provided, locations \
                                         will be predicted for randomly selected \
                                         individuals.")
parser.add_argument("--train_split",default=None,type=float)
parser.add_argument("--batch_size",default=128,type=int)
parser.add_argument("--max_epochs",default=5000,type=int)
parser.add_argument("--max_SNPs",default=None,type=int)
parser.add_argument("--min_mac",default=2,type=int)
parser.add_argument("--patience",type=int,default=100)
parser.add_argument("--model",default="CNN")
parser.add_argument("--outname")
parser.add_argument("--outdir")
args=parser.parse_args()

#debugging params
# args=argparse.Namespace(vcf="/Users/cj/locator/data/ag1000g2L_1e6_to_2.5e6.vcf.gz",
#                         sample_data="/Users/cj/locator/data/anopheles_samples_sp.txt",
#                         train_split=0.9,
#                         batch_size=128,
#                         max_epochs=5000,
#                         patience=200,
#                         max_SNPs=None,
#                         min_mac=2,
#                         outname="anopheles",
#                         model="dense",
#                         outdir="/Users/cj/locator/stats/")

#load genotype matrices from VCF
vcf=allel.read_vcf(args.vcf)
genotypes=allel.GenotypeArray(vcf['calldata/GT'])
samples=vcf['samples']

#load and sort sample data to match VCF sample order
sample_data=pd.read_csv(args.sample_data,sep="\t")
sample_data['sampleID2']=sample_data['sampleID']
sample_data.set_index('sampleID',inplace=True)
sample_data=sample_data.reindex(samples) #sort loc table so samples are in same order as vcf samples
if not all([sample_data['sampleID2'][x]==samples[x] for x in range(len(samples))]): #check that all sample names are present
    print("sample ordering failed! Check that sample IDs match the VCF.")
print("loaded "+str(np.shape(genotypes))+" genotypes\n\n")

#drop low-frequency sites
derived_counts=genotypes.count_alleles()[:,1]
af_filter=[x >= args.min_mac for x in derived_counts] #drop low-frequency alleles
genotypes=genotypes[af_filter,:,:]

if not args.max_SNPs==None:
    ac=genotypes.to_allele_counts()[0:args.max_SNPs,:,1]
else:
    ac=genotypes.to_allele_counts()[:,:,1]
locs=np.array(sample_data[["longitude","latitude"]])
print("running on "+str(len(ac))+" genotypes after filtering\n\n\n")

#normalize coordinates
meanlong=np.nanmean(locs[:,0])
sdlong=np.nanstd(locs[:,0])
meanlat=np.nanmean(locs[:,1])
sdlat=np.nanstd(locs[:,1])
locs=np.array([[(x[0]-meanlong)/sdlong,(x[1]-meanlat)/sdlat] for x in locs])

#split training, testing, and prediction sets
if any(np.isnan(locs[:,0])):
    train=np.argwhere(~np.isnan(locs[:,0]))
    train=[x[0] for x in train]
    pred=[x for x in range(len(locs)) if not x in train]
    test=np.random.choice(train,len(train)-round(args.train_split*len(train)))
    train=[x for x in train if x not in test]
    traingen=np.transpose(ac[:,train])
    trainlocs=locs[train]
    testgen=np.transpose(ac[:,test])
    testlocs=locs[test]
    predgen=np.transpose(ac[:,pred])
else:
    train=np.random.choice(range(len(locs)),
                           round(args.train_split*len(locs)),
                           replace=False)
    test=[x for x in range(len(locs)) if not x in train]
    traingen=np.transpose(ac[:,train])
    trainlocs=locs[train]
    testgen=np.transpose(ac[:,test])
    testlocs=locs[test]

#define a 1D CNN for regression
from keras.models import Sequential
from keras import layers
if args.model=="CNN":
    train_x=traingen.reshape(traingen.shape+(1,))
    test_x=testgen.reshape(testgen.shape+(1,))
    pred_x=predgen.reshape(predgen.shape+(1,))
    model = Sequential()
    model.add(layers.Conv1D(256, 7, activation='relu',input_shape=(np.shape(train_x)[1],1)))
    model.add(layers.Conv1D(64, 7, activation='relu',input_shape=(np.shape(train_x)[1],1)))
    model.add(layers.Conv1D(32, 7, activation='relu'))
    model.add(layers.Dense(2))
    odel.compile(optimizer="Adam",
                  loss=keras.losses.mean_squared_error,
                  metrics=['mae'])

if args.model=="dense":
    train_x=traingen
    test_x=testgen
    pred_x=testgen
    model = Sequential()
    model.add(layers.Dense(256, activation='elu',input_shape=(np.shape(train_x)[1],)))
    model.add(layers.Dense(128,activation='elu'))
    model.add(layers.Dense(64,activation='elu'))
    model.add(layers.Dense(16,activation='elu'))
    model.add(layers.Dense(2))
    model.compile(optimizer="Adam",
                  loss=keras.losses.mean_squared_error,
                  metrics=['mae'])



#fit model and choose best weights
checkpointer=keras.callbacks.ModelCheckpoint(
                                filepath='weights.hdf5',
                                verbose=1,
                                save_best_only=True,
                                monitor="val_loss",
                                period=1)
earlystop=keras.callbacks.EarlyStopping(monitor="val_loss",
                                        min_delta=0,
                                        patience=args.patience)
history = model.fit(train_x, trainlocs,
                    epochs=args.max_epochs,
                    batch_size=args.batch_size,
                    validation_data=(test_x,testlocs),
                    callbacks=[checkpointer,earlystop])
model.load_weights("weights.hdf5")

#predict and plot
pred=model.predict(pred_x)
pred=np.array([[x[0]*sdlong+meanlong,x[1]*sdlat+meanlat] for x in pred]) #reverse normalization
testlocs=np.array([[x[0]*sdlong+meanlong,x[1]*sdlat+meanlat] for x in testlocs])
np.savetxt(fname=os.path.join(args.outdir,args.outname+"_predlocs.txt"),X=pred)
np.savetxt(fname=os.path.join(args.outdir,args.outname+"_testlocs.txt"),X=testlocs)

#print correlation coefficient for longitude
r2_long=np.corrcoef(pred[:,0],testlocs[:,0])[0][1]**2
r2_lat=np.corrcoef(pred[:,1],testlocs[:,1])[0][1]**2
print("R2(longitude)="+str(r2_long)+"\nR2(latitude)="+str(r2_lat))

fig = plt.figure(figsize=(4,2),dpi=200)
plt.rcParams.update({'font.size': 7})
ax1=fig.add_axes([0,.59,0.25,.375])
ax1.plot(history.history['val_loss'][5:],"-",color="black",lw=0.5)
ax1.set_xlabel("Validation Loss")
ax1.set_yscale("log")

ax2=fig.add_axes([0,0,0.25,.375])
ax2.plot(history.history['loss'][5:],"-",color="black",lw=0.5)
ax2.set_xlabel("Training Loss")
ax2.set_yscale("log")

ax3=fig.add_axes([0.44,0.01,0.55,.94])
ax3.scatter(testlocs[:,0],testlocs[:,1],s=4,linewidth=.4,facecolors="none",edgecolors="black")
ax3.scatter(pred[:,0],pred[:,1],s=2,color="black")
for x in range(len(pred)):
    ax3.plot([pred[x,0],testlocs[x,0]],[pred[x,1],testlocs[x,1]],lw=.3,color="black")
#ax3.set_xlabel("simulated X coordinate")
#ax3.set_ylabel("predicted X coordinate")
#ax3.set_title(r"$R^2$="+str(round(cor[0][1]**2,4)))
fig.savefig(os.path.join(args.outdir,args.outname+"_fitplot.pdf"),bbox_inches='tight')
