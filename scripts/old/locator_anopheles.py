#estimating sample locations from genotype matrices
import allel, re, os, tensorflow, keras, geopy, matplotlib, sys
import numpy as np, pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
#os.environ["CUDA_VISIBLE_DEVICES"]="1" #set to "" to run on CPU
#config = tensorflow.ConfigProto(device_count={'CPU': 60}) #specify n cpus
#sess = tensorflow.Session(config=config)

homedir="/home/cbattey2/locator/"
infile="data/ag1000g2L_1e6_to_4e6.vcf.gz"
train_split=0.9 #proportion of data to use for training

os.chdir(homedir)

#load genotype matrices from VCF
vcf=allel.read_vcf(os.path.join(homedir,infile))
genotypes=allel.GenotypeArray(vcf['calldata/GT'])
samples=vcf['samples']
sample_data=pd.read_csv("data/anopheles_samples_sp.txt",sep="\t")
sample_data['sampleID']=sample_data['ox_code']
sample_data.set_index('sampleID',inplace=True)
sample_data=sample_data.reindex(samples) #sort loc table so samples are in same order as vcf samples
if not all([sample_data['ox_code'][x]==samples[x] for x in range(len(samples))]): #check that all sample names are present
    print("sample ordering failed!")
print("loaded "+str(np.shape(genotypes))+" genotypes\n\n")
derived_counts=genotypes.count_alleles()[:,1]
af_filter=[x > 1 and x < np.shape(genotypes)[1]*2-1 for x in derived_counts] #drop singletons
genotypes=genotypes[af_filter,:,:]
ac=genotypes.to_allele_counts()[0:80000,sample_data['species']=="gambiae",1]
locs=np.array(sample_data[["longitude","latitude"]])[sample_data['species']=="gambiae"]
print("running on "+str(len(ac))+" genotypes after filtering\n\n\n")

#normalize coordinates
meanlong=np.mean(locs[:,0])
sdlong=np.std(locs[:,0])
meanlat=np.mean(locs[:,1])
sdlat=np.std(locs[:,1])
locs=np.array([[(x[0]-meanlong)/sdlong,(x[1]-meanlat)/sdlat] for x in locs])

#genotype version
train=np.random.choice(range(len(locs)),
                       round(train_split*len(locs)),
                       replace=False)
test=[x for x in range(len(locs)) if not x in train]
traingen=np.transpose(ac[:,train])
trainlocs=locs[train]
testgen=np.transpose(ac[:,test])
testlocs=locs[test]
train_x=traingen.reshape(traingen.shape+(1,))
test_x=testgen.reshape(testgen.shape+(1,))

#define a 1D CNN for regression
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
model = Sequential()
model.add(layers.Conv1D(64, 7, activation='relu',input_shape=(np.shape(train_x)[1],1)))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(2))
model.compile(optimizer="Adam",
              loss=keras.losses.mean_absolute_error,
              metrics=['mse'])

#fit model and choose best weights
checkpointer=keras.callbacks.ModelCheckpoint(
                                filepath='anopheles_weights.hdf5',
                                verbose=1,
                                save_best_only=True,
                                monitor="val_loss",
                                period=1)
earlystop=keras.callbacks.EarlyStopping(monitor="val_loss",
                                        min_delta=0,
                                        patience=200)
history = model.fit(train_x, trainlocs,
                    epochs=5000,
                    batch_size=128,
                    validation_data=(test_x,testlocs),
                    callbacks=[checkpointer,earlystop])
model.load_weights("anopheles_weights.hdf5")

#predict and plot
pred=model.predict(test_x)
pred=np.array([[x[0]*sdlong+meanlong,x[1]*sdlat+meanlat] for x in pred]) #reverse normalization
testlocs=np.array([[x[0]*sdlong+meanlong,x[1]*sdlat+meanlat] for x in testlocs])
np.savetxt(fname=os.path.join(homedir,"stats/anopheles_predlocs.txt"),X=pred)
np.savetxt(fname=os.path.join(homedir,"stats/anopheles_testlocs.txt"),X=testlocs)

#print correlation coefficient for longitude
r2_long=np.corrcoef(pred[:,0],testlocs[:,0])[0][1]**2
r2_lat=np.corrcoef(pred[:,1],testlocs[:,1])[0][1]**2
print("R2(longitude)="+str(r2_long)+"\nR2(latitude)="+str(r2_lat))

fig = plt.figure(figsize=(4,2),dpi=200)
plt.rcParams.update({'font.size': 7})
ax1=fig.add_axes([0,.59,0.25,.375])
ax1.plot(history.history['val_loss'][10:],"-",color="black",lw=0.5)
ax1.set_xlabel("Validation Loss")

ax2=fig.add_axes([0,0,0.25,.375])
ax2.plot(history.history['loss'][10:],"-",color="black",lw=0.5)
ax2.set_xlabel("Training Loss")

ax3=fig.add_axes([0.44,0.01,0.55,.94])
ax3.scatter(testlocs[:,0],testlocs[:,1],s=4,linewidth=.4,facecolors="none",edgecolors="black")
ax3.scatter(pred[:,0],pred[:,1],s=2,color="black")
for x in range(len(pred)):
    ax3.plot([pred[x,0],testlocs[x,0]],[pred[x,1],testlocs[x,1]],lw=.3,color="black")
#ax3.set_xlabel("simulated X coordinate")
#ax3.set_ylabel("predicted X coordinate")
#ax3.set_title(r"$R^2$="+str(round(cor[0][1]**2,4)))
fig.savefig("/home/cbattey2/locator/fig/anopheles_fitplot.pdf",bbox_inches='tight')
