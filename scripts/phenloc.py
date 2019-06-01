#estimating sample locations and polygenic phenotypes from genotype matrices
from slimtools import *
import tensorflow, keras
from scipy import spatial
os.environ["CUDA_VISIBLE_DEVICES"]="0"

homedir="/home/cbattey2/popvae/"
os.chdir(homedir)

#sample and mutate treeseq
infile="/home/cbattey2/spaceness/sims/slimout/spatial/W50_run3/sigma_0.400670106952987_.trees_1541565"
ts=sample_treeseq(infile=infile,
               outfile="",
               nSamples=1000,
               recapitate=False,
               recombination_rate=1e-8,
               write_to_file=False,
               sampling="random",
               sampling_locs=[[12.5,12.5],[12.5,37.5],[37.5,37.5],[37.5,12.5]],
               plot=True,
               dist_scaling=2)

label=float(re.sub("sigma_|_.trees*|.trees","",os.path.basename(infile)))
gentimes=np.loadtxt("sims/W50sp_gentimes.txt")
gentime=[x[0] for x in gentimes if np.round(x[1],5)==np.round(label,5)]
ts=msp.mutate(ts,1e-9/gentime[0])
haps,pos,locs=get_ms_outs(ts)
haplocs=np.repeat(locs,repeats=2,axis=0)
print(np.shape(haps))
haplocs=np.array([[x[0]/50,x[1]/50] for x in haplocs])

#generate polygenic phentoypes from 1000 sites, exponentally distributed effect sizes
ac=allel.HaplotypeArray(haps).to_genotypes(ploidy=2).to_allele_counts()[:,:,1]
af=allel.HaplotypeArray(haps).to_genotypes(ploidy=2).count_alleles()[:,1]/np.shape(haps)[1]
sites=np.random.choice(range(len(pos)),10000)
effects=[np.random.gamma(scale=.1,shape=.1) for x in sites]
phensnps=ac[sites,:]
phenotypes=[]
for i in range(np.shape(phensnps)[1]):
    h=phensnps[:,i]
    phenotypes.append(np.sum(h*effects))
phenotypes=[(x-np.mean(phenotypes))/np.std(phenotypes) for x in phenotypes]
phenotypes=np.array(phenotypes)
print(np.corrcoef(phenotypes,locs[:,0])[0][1]**2)
#plt.scatter(phenotypes,locs[:,0])

#split training and testing sets
train=np.random.choice(range(len(locs)),900,replace=False)
test=[x for x in range(len(locs)) if not x in train]
trainhaps=np.transpose(ac[:,train])
trainlocs=haplocs[train]
testhaps=np.transpose(ac[:,test])
testlocs=haplocs[test]

#1D CNN
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
model = Sequential()
model.add(layers.Conv1D(64, 7, activation='relu',input_shape=(np.shape(trainhaps)[1],1)))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))
#model.summary()

#dense network (see if this can be fixed...)
# model=Sequential()
# model.add(layers.Dense(256,
#                        input_shape=(np.shape(trainhaps)[1],1)))
# model.add(layers.Dense(128))
# model.add(layers.Dense(64))
# model.add(layers.GlobalMaxPooling1D())
# model.add(layers.Dense(1))
# model.summary()
model.compile(optimizer="Adam",
              loss=keras.losses.mean_squared_error,
              metrics=['mae'])
trainhaps2=trainhaps.reshape(trainhaps.shape+(1,))
testhaps2=testhaps.reshape(testhaps.shape+(1,))
np.shape(trainhaps2)
# history = model.fit(trainhaps2, trainlocs[:,0],
#                     epochs=1000,
#                     batch_size=128,
#                     validation_data=(testhaps2,testlocs[:,0]))
checkpointer=keras.callbacks.ModelCheckpoint(filepath='weights.hdf5',
                                verbose=0,
                                save_best_only=True,
                                monitor="val_loss",
                                period=10)
history = model.fit(trainhaps2, phenotypes[train],
                    epochs=1000,
                    batch_size=128,
                    validation_data=(testhaps2,phenotypes[test]),
                    callbacks=[checkpointer])
model.load_weights("weights.hdf5")
pred=model.predict(testhaps2)
print(np.shape(pred))
cor=np.corrcoef(pred[:,0],phenotypes[test])
print(cor)
print(np.corrcoef(model.predict(trainhaps2)[:,0],phenotypes[train]))
fig = plt.figure(figsize=(4,2),dpi=200)
plt.rcParams.update({'font.size': 7})
ax1=fig.add_axes([0,.59,0.25,.375])
ax1.plot(history.history['val_loss'][20:],"-",color="black",lw=0.5)
ax1.set_xlabel("Validation Loss")

ax2=fig.add_axes([0,0,0.25,.375])
ax2.plot(history.history['val_mean_absolute_error'][20:],"-",color="black",lw=0.5)
ax2.set_xlabel("Validation MAE")

ax3=fig.add_axes([0.44,0.01,0.55,.94])
ax3.plot([np.min(phenotypes[test]),np.max(phenotypes[test])],
         [np.min(phenotypes[test]),np.max(phenotypes[test])],c="grey",lw=0.5)
ax3.scatter(phenotypes[test],pred[:,0],s=2,color="black")
ax3.set_xlabel("simulated phenotype")
ax3.set_ylabel("predicted phenotype")
ax3.set_title(r"$R^2$="+str(round(cor[0][1]**2,4)))
fig.savefig("/home/cbattey2/popvae/fig/phenloc_plots.pdf",bbox_inches='tight')
