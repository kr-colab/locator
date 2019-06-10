#estimating sample locations from genotype matrices
from slimtools import *
import tensorflow, keras
os.environ["CUDA_VISIBLE_DEVICES"]="0"

homedir="/home/cbattey2/locator/"
os.chdir(homedir)

#sample and mutate treeseq
infile="/home/cbattey2/spaceness/sims/slimout/spatial/W50_run3/sigma_0.25409720954606546_.trees_6081404"
#infile="/home/cbattey2/spaceness/sims/slimout/spatial/W50_run3/sigma_1.0034461761403148_.trees_9532665"
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
gentimes=np.loadtxt("/home/cbattey2/spaceness/W50sp_gentimes.txt")
gentime=[x[0] for x in gentimes if np.round(x[1],5)==np.round(label,5)]
ts=msp.mutate(ts,1e-9/gentime[0])
haps,pos,locs=get_ms_outs(ts)
haplocs=np.repeat(locs,repeats=2,axis=0)

#0-1 norm
#haplocs=np.array([[x[0]/50,x[1]/50] for x in haplocs])
#locs=np.array([[x[0]/50,x[1]/50] for x in locs])
#Z-norm (then reverse transform the predictions? with these means and sd's?)
# haplocs=np.array([[(x[0]-mean(haplocs[:,0]))/np.std(haplocs[:,0]),
#                    (x[1]-mean(haplocs[:,1]))/np.std(haplocs[:,1])] for x in haplocs]) #normalize 0-1
locs=np.array([[(x[0]-np.mean(locs[:,0]))/np.std(locs[:,0]),
                (x[1]-np.mean(locs[:,1]))/np.std(locs[:,1])] for x in locs])

#split training and testing sets (haplotype version)
# train=np.random.choice(range(len(haplocs)),900,replace=False)
# test=[x for x in range(len(haplocs)) if not x in train]
# trainhaps=np.transpose(haps[:,train])
# trainlocs=haplocs[train]
# testhaps=np.transpose(haps[:,test])
# testlocs=haplocs[test]
# np.max(testlocs[:,0])
# np.min(testlocs[:,0])
# train_x=trainhaps.reshape(trainhaps.shape+(1,))
# test_x=testhaps.reshape(testhaps.shape+(1,))

#genotype version
ac=allel.HaplotypeArray(haps).to_genotypes(ploidy=2).to_allele_counts()[:,:,1]
train=np.random.choice(range(len(locs)),950,replace=False)
test=[x for x in range(len(locs)) if not x in train]
traingen=np.transpose(ac[:,train])
trainlocs=locs[train]
testgen=np.transpose(ac[:,test])
testlocs=locs[test]
train_x=traingen.reshape(traingen.shape+(1,))
test_x=testgen.reshape(testgen.shape+(1,))
print(np.shape(ac))

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
                                filepath='locator2d_weights.hdf5',
                                verbose=0,
                                save_best_only=True,
                                monitor="val_loss",
                                period=10)
earlystop=keras.callbacks.EarlyStopping(monitor="val_loss",
                                        min_delta=0,
                                        patience=200)
history = model.fit(train_x, trainlocs,
                    epochs=3000,
                    batch_size=128,
                    validation_data=(test_x,testlocs),
                    callbacks=[checkpointer,earlystop])
model.load_weights("locator2d_weights.hdf5")

#predict and plot
pred=model.predict(test_x)
cor=np.corrcoef(pred[:,0],testlocs[:,0])
print(cor)
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
fig.savefig("/home/cbattey2/locator/fig/spaceness_fitplot.pdf",bbox_inches='tight')
