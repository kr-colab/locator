#ukbb data munging for locator
import allel, sys, os, re
import numpy as np, pandas as pd
from tqdm import tqdm

os.chdir("/Users/cj/locator/data")
samples=pd.read_csv("ukb26103.csv")
fam=pd.read_csv("ukb_cal_chr22_v2.fam",sep=" ",header=None)

samples=np.array(samples[['eid','130-0.0','129-0.0']])
samples=samples[~np.isnan(samples[:,2]),:]
goodlocs=[x>0 for x in samples[:,1]]
samples=samples[goodlocs]
samples=np.array([[int(x[0]),int(x[1]),int(x[2])] for x in samples])
intersect=np.intersect1d(np.array(samples[:,0]),np.array(fam[0]))

for i in [2000,10000,50000,100000]:
    samples_to_run=np.random.choice(intersect,i,replace=False)
    sout=pd.DataFrame([x for x in samples if x[0] in samples_to_run])
    sout=sout.rename({0:"sampleID",1:"longitude",2:"latitude"},axis=1)
    sout['sampleID']=[str(int(x))+"_"+str(int(x)) for x in sout['sampleID']]
    sout.to_csv("/Users/cj/locator/data/ukb_"+str(i)+"inds_sample_data.txt",sep="\t",index=False)
    samples_to_run=[str(int(x))+"_"+str(int(x)) for x in samples_to_run]
    pd.DataFrame(samples_to_run).to_csv("/Users/cj/locator/data/ukb_"+str(i)+"inds.txt",
                                        sep=" ",index=False,header=False)
