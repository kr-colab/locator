#ukbb data munging for locator
import allel, sys, os, re
import numpy as np, pandas as pd
from tqdm import tqdm

os.chdir("/Users/cj/locator/data")
samples=pd.read_csv("ukb26103.csv")
fam=pd.read_csv("ukb_cal_chr22_v2.fam",sep=" ",header=None)
intersect=np.intersect1d(np.array(samples['eid']),np.array(fam[0]))

samples=np.array(samples[['eid','130-0.0','129-0.0']])
samples=samples[~np.isnan(samples[:,2]),:]
samples=np.array([[int(x[0]),int(x[1]),int(x[2])] for x in samples])

samples_to_run=np.random.choice(intersect,100000,replace=False)
sout=pd.DataFrame([x for x in samples if x[0] in samples_to_run])
sout=sout.rename({0:"sampleID",1:"longitude",2:"latitude"},axis=1)
sout.to_csv("/Users/cj/locator/data/ukb_testkinds_sample_data.txt",sep="\t",index=False)
samples_to_run=[str(int(x))+"_"+str(int(x)) for x in samples_to_run]
pd.DataFrame(samples_to_run).to_csv("/Users/cj/locator/data/ukb_testkinds.txt",
                                    sep=" ",index=False,header=False)
