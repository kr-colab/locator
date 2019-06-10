from slimtools import *
import pandas as pd
os.chdir("/data0/cbattey2/locator/data")

ts=pyslim.load("sigma_0.6254100218551681_.trees_2903424_1118000")
ts=sample_treeseq(infile=ts,
                  nSamples=500,
                  sampling="random",
                  recapitate=False,
                  recombination_rate=1e-9,
                  write_to_file=False,
                  outfile="",
                  sampling_locs=[],
                  plot=False,
                  seed=12345)
ts=ts.simplify()
locs=get_ms_outs(ts)[2]
locs=pd.DataFrame(locs)
locs['sampleID']=["msp_"+str(i) for i in range(len(locs))]
locs=locs.rename(index=str, columns={0: "longitude", 1: "latitude"})
locs.to_csv("sigma_0.65_rand500_samples.txt",sep="\t",index=False)
with open("sigma_0.65_rand500.vcf","w") as outfile:
    ts.write_vcf(outfile,ploidy=2)
