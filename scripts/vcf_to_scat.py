#vcf to scat genotype format conversion
import allel, numpy as np, pandas as pd

infile="/Users/cj/locator/data/sigma_0.45_rand500.vcf"

vcf=allel.read_vcf(infile)
genotypes=vcf['calldata/GT']
