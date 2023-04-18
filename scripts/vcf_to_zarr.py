import allel,argparse
parser=argparse.ArgumentParser()
parser.add_argument("--vcf",help="path to VCF (or .vcf.gz)")
parser.add_argument("--zarr",help="path for zarr output")
args=parser.parse_args()

allel.vcf_to_zarr(args.vcf,args.zarr,fields=['variants/POS','calldata/GT','samples'])
