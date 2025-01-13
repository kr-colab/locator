"""Command line interface for locator"""

import argparse
import sys
import os
import json
from .core import Locator
import time


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--vcf", help="VCF with SNPs for all samples.")
    parser.add_argument("--zarr", help="zarr file of SNPs for all samples.")
    parser.add_argument(
        "--matrix",
        help="tab-delimited matrix of minor allele counts with first column named 'sampleID'.\
                                     E.g., \
                                     \
                                     sampleID\tsite1\tsite2\t...\n \
                                     msp1\t0\t1\t...\n \
                                     msp2\t2\t0\t...\n ",
    )
    parser.add_argument(
        "--sample_data",
        help="tab-delimited text file with columns\
                         'sampleID \t x \t y'.\
                          SampleIDs must exactly match those in the \
                          VCF. X and Y values for \
                          samples without known locations should \
                          be NA.",
    )
    parser.add_argument(
        "--train_split",
        default=0.9,
        type=float,
        help="0-1, proportion of samples to use for training. \
                          default: 0.9 ",
    )
    parser.add_argument(
        "--bootstrap",
        default=False,
        action="store_true",
        help="Run bootstrap replicates by retraining on bootstrapped data.",
    )
    parser.add_argument(
        "--jacknife",
        default=False,
        action="store_true",
        help="Run jacknife uncertainty estimate on a trained network. \
                    NOTE: we recommend this only as a fast heuristic -- use the bootstrap \
                    option or run windowed analyses for final results.",
    )
    parser.add_argument(
        "--jacknife_prop",
        default=0.05,
        type=float,
        help="proportion of SNPs to remove for jacknife resampling.\
                    default: 0.05",
    )
    parser.add_argument(
        "--nboots",
        default=50,
        type=int,
        help="number of bootstrap replicates to run.\
                    default: 50",
    )
    parser.add_argument("--batch_size", default=32, type=int, help="default: 32")
    parser.add_argument("--max_epochs", default=5000, type=int, help="default: 5000")
    parser.add_argument(
        "--patience",
        type=int,
        default=100,
        help="n epochs to run the optimizer after last \
                          improvement in validation loss. \
                          default: 100",
    )
    parser.add_argument(
        "--min_mac",
        default=2,
        type=int,
        help="minimum minor allele count.\
                          default: 2",
    )
    parser.add_argument(
        "--max_SNPs",
        default=None,
        type=int,
        help="randomly select max_SNPs variants to use in the analysis",
    )
    parser.add_argument(
        "--impute_missing",
        default=False,
        action="store_true",
        help="impute missing genotypes using mean allele frequency",
    )
    parser.add_argument(
        "--dropout_prop",
        default=0.25,
        type=float,
        help="proportion of weights to zero at the dropout layer. \
                          default: 0.25",
    )
    parser.add_argument(
        "--nlayers",
        default=10,
        type=int,
        help="number of layers in the network. \
                          default: 10",
    )
    parser.add_argument(
        "--width",
        default=256,
        type=int,
        help="number of units in each layer. \
                          default: 256",
    )
    parser.add_argument(
        "--out",
        default="locator_out",
        help="file name stem for output; will generate 'stem_predlocs.txt', \
                          'stem_history.txt', 'stem_weights.hdf5', and 'stem_params.json'. \
                          default: locator_out",
    )
    parser.add_argument(
        "--seed", default=None, type=int, help="random seed. default: None"
    )
    parser.add_argument(
        "--gpu_number",
        default=None,
        help="restrict to specific GPU. default: None",
    )
    parser.add_argument(
        "--plot_history",
        default=False,
        action="store_true",
        help="plot training history and prediction error",
    )
    parser.add_argument(
        "--gnuplot",
        default=False,
        action="store_true",
        help="print training history to terminal",
    )
    parser.add_argument(
        "--keras_verbose",
        default=1,
        type=int,
        help="verbose argument passed to keras in model training. \
                    0 = silent. 1 = progress bars for minibatches. 2 = show epochs. \
                    Yes, 1 is more verbose than 2. Blame keras. \
                    default: 1. ",
    )
    parser.add_argument(
        "--windows",
        default=False,
        action="store_true",
        help="Run windowed analysis over a single chromosome (requires zarr input).",
    )
    parser.add_argument("--window_start", default=0, help="default: 0")
    parser.add_argument("--window_stop", default=None, help="default: max snp position")
    parser.add_argument("--window_size", default=5e5, help="default: 500000")
    parser.add_argument("--load_params", help="Load parameters from previous run")
    parser.add_argument("--predict_from_weights", help="Load saved weights")
    parser.add_argument("--keep_weights", default=False, action="store_true")

    return parser.parse_args()


def main():
    """Main entry point for CLI"""
    args = parse_args()

    # Set GPU and seed
    if args.seed is not None:
        import numpy as np

        np.random.seed(args.seed)
    if args.gpu_number is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_number

    # Load old parameters if specified
    if args.load_params is not None:
        with open(args.predict_from_weights + "_params", "r") as f:
            args.__dict__ = json.load(f)

    # Initialize locator
    loc = Locator(vars(args))

    # Store run parameters
    if args.out is not None:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out + "_params.json", "w") as f:
            json.dump(vars(args), f, indent=2)

    # Load data
    genotypes, samples = loc.load_genotypes(
        vcf=args.vcf, zarr=args.zarr, matrix=args.matrix
    )

    # Track runtime
    start = time.time()

    # Run analysis based on mode
    if args.windows:
        # Windowed analysis logic
        pass
    elif args.jacknife:
        # Jacknife analysis logic
        pass
    elif args.bootstrap:
        for boot in range(args.nboots):
            loc.train(genotypes, samples, boot=boot)
            loc.predict(genotypes, boot=boot)
    else:
        loc.train(genotypes, samples)
        loc.predict(genotypes)

    # Clean up weights if not keeping them
    if not args.keep_weights:
        if args.bootstrap:
            for boot in range(args.nboots):
                os.remove(f"{args.out}_boot{boot}_weights.h5")
        else:
            os.remove(f"{args.out}_weights.h5")

    # Report runtime
    end = time.time()
    print(f"run time {(end-start)/60} minutes")

    return 0


if __name__ == "__main__":
    sys.exit(main())
