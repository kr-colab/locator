#!/usr/bin/env python3
# estimating sample locations from genotype matrices
import allel, re, os, matplotlib, sys, zarr, time, subprocess, copy
import numpy as np, pandas as pd, tensorflow as tf
from scipy import spatial
from tqdm import tqdm
from matplotlib import pyplot as plt
import argparse
import json
from tensorflow.keras import backend as K

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
    "--windows",
    default=False,
    action="store_true",
    help="Run windowed analysis over a single chromosome (requires zarr input).",
)
parser.add_argument("--window_start", default=0, help="default: 0")
parser.add_argument("--window_stop", default=None, help="default: max snp position")
parser.add_argument("--window_size", default=5e5, help="default: 500000")
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
                          default: 2.",
)
parser.add_argument(
    "--max_SNPs",
    default=None,
    type=int,
    help="randomly select max_SNPs variants to use in the analysis \
                    default: None.",
)
parser.add_argument(
    "--impute_missing",
    default=False,
    action="store_true",
    help="default: True (if False, all alleles at missing sites are ancestral)",
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
    help="number of units per layer in the network\
                    default:256",
)
parser.add_argument("--out", help="file name stem for output")
parser.add_argument(
    "--seed",
    default=None,
    type=int,
    help="random seed for train/test splits and SNP subsetting.",
)
parser.add_argument("--gpu_number", default=None, type=str)
parser.add_argument(
    "--plot_history",
    default=True,
    type=bool,
    help="plot training history? \
                    default: True",
)
parser.add_argument(
    "--keep_weights",
    default=False,
    action="store_true",
    help="keep model weights after training? \
                    default: False.",
)
parser.add_argument(
    "--load_params",
    default=None,
    type=str,
    help="Path to a _params.json file to load parameters from a previous run.\
                          Parameters from the json file will supersede all parameters provided \
                          via command line.",
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
args = parser.parse_args()

# set seed and gpu
if args.seed is not None:
    np.random.seed(args.seed)
if args.gpu_number is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_number

# load old run parameters
if args.load_params is not None:
    with open(args.load_params, "r") as f:
        args.__dict__ = json.load(f)
    f.close()

# store run params
with open(args.out + "_params.json", "w") as f:
    json.dump(args.__dict__, f, indent=2)
f.close()


def load_genotypes():
    if args.zarr is not None:
        print("reading zarr")
        callset = zarr.open_group(args.zarr, mode="r")
        gt = callset["calldata/GT"]
        genotypes = allel.GenotypeArray(gt[:])
        samples = callset["samples"][:]
        positions = callset["variants/POS"]
    elif args.vcf is not None:
        print("reading VCF")
        vcf = allel.read_vcf(args.vcf, log=sys.stderr)
        genotypes = allel.GenotypeArray(vcf["calldata/GT"])
        samples = vcf["samples"]
    elif args.matrix is not None:
        gmat = pd.read_csv(args.matrix, sep="\t")
        samples = np.array(gmat["sampleID"])
        gmat = gmat.drop(labels="sampleID", axis=1)
        gmat = np.array(gmat, dtype="int8")
        for i in range(
            gmat.shape[0]
        ):  # kludge to get haplotypes for reading in to allel.
            h1 = []
            h2 = []
            for j in range(gmat.shape[1]):
                count = gmat[i, j]
                if count == 0:
                    h1.append(0)
                    h2.append(0)
                elif count == 1:
                    h1.append(1)
                    h2.append(0)
                elif count == 2:
                    h1.append(1)
                    h2.append(1)
            if i == 0:
                hmat = h1
                hmat = np.vstack((hmat, h2))
            else:
                hmat = np.vstack((hmat, h1))
                hmat = np.vstack((hmat, h2))
        genotypes = allel.HaplotypeArray(np.transpose(hmat)).to_genotypes(ploidy=2)
    return genotypes, samples


def sort_samples(samples, genotypes):
    sample_data = pd.read_csv(args.sample_data, sep="\t")
    sample_data["sampleID2"] = sample_data["sampleID"]
    sample_data.set_index("sampleID", inplace=True)
    samples = samples.astype("str")
    sample_data = sample_data.reindex(np.array(samples))

    # Update to use .iloc for pandas 2.0+ compatibility
    if not all(
        [sample_data["sampleID2"].iloc[x] == samples[x] for x in range(len(samples))]
    ):
        print("sample ordering failed! Check that sample IDs match the VCF.")
        sys.exit()

    locs = np.array(sample_data[["x", "y"]])
    print("loaded " + str(np.shape(genotypes)) + " genotypes\n\n")
    return sample_data, locs


# replace missing sites with binomial(2,mean_allele_frequency)
def replace_md(genotypes):
    print("imputing missing data")
    dc = genotypes.count_alleles()[:, 1]
    ac = genotypes.to_allele_counts()[:, :, 1]
    missingness = genotypes.is_missing()
    ninds = np.array([np.sum(x) for x in ~missingness])
    af = np.array([dc[x] / (2 * ninds[x]) for x in range(len(ninds))])
    for i in tqdm(range(np.shape(ac)[0])):
        for j in range(np.shape(ac)[1]):
            if missingness[i, j]:
                ac[i, j] = np.random.binomial(2, af[i])
    return ac


def filter_snps(genotypes):
    print("filtering SNPs")
    tmp = genotypes.count_alleles()
    biallel = tmp.is_biallelic()
    genotypes = genotypes[biallel, :, :]
    if not args.min_mac == 1:
        derived_counts = genotypes.count_alleles()[:, 1]
        ac_filter = [x >= args.min_mac for x in derived_counts]
        genotypes = genotypes[ac_filter, :, :]
    if args.impute_missing:
        ac = replace_md(genotypes)
    else:
        ac = genotypes.to_allele_counts()[:, :, 1]
    if not args.max_SNPs == None:
        ac = ac[np.random.choice(range(ac.shape[0]), args.max_SNPs, replace=False), :]
    print("running on " + str(len(ac)) + " genotypes after filtering\n\n\n")
    return ac


def normalize_locs(locs):
    meanlong = np.nanmean(locs[:, 0])
    sdlong = np.nanstd(locs[:, 0])
    meanlat = np.nanmean(locs[:, 1])
    sdlat = np.nanstd(locs[:, 1])
    locs = np.array(
        [[(x[0] - meanlong) / sdlong, (x[1] - meanlat) / sdlat] for x in locs]
    )
    return meanlong, sdlong, meanlat, sdlat, locs


def split_train_test(ac, locs):
    train = np.argwhere(~np.isnan(locs[:, 0]))
    train = np.array([x[0] for x in train])
    pred = np.array([x for x in range(len(locs)) if not x in train])
    test = np.random.choice(
        train, round((1 - args.train_split) * len(train)), replace=False
    )
    train = np.array([x for x in train if x not in test])
    traingen = np.transpose(ac[:, train])
    trainlocs = locs[train]
    testgen = np.transpose(ac[:, test])
    testlocs = locs[test]
    predgen = np.transpose(ac[:, pred])
    return train, test, traingen, testgen, trainlocs, testlocs, pred, predgen


def load_network(traingen, dropout_prop):
    from tensorflow.keras import backend as K

    def euclidean_distance_loss(y_true, y_pred):
        return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.BatchNormalization(input_shape=(traingen.shape[1],)))
    for i in range(int(np.floor(args.nlayers / 2))):
        model.add(tf.keras.layers.Dense(args.width, activation="elu"))
    model.add(tf.keras.layers.Dropout(args.dropout_prop))
    for i in range(int(np.ceil(args.nlayers / 2))):
        model.add(tf.keras.layers.Dense(args.width, activation="elu"))
    model.add(tf.keras.layers.Dense(2))
    model.add(tf.keras.layers.Dense(2))
    model.compile(optimizer="Adam", loss=euclidean_distance_loss)
    return model


def load_callbacks(boot):
    if args.bootstrap or args.jacknife:
        checkpointer = tf.keras.callbacks.ModelCheckpoint(
            filepath=args.out + "_boot" + str(boot) + ".weights.h5",
            verbose=args.keras_verbose,
            save_best_only=True,
            save_weights_only=True,
            monitor="val_loss",
            save_freq="epoch",
        )
    else:
        checkpointer = tf.keras.callbacks.ModelCheckpoint(
            filepath=args.out + ".weights.h5",
            verbose=args.keras_verbose,
            save_best_only=True,
            save_weights_only=True,
            monitor="val_loss",
            save_freq="epoch",
        )
    earlystop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=0, patience=args.patience
    )
    reducelr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=int(args.patience / 6),
        verbose=args.keras_verbose,
        mode="auto",
        min_delta=0,
        cooldown=0,
        min_lr=0,
    )
    return checkpointer, earlystop, reducelr


def train_network(model, traingen, testgen, trainlocs, testlocs, callbacks, boot=0):
    start = time.time()
    history = model.fit(
        traingen,
        trainlocs,
        epochs=args.max_epochs,
        batch_size=args.batch_size,
        shuffle=True,
        verbose=args.keras_verbose,
        validation_data=(testgen, testlocs),
        callbacks=callbacks,
    )

    # Load best weights and cleanup
    if args.bootstrap or args.jacknife:
        model.load_weights(args.out + "_boot" + str(boot) + ".weights.h5")
        if not args.keep_weights:
            subprocess.run(
                "rm " + args.out + "_boot" + str(boot) + ".weights.h5", shell=True
            )
    else:
        model.load_weights(args.out + ".weights.h5")
        if not args.keep_weights:
            subprocess.run("rm " + args.out + ".weights.h5", shell=True)

    end = time.time()
    elapsed = end - start
    print("run time " + str(elapsed / 60) + " minutes")

    return history, model


def predict_locs(
    model,
    predgen,
    sdlong,
    meanlong,
    sdlat,
    meanlat,
    testlocs,
    pred,
    samples,
    testgen,
    history,
    boot=0,
    verbose=True,
):
    if verbose == True:
        print("predicting locations...")
    prediction = model.predict(predgen)
    prediction = np.array(
        [[x[0] * sdlong + meanlong, x[1] * sdlat + meanlat] for x in prediction]
    )
    predout = pd.DataFrame(prediction)
    predout.columns = ["x", "y"]
    predout["sampleID"] = samples[pred]

    # Get base filename for predictions
    if args.bootstrap or args.jacknife:
        outfile = args.out + "_boot" + str(boot) + "_predlocs.txt"
    elif args.windows:
        # For windowed analysis, use the window index from args
        window_start = int(args.window_start)
        window_size = int(args.window_size)
        outfile = (
            f"{args.out}_{window_start}-{window_start + window_size - 1}_predlocs.txt"
        )
    else:
        outfile = args.out + "_predlocs.txt"

    predout.to_csv(outfile, index=False)

    testlocs2 = np.array(
        [[x[0] * sdlong + meanlong, x[1] * sdlat + meanlat] for x in testlocs]
    )

    p2 = model.predict(testgen)
    p2 = np.array([[x[0] * sdlong + meanlong, x[1] * sdlat + meanlat] for x in p2])
    r2_long = np.corrcoef(p2[:, 0], testlocs2[:, 0])[0][1] ** 2
    r2_lat = np.corrcoef(p2[:, 1], testlocs2[:, 1])[0][1] ** 2
    mean_dist = np.mean(
        [spatial.distance.euclidean(p2[x, :], testlocs2[x, :]) for x in range(len(p2))]
    )
    median_dist = np.median(
        [spatial.distance.euclidean(p2[x, :], testlocs2[x, :]) for x in range(len(p2))]
    )
    dists = [
        spatial.distance.euclidean(p2[x, :], testlocs2[x, :]) for x in range(len(p2))
    ]
    if verbose == True:
        print(
            "R2(x)="
            + str(r2_long)
            + "\nR2(y)="
            + str(r2_lat)
            + "\n"
            + "mean validation error "
            + str(mean_dist)
            + "\n"
            + "median validation error "
            + str(median_dist)
            + "\n"
        )
    hist = pd.DataFrame(history.history)
    hist.to_csv(args.out + "_history.txt", sep="\t", index=False)
    return dists


def plot_history(history, dists):
    if args.plot_history:
        plt.switch_backend("agg")
        fig = plt.figure(figsize=(4, 1.5), dpi=200)
        plt.rcParams.update({"font.size": 7})
        ax1 = fig.add_axes([0, 0, 0.4, 1])
        ax1.plot(history.history["val_loss"][3:], "-", color="black", lw=0.5)
        ax1.set_xlabel("Validation Loss")
        ax2 = fig.add_axes([0.55, 0, 0.4, 1])
        ax2.plot(history.history["loss"][3:], "-", color="black", lw=0.5)
        ax2.set_xlabel("Training Loss")
        fig.savefig(args.out + "_fitplot.pdf", bbox_inches="tight")


def main():
    global args  # Use the global args object

    # set seed and gpu
    if args.seed is not None:
        np.random.seed(args.seed)
    if args.gpu_number is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_number

    # load old run parameters
    if args.load_params is not None:
        with open(args.load_params, "r") as f:
            args.__dict__ = json.load(f)
        f.close()

    # store run params
    with open(args.out + "_params.json", "w") as f:
        json.dump(args.__dict__, f, indent=2)
    f.close()

    # Load data
    genotypes, samples = load_genotypes()
    sample_data, locs = sort_samples(samples, genotypes)
    meanlong, sdlong, meanlat, sdlat, locs = normalize_locs(locs)
    ac = filter_snps(genotypes)

    # Split train/test
    train, test, traingen, testgen, trainlocs, testlocs, pred, predgen = (
        split_train_test(ac, locs)
    )

    # Windows analysis
    if args.windows:
        callset = zarr.open_group(args.zarr, mode="r")
        gt = callset["calldata/GT"]
        samples = callset["samples"][:]
        positions = np.array(callset["variants/POS"])
        start = int(args.window_start)
        if args.window_stop == None:
            stop = np.max(positions)
        else:
            stop = int(args.window_stop)
        size = int(args.window_size)

        for i in np.arange(start, stop, size):
            print(f"\nProcessing window {i}-{i+size}")
            mask = np.logical_and(positions >= i, positions < i + size)
            a = np.min(np.argwhere(mask))
            b = np.max(np.argwhere(mask))
            print(f"SNPs {a}-{b}")

            genotypes = allel.GenotypeArray(gt[a:b, :, :])
            sample_data, locs = sort_samples(samples, genotypes)
            meanlong, sdlong, meanlat, sdlat, locs = normalize_locs(locs)
            ac = filter_snps(genotypes)

            train, test, traingen, testgen, trainlocs, testlocs, pred, predgen = (
                split_train_test(ac, locs)
            )
            model = load_network(traingen, args.dropout_prop)
            callbacks = load_callbacks()

            t1 = time.time()
            history, model = train_network(
                model, traingen, testgen, trainlocs, testlocs, callbacks
            )

            # Save predictions with window-specific filename
            window_out = f"{args.out}_{i}-{i+size-1}"
            original_out = args.out
            args.out = window_out  # Temporarily modify output path

            dists = predict_locs(
                model,
                predgen,
                sdlong,
                meanlong,
                sdlat,
                meanlat,
                testlocs,
                pred,
                samples,
                testgen,
                history,
            )

            args.out = original_out  # Restore original output path

            if args.plot_history:
                plot_history(history, dists)

            if not args.keep_weights:
                subprocess.run(f"rm {args.out}.weights.h5", shell=True)

            t2 = time.time()
            elapsed = t2 - t1
            print(f"Window run time {elapsed/60:.2f} minutes")

    else:
        if not args.bootstrap and not args.jacknife:
            # Standard single run
            model = load_network(traingen, args.dropout_prop)
            callbacks = load_callbacks()
            history, model = train_network(
                model, traingen, testgen, trainlocs, testlocs, callbacks
            )
            dists = predict_locs(
                model,
                predgen,
                sdlong,
                meanlong,
                sdlat,
                meanlat,
                testlocs,
                pred,
                samples,
                testgen,
                history,
            )
            if args.plot_history:
                plot_history(history, dists)

        elif args.bootstrap:
            # 1. Initial Full Run
            boot = "FULL"
            model = load_network(traingen, args.dropout_prop)
            callbacks = load_callbacks(boot)
            history, model = train_network(
                model, traingen, testgen, trainlocs, testlocs, callbacks, boot
            )
            dists = predict_locs(
                model,
                predgen,
                sdlong,
                meanlong,
                sdlat,
                meanlat,
                testlocs,
                pred,
                samples,
                testgen,
                history,
                boot,
            )
            if args.plot_history:
                plot_history(history, dists)

            # 2. Bootstrap Replicates
            for boot in range(args.nboots):
                # Set random seed for reproducibility
                np.random.seed(np.random.choice(range(int(1e6)), 1))
                print("starting bootstrap " + str(boot))

                # 3. Create bootstrapped datasets
                traingen2 = copy.deepcopy(
                    traingen
                )  # Deep copy to avoid modifying original
                testgen2 = copy.deepcopy(testgen)
                predgen2 = copy.deepcopy(predgen)

                # 4. Resample sites with replacement
                site_order = np.random.choice(
                    traingen2.shape[1], traingen2.shape[1], replace=True
                )
                traingen2 = traingen2[:, site_order]  # Reorder sites in training data
                testgen2 = testgen2[:, site_order]  # Apply same reordering to test data
                predgen2 = predgen2[:, site_order]  # And prediction data

                # 5. Train new model on bootstrapped data
                model = load_network(traingen2, args.dropout_prop)
                callbacks = load_callbacks(boot)
                history, model = train_network(
                    model, traingen2, testgen2, trainlocs, testlocs, callbacks, boot
                )

                # 6. Make predictions with bootstrapped model
                dists = predict_locs(
                    model,
                    predgen2,
                    sdlong,
                    meanlong,
                    sdlat,
                    meanlat,
                    testlocs,
                    pred,
                    samples,
                    testgen2,
                    history,
                    boot,
                )
                if args.plot_history:
                    plot_history(history, dists)

                # 7. Cleanup to prevent memory leaks
                K.clear_session()

        elif args.jacknife:
            # Initial full run
            boot = "FULL"
            model = load_network(traingen, args.dropout_prop)
            callbacks = load_callbacks(boot)
            start = time.time()  # Add timing
            history, model = train_network(
                model, traingen, testgen, trainlocs, testlocs, callbacks, boot
            )
            dists = predict_locs(
                model,
                predgen,
                sdlong,
                meanlong,
                sdlat,
                meanlat,
                testlocs,
                pred,
                samples,
                testgen,
                history,
                boot,
            )
            if args.plot_history:
                plot_history(history, dists)
            end = time.time()
            elapsed = end - start
            print("run time " + str(elapsed / 60) + " minutes")

            # Jacknife resampling
            print("starting jacknife resampling")
            af = []
            for i in tqdm(range(ac.shape[0])):
                af.append(sum(ac[i, :]) / (ac.shape[1] * 2))
            af = np.array(af)

            for boot in tqdm(range(args.nboots)):
                callbacks = load_callbacks(boot)
                pg = copy.deepcopy(predgen)
                sites_to_remove = np.random.choice(
                    pg.shape[1], int(pg.shape[1] * args.jacknife_prop), replace=False
                )

                for i in sites_to_remove:
                    pg[:, i] = np.random.binomial(2, af[i], pg.shape[0])

                dists = predict_locs(
                    model,
                    pg,
                    sdlong,
                    meanlong,
                    sdlat,
                    meanlat,
                    testlocs,
                    pred,
                    samples,
                    testgen,
                    history,
                    boot,
                    verbose=False,
                )

            # Cleanup weights file if not keeping
            if not args.keep_weights:
                subprocess.run("rm " + args.out + "_bootFULL.weights.h5", shell=True)

    return 0


if __name__ == "__main__":
    main()

# ag1000g.phase1.ar3.pass.2L.0-5e6.zarr
###debugging params
# args=argparse.Namespace(vcf=None,#"/Users/cj/locator/data/test_genotypes.vcf.gz",
#                         matrix=None,#"/Users/cj/locator/data/test_genotypes.vcf.gz",
#                         zarr="/Users/cj/locator/data/test_genotypes.zarr",
#                         sample_data="/Users/cj/locator/data/test_sample_data.txt",
#                         train_split=0.9,
#                         windows=True,
#                         window_start=0,
#                         window_stop=None,
#                         window_size=2e5,
#                         seed=12345,
#                         boot=False,
#                         load_params=None,
#                         nboots=100,
#                         nlayers=8,
#                         jacknife=False,
#                         width=256,
#                         batch_size=32,
#                         max_epochs=5000,
#                         bootstrap=False,
#                         patience=20,
#                         impute_missing=True,
#                         max_SNPs=None,
#                         min_mac=2,
#                         gnuplot=True,
#                         out="/Users/cj/Desktop/test",
#                         plot_history='True',
#                         dropout_prop=0.25,
#                         gpu_number="0")
