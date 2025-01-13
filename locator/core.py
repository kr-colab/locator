"""Core functionality for locator"""

import numpy as np
import pandas as pd
import allel
import zarr
import sys
from tensorflow import keras
import matplotlib.pyplot as plt
import gnuplotlib as gp

from .models import create_network
from .utils import normalize_locs, filter_snps


class Locator:
    """Main class for locator functionality"""

    def __init__(self, config=None):
        """Initialize Locator with configuration"""
        self.config = config or {}
        self.model = None
        self.history = None
        self.meanlong = None
        self.sdlong = None
        self.meanlat = None
        self.sdlat = None

    def _load_from_zarr(self, zarr_path):
        """Load genotypes from zarr file"""
        print("reading zarr")
        callset = zarr.open_group(zarr_path, mode="r")
        gt = callset["calldata/GT"]
        genotypes = allel.GenotypeArray(gt[:])
        samples = callset["samples"][:]
        return genotypes, samples

    def _load_from_vcf(self, vcf_path):
        """Load genotypes from VCF file"""
        print("reading VCF")
        vcf = allel.read_vcf(vcf_path)
        if vcf is None:
            raise ValueError(f"Could not read VCF file: {vcf_path}")
        genotypes = allel.GenotypeArray(vcf["calldata/GT"])
        samples = vcf["samples"]
        return genotypes, samples

    def _load_from_matrix(self, matrix_path):
        """Load genotypes from matrix file"""
        gmat = pd.read_csv(matrix_path, sep="\t")
        samples = np.array(gmat["sampleID"])
        gmat = gmat.drop(labels="sampleID", axis=1)
        gmat = np.array(gmat, dtype="int8")

        # Convert to haplotype format
        hmat = None
        for i in range(gmat.shape[0]):
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

    def load_genotypes(self, vcf=None, zarr=None, matrix=None):
        """Load genotype data from various sources"""
        if zarr is not None:
            return self._load_from_zarr(zarr)
        elif vcf is not None:
            return self._load_from_vcf(vcf)
        elif matrix is not None:
            return self._load_from_matrix(matrix)
        else:
            raise ValueError("No input specified. Please provide vcf, zarr, or matrix")

    def _split_train_test(self, genotypes, locations):
        """Split data into training and test sets"""
        train = np.random.choice(
            range(len(locations)),
            size=int(self.config["train_split"] * len(locations)),
            replace=False,
        )
        test = np.array([x for x in range(len(locations)) if x not in train])
        pred = np.array([x for x in range(len(locations))])

        traingen = np.transpose(genotypes[:, train])
        testgen = np.transpose(genotypes[:, test])
        trainlocs = locations[train]
        testlocs = locations[test]
        predgen = np.transpose(genotypes[:, pred])

        return train, test, traingen, testgen, trainlocs, testlocs, pred, predgen

    def _create_callbacks(self, boot=0):
        """Create Keras callbacks for training"""
        filepath = (
            f"{self.config['out']}_boot{boot}_weights.h5"
            if self.config.get("bootstrap", False)
            else f"{self.config['out']}_weights.h5"
        )

        checkpointer = keras.callbacks.ModelCheckpoint(
            filepath=filepath,
            verbose=self.config.get("keras_verbose", 1),
            save_best_only=True,
            save_weights_only=True,
            monitor="val_loss",
            save_freq="epoch",
        )

        earlystop = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=self.config.get("patience", 100),
        )

        reducelr = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=self.config.get("patience", 100) // 6,
            verbose=self.config.get("keras_verbose", 1),
            mode="auto",
            min_delta=0,
            cooldown=0,
            min_lr=0,
        )

        return [checkpointer, earlystop, reducelr]

    def train(self, genotypes, samples, boot=0):
        """Train the model"""
        # Load and process sample data
        sample_data = pd.read_csv(self.config["sample_data"], sep="\t")
        locations = np.array(sample_data[["x", "y"]])

        # Normalize locations
        self.meanlong, self.sdlong, self.meanlat, self.sdlat, normalized_locs = (
            normalize_locs(locations)
        )

        # Filter SNPs
        filtered_genotypes = filter_snps(
            genotypes,
            min_mac=self.config.get("min_mac", 2),
            max_snps=self.config.get("max_SNPs"),
            impute=self.config.get("impute_missing", False),
        )

        # Split data
        train, test, traingen, testgen, trainlocs, testlocs, pred, predgen = (
            self._split_train_test(filtered_genotypes, normalized_locs)
        )

        # Create and train model
        self.model = create_network(
            input_shape=traingen.shape[1],
            width=self.config.get("width", 256),
            n_layers=self.config.get("nlayers", 8),
            dropout_prop=self.config.get("dropout_prop", 0.25),
        )

        callbacks = self._create_callbacks(boot)

        self.history = self.model.fit(
            traingen,
            trainlocs,
            epochs=self.config.get("max_epochs", 5000),
            batch_size=self.config.get("batch_size", 32),
            shuffle=True,
            verbose=self.config.get("keras_verbose", 1),
            validation_data=(testgen, testlocs),
            callbacks=callbacks,
        )

        # Save training history
        hist_df = pd.DataFrame(self.history.history)
        hist_df.to_csv(f"{self.config['out']}_history.txt", sep="\t", index=False)

        return self.history

    def predict(self, genotypes, boot=0):
        """Predict locations"""
        if self.model is None:
            raise ValueError("Model must be trained before prediction")

        predictions = self.model.predict(genotypes)

        # Denormalize predictions
        predictions = np.array(
            [
                [x[0] * self.sdlong + self.meanlong, x[1] * self.sdlat + self.meanlat]
                for x in predictions
            ]
        )

        # Save predictions
        pred_df = pd.DataFrame(predictions, columns=["x", "y"])
        outfile = (
            f"{self.config['out']}_boot{boot}_predlocs.txt"
            if self.config.get("bootstrap", False)
            else f"{self.config['out']}_predlocs.txt"
        )
        pred_df.to_csv(outfile, index=False)

        return predictions

    def sort_samples(self, samples, genotypes):
        """Sort and validate sample data"""
        sample_data = pd.read_csv(self.config["sample_data"], sep="\t")
        sample_data["sampleID2"] = sample_data["sampleID"]
        sample_data.set_index("sampleID", inplace=True)
        samples = samples.astype("str")
        sample_data = sample_data.reindex(np.array(samples))

        # Update to use .iloc for pandas 2.0+ compatibility
        if not all(
            [
                sample_data["sampleID2"].iloc[x] == samples[x]
                for x in range(len(samples))
            ]
        ):
            print("sample ordering failed! Check that sample IDs match the VCF.")
            sys.exit()

        locs = np.array(sample_data[["x", "y"]])
        print("loaded " + str(np.shape(genotypes)) + " genotypes\n\n")
        return sample_data, locs

    def plot_history(self, history, dists, gnuplot):
        """Plot training history and prediction error"""
        if self.config.get("plot_history", False):
            plt.switch_backend("agg")
            fig = plt.figure(figsize=(4, 1.5), dpi=200)
            plt.rcParams.update({"font.size": 7})
            ax1 = fig.add_axes([0, 0, 0.4, 1])
            ax1.plot(history.history["val_loss"][3:], "-", color="black", lw=0.5)
            ax1.set_xlabel("Validation Loss")
            ax2 = fig.add_axes([0.55, 0, 0.4, 1])
            ax2.plot(history.history["loss"][3:], "-", color="black", lw=0.5)
            ax2.set_xlabel("Training Loss")
            fig.savefig(self.config["out"] + "_fitplot.pdf", bbox_inches="tight")
            if gnuplot:
                gp.plot(
                    np.array(history.history["val_loss"][3:]),
                    unset="grid",
                    terminal="dumb 60 20",
                    title="Validation Loss by Epoch",
                )
                gp.plot(
                    (
                        np.array(dists),
                        dict(histogram="freq", binwidth=np.std(dists) / 5),
                    ),
                    unset="grid",
                    terminal="dumb 60 20",
                    title="Test Error",
                )

    def run_windows(
        self, genotypes, samples, window_start=0, window_size=5e5, window_stop=None
    ):
        """Run analysis in windows across the genome"""
        # Get positions from zarr
        positions = zarr.open_group(self.config["zarr"])["variants/POS"][:]

        if window_stop is None:
            window_stop = max(positions)

        windows = range(window_start, window_stop, int(window_size))

        for start in windows:
            stop = start + int(window_size)
            in_window = (positions >= start) & (positions < stop)

            if sum(in_window) > 0:
                window_genos = genotypes[in_window, :, :]
                self.train(window_genos, samples)
                self.predict(window_genos)

    def run_jacknife(self, genotypes, samples, prop=0.05):
        """Run jacknife analysis by dropping SNPs"""
        n_snps = genotypes.shape[0]
        n_drop = int(n_snps * prop)

        # Get base predictions
        base_preds = self.predict(genotypes)

        # Run jacknife replicates
        jack_preds = []
        for i in range(self.config.get("nboots", 50)):
            drop_idx = np.random.choice(n_snps, n_drop, replace=False)
            keep_idx = np.array([x for x in range(n_snps) if x not in drop_idx])
            jack_genos = genotypes[keep_idx, :, :]

            self.train(jack_genos, samples)
            preds = self.predict(jack_genos)
            jack_preds.append(preds)

        # Calculate uncertainty
        jack_preds = np.array(jack_preds)
        std_x = np.std(jack_preds[:, :, 0], axis=0)
        std_y = np.std(jack_preds[:, :, 1], axis=0)

        # Save results
        results = pd.DataFrame(
            {
                "x": base_preds[:, 0],
                "y": base_preds[:, 1],
                "x_std": std_x,
                "y_std": std_y,
            }
        )
        results.to_csv(f"{self.config['out']}_jacknife.txt", index=False)
