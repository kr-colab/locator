"""Core functionality for locator"""

import numpy as np
import pandas as pd
import allel
import zarr
import sys
from tensorflow import keras
import matplotlib.pyplot as plt

from .models import create_network
from .utils import normalize_locs, filter_snps


class Locator:
    """Main class for locator functionality"""

    def __init__(self, config=None):
        """Initialize Locator with configuration.

        Args:
            config: Optional dictionary of configuration parameters that will override defaults

        Default config parameters:
            train_split: Proportion of data to use for training (0.9)
            out: Output file prefix ("locator_out")
            bootstrap: Whether to run bootstrap replicates (False)
            keras_verbose: Verbosity level for Keras (1)
            patience: Epochs to wait before early stopping (100)
            batch_size: Training batch size (32)
            max_epochs: Maximum training epochs (5000)
            min_mac: Minimum minor allele count (2)
            max_SNPs: Maximum number of SNPs to use (None)
            impute_missing: Whether to impute missing genotypes (False)
            dropout_prop: Dropout proportion (0.25)
            nlayers: Number of neural network layers (8)
            width: Width of neural network layers (256)
        """
        default_config = {
            # Data splitting
            "train_split": 0.9,
            # Output
            "out": "locator_out",
            # Training parameters
            "bootstrap": False,
            "keras_verbose": 1,
            "patience": 100,
            "batch_size": 32,
            "max_epochs": 5000,
            # Data processing
            "min_mac": 2,
            "max_SNPs": None,
            "impute_missing": False,
            # Network architecture
            "dropout_prop": 0.25,
            "nlayers": 8,
            "width": 256,
        }

        self.config = {**default_config, **(config or {})}
        self.model = None
        self.history = None
        self.meanlong = None
        self.sdlong = None
        self.meanlat = None
        self.sdlat = None

    def _load_from_zarr(self, zarr_path):
        """Load genotypes from zarr file.

        Args:
            zarr_path: Path to zarr file containing genotype data

        Returns:
            tuple: (genotypes, samples) where:
                - genotypes is an allel.GenotypeArray containing genetic data
                - samples is a numpy array of sample IDs
        """
        print("reading zarr")
        callset = zarr.open_group(zarr_path, mode="r")
        gt = callset["calldata/GT"]
        genotypes = allel.GenotypeArray(gt[:])
        samples = callset["samples"][:]
        return genotypes, samples

    def _load_from_vcf(self, vcf_path):
        """Load genotypes from VCF file.

        Args:
            vcf_path: Path to VCF file containing genotype data

        Returns:
            tuple: (genotypes, samples) where:
                - genotypes is an allel.GenotypeArray containing genetic data
                - samples is a numpy array of sample IDs

        Raises:
            ValueError: If VCF file cannot be read
        """
        print("reading VCF")
        vcf = allel.read_vcf(vcf_path)
        if vcf is None:
            raise ValueError(f"Could not read VCF file: {vcf_path}")
        genotypes = allel.GenotypeArray(vcf["calldata/GT"])
        samples = vcf["samples"]
        return genotypes, samples

    def _load_from_matrix(self, matrix_path):
        """Load genotypes from matrix file.

        Args:
            matrix_path: Path to tab-delimited matrix file containing genotype data.
                File should have a header row with 'sampleID' as first column,
                followed by variant columns. Each row contains genotype counts (0,1,2)
                for one sample.

        Returns:
            tuple: (genotypes, samples) where:
                - genotypes is an allel.GenotypeArray containing genetic data
                - samples is a numpy array of sample IDs
        """
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
        """Load genotype data from various input file formats.

        Args:
            vcf: Path to VCF file containing genotype data
            zarr: Path to Zarr file containing genotype data
            matrix: Path to tab-delimited matrix file containing genotype data

        Returns:
            tuple: (genotypes, samples) where genotypes is an allel.GenotypeArray
            and samples is a numpy array of sample IDs

        Raises:
            ValueError: If no input file is specified
        """
        """Load genotype data from various sources"""
        if zarr is not None:
            return self._load_from_zarr(zarr)
        elif vcf is not None:
            return self._load_from_vcf(vcf)
        elif matrix is not None:
            return self._load_from_matrix(matrix)
        else:
            raise ValueError("No input specified. Please provide vcf, zarr, or matrix")

    def _split_train_test(self, genotypes, locations, train_split=0.9):
        """Split genotype and location data into training and test sets.

        Args:
            genotypes: GenotypeArray containing genetic data for all samples
            locations: Array of geographic coordinates (x,y) for each sample,
                      with NaN values for samples with unknown locations
            train_split: Proportion of samples to use for training (default: 0.9)

        Returns:
            tuple: (train_idx, test_idx, train_gen, test_gen, train_locs, test_locs, pred_idx, pred_gen)
                train_idx: Indices of training samples
                test_idx: Indices of test samples
                train_gen: Genotype data for training samples
                test_gen: Genotype data for test samples
                train_locs: Location data for training samples
                test_locs: Location data for test samples
                pred_idx: Indices of samples with unknown locations
                pred_gen: Genotype data for samples with unknown locations
        """
        # Get indices of samples with known locations
        train = np.argwhere(~np.isnan(locations[:, 0]))
        train = np.array([x[0] for x in train])

        # Get indices of samples with unknown locations
        pred = np.array([x for x in range(len(locations)) if x not in train])

        # Split known locations into train/test
        test = np.random.choice(
            train, round((1 - train_split) * len(train)), replace=False
        )
        train = np.array([x for x in train if x not in test])

        # Prepare data arrays
        traingen = np.transpose(genotypes[:, train])
        testgen = np.transpose(genotypes[:, test])
        trainlocs = locations[train]
        testlocs = locations[test]
        predgen = np.transpose(genotypes[:, pred])

        return train, test, traingen, testgen, trainlocs, testlocs, pred, predgen

    def _create_callbacks(self, boot=0):
        """Create Keras callbacks for training.

        Args:
            boot: Bootstrap replicate number (default: 0)

        Returns:
            list: List of Keras callbacks [ModelCheckpoint, EarlyStopping, ReduceLROnPlateau]
        """
        filepath = (
            f"{self.config['out']}_boot{boot}.weights.h5"
            if self.config.get("bootstrap", False)
            else f"{self.config['out']}.weights.h5"
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

    def train(
        self,
        *,  # Force keyword arguments
        genotypes,
        samples,
        sample_data_file=None,
        boot=0,
    ):
        """Train the locator model on genotype data.

        Args:
            genotypes: Array of genotype data
            samples: Sample IDs corresponding to genotypes
            sample_data_file: Path to sample data file (overrides config["sample_data"])
            boot: Bootstrap replicate number (default: 0)

        Raises:
            ValueError: If sample_data file path is not provided in config or as argument

        Returns:
            keras.callbacks.History: Training history
        """
        # Get sample data file path from argument or config
        sample_data_path = sample_data_file or self.config.get("sample_data")
        if not sample_data_path:
            raise ValueError(
                "sample_data file path must be provided in config or as argument"
            )

        # Get sorted sample data and locations
        sample_data, locs = self.sort_samples(samples, sample_data_path)

        # Normalize locations
        self.meanlong, self.sdlong, self.meanlat, self.sdlat, normalized_locs = (
            normalize_locs(locs)
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
            self._split_train_test(
                filtered_genotypes,
                normalized_locs,
                train_split=self.config.get("train_split", 0.9),
            )
        )

        # Store prediction and test data for later use
        self.pred_indices = pred
        self.predgen = predgen
        self.testgen = testgen
        self.testlocs = testlocs

        # Create and train model
        self.model = create_network(
            input_shape=traingen.shape[1],
            width=self.config.get("width", 256),
            n_layers=self.config.get("nlayers", 8),
            dropout_prop=self.config.get("dropout_prop", 0.25),
        )

        callbacks = self._create_callbacks(boot=boot)

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

    def predict(self, boot=0, verbose=True):
        """Make predictions for samples with unknown locations.

        Args:
            boot (int, optional): Bootstrap replicate number. Defaults to 0.
            verbose (bool, optional): Whether to print validation metrics. Defaults to True.

        Returns:
            numpy.ndarray: Array of predicted coordinates (longitude, latitude) for samples
                with unknown locations.

        Raises:
            ValueError: If model has not been trained before prediction.
        """
        """Predict locations"""
        if self.model is None:
            raise ValueError("Model must be trained before prediction")

        # Use stored prediction data
        predictions = self.model.predict(self.predgen)

        # Calculate validation metrics using test data
        test_predictions = self.model.predict(self.testgen)

        # Denormalize test predictions and actual test locations
        test_predictions = np.array(
            [
                [x[0] * self.sdlong + self.meanlong, x[1] * self.sdlat + self.meanlat]
                for x in test_predictions
            ]
        )
        test_actual = np.array(
            [
                [x[0] * self.sdlong + self.meanlong, x[1] * self.sdlat + self.meanlat]
                for x in self.testlocs
            ]
        )

        if verbose:
            # Calculate R² for longitude and latitude
            r2_long = np.corrcoef(test_predictions[:, 0], test_actual[:, 0])[0][1] ** 2
            r2_lat = np.corrcoef(test_predictions[:, 1], test_actual[:, 1])[0][1] ** 2

            # Calculate mean and median distances
            from scipy import spatial

            distances = [
                spatial.distance.euclidean(test_predictions[x, :], test_actual[x, :])
                for x in range(len(test_predictions))
            ]
            mean_dist = np.mean(distances)
            median_dist = np.median(distances)

            print(
                f"R²(x)={r2_long:.4f}\n"
                f"R²(y)={r2_lat:.4f}\n"
                f"Mean validation error: {mean_dist:.4f}\n"
                f"Median validation error: {median_dist:.4f}\n"
            )

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

    def sort_samples(self, samples=None, sample_data_file=None):
        """Sort samples and match with location data from a sample data file.

        This method reads a tab-delimited sample data file containing location coordinates,
        matches the samples with the genotype data, and ensures the ordering is consistent.
        The sample data file must contain columns 'sampleID', 'x', and 'y', where x and y
        represent geographic coordinates. Sample IDs must exactly match those in the genotype data.

        Args:
            samples (numpy.ndarray): Array of sample IDs from the genotype data
            sample_data_file (str): Path to tab-delimited file with columns 'sampleID', 'x', 'y'.
                                  X and Y values for samples without known locations should be NA.

        Returns:
            tuple: A tuple containing:
                - sample_data (pandas.DataFrame): DataFrame with sample metadata and coordinates
                - locs (numpy.ndarray): Array of x,y coordinates for each sample

        Raises:
            ValueError: If sample_data file is missing 'sampleID' column or if sample IDs
                      don't match between genotype and sample data.
        """
        """Sort samples and match with location data

        Args:
            samples: array of sample IDs from genotype data
            sample_data_file: path to tab-delimited file with columns 'sampleID', 'x', 'y'

        Returns:
            tuple: (sample_data DataFrame, locations array)
        """
        if samples is None or sample_data_file is None:
            raise ValueError("samples and sample_data_file must be provided")

        # Read sample data file
        sample_data = pd.read_csv(sample_data_file, sep="\t")

        # Ensure sampleID column exists
        if "sampleID" not in sample_data.columns:
            raise ValueError("sample_data file must contain 'sampleID' column")

        # Create backup of sampleID and set as index
        sample_data["sampleID2"] = sample_data["sampleID"]
        sample_data.set_index("sampleID", inplace=True)

        # Reindex to match genotype sample order
        samples = samples.astype("str")
        sample_data = sample_data.reindex(np.array(samples))

        # Verify sample order matches
        if not all(
            [
                sample_data["sampleID2"].iloc[x] == samples[x]
                for x in range(len(samples))
            ]
        ):
            raise ValueError(
                "Sample ordering failed! Check that sample IDs match the VCF."
            )

        # Extract location data
        locs = np.array(sample_data[["x", "y"]])

        return sample_data, locs

    def plot_history(self, history):
        """Plot training history and prediction error.

        Creates a figure with two subplots showing the validation loss and training loss
        over epochs. Saves the plot to a PDF file using the output prefix specified in config.

        Args:
            history: keras.callbacks.History object containing training history
        """
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

    def run_windows(
        self, genotypes, samples, window_start=0, window_size=5e5, window_stop=None
    ):
        """Run analysis in windows across the genome.

        Args:
            genotypes: Array of genotype data
            samples: Sample IDs corresponding to genotypes
            window_start (int, optional): Start position for windowed analysis. Defaults to 0.
            window_size (float, optional): Size of each window in base pairs. Defaults to 5e5.
            window_stop (int, optional): Stop position for windowed analysis.
                Defaults to None (max position).
        """
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
        """Run jacknife analysis by dropping SNPs.

        Args:
            genotypes: Array of genotype data
            samples: Sample IDs corresponding to genotypes
            prop (float, optional): Proportion of SNPs to drop in each replicate.
                Defaults to 0.05.

        The jacknife analysis:
        1. Gets base predictions using all SNPs
        2. Runs multiple replicates dropping a random subset of SNPs
        3. Calculates uncertainty estimates from the variation in predictions
        4. Saves results including standard deviations of predictions
        """
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

            self.train(genotypes=jack_genos, samples=samples)
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
