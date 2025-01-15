"""Core functionality for locator"""

import numpy as np
import pandas as pd
import allel
import zarr
import sys
from tensorflow import keras
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm

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
        boot=None,
        train_gen=None,
        test_gen=None,
        pred_gen=None,
        train_locs=None,
        test_locs=None,
        setup_only=False,
    ):
        """Train the locator model on genotype data."""
        # Store samples
        self.samples = samples

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

        # Filter SNPs if not using pre-processed data
        if train_gen is None:
            filtered_genotypes = filter_snps(
                genotypes,
                min_mac=self.config.get("min_mac", 2),
                max_snps=self.config.get("max_SNPs"),
                impute=self.config.get("impute_missing", False),
            )

            # Split data
            (
                train,
                test,
                self.traingen,
                self.testgen,
                trainlocs,
                testlocs,
                pred,
                self.predgen,
            ) = self._split_train_test(
                filtered_genotypes,
                normalized_locs,
                train_split=self.config.get("train_split", 0.9),
            )
            # Store prediction indices
            self.pred_indices = pred
        else:
            # Use pre-processed data (for bootstrapping)
            self.traingen = train_gen
            self.testgen = test_gen
            self.predgen = pred_gen
            # Use provided locations if available
            if train_locs is not None and test_locs is not None:
                trainlocs = train_locs
                testlocs = test_locs
            else:
                # Get train/test indices and locations from original split
                train = np.where(~np.isnan(normalized_locs[:, 0]))[0]
                test = np.random.choice(
                    train,
                    round((1 - self.config.get("train_split", 0.9)) * len(train)),
                    replace=False,
                )
                train = np.array([x for x in train if x not in test])
                trainlocs = normalized_locs[train]
                testlocs = normalized_locs[test]

        # Store both training and test locations
        self.trainlocs = trainlocs
        self.testlocs = testlocs

        # Create and train model if not already created
        if self.model is None:
            self.model = create_network(
                input_shape=self.traingen.shape[1],
                width=self.config.get("width", 256),
                n_layers=self.config.get("nlayers", 8),
                dropout_prop=self.config.get("dropout_prop", 0.25),
            )

        # Return early if setup_only
        if setup_only:
            return None

        callbacks = self._create_callbacks(boot=boot)

        self.history = self.model.fit(
            self.traingen,
            trainlocs,
            epochs=self.config.get("max_epochs", 5000),
            batch_size=self.config.get("batch_size", 32),
            shuffle=True,
            verbose=self.config.get("keras_verbose", 1),
            validation_data=(self.testgen, testlocs),
            callbacks=callbacks,
        )

        # Save training history
        hist_df = pd.DataFrame(self.history.history)
        hist_df.to_csv(f"{self.config['out']}_history.txt", sep="\t", index=False)

        return self.history

    def predict(
        self,
        boot=0,
        verbose=True,
        prediction_genotypes=None,
        return_df=False,
        save_preds_to_disk=True,
    ):
        """Make predictions for samples with unknown locations.

        Args:
            boot (int, optional): Bootstrap replicate number. Defaults to 0.
            verbose (bool, optional): Whether to print validation metrics. Defaults to True.
            prediction_genotypes (numpy.ndarray, optional): Override default prediction genotypes.
                Used for jacknife resampling. Defaults to None.
            return_df (bool, optional): Whether to return predictions as pandas DataFrame.
                Defaults to False.
            save_preds_to_disk (bool, optional): Whether to save predictions to disk.
                Defaults to True.
        Returns:
            numpy.ndarray or pandas.DataFrame: Array of predicted coordinates or DataFrame with
                x,y coordinates and sampleID columns
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")

        # Use provided prediction genotypes if available, otherwise use stored ones
        predgen = (
            prediction_genotypes if prediction_genotypes is not None else self.predgen
        )

        # Get predictions
        predictions = self.model.predict(predgen)

        # Denormalize predictions
        predictions = np.array(
            [
                [x[0] * self.sdlong + self.meanlong, x[1] * self.sdlat + self.meanlat]
                for x in predictions
            ]
        )

        # Create DataFrame
        pred_df = pd.DataFrame(predictions, columns=["x", "y"])
        if hasattr(self, "samples") and hasattr(self, "pred_indices"):
            pred_df.insert(0, "sampleID", self.samples[self.pred_indices])

        # Save predictions to file
        outfile = (
            f"{self.config['out']}_boot{boot}_predlocs.txt"
            if self.config.get("bootstrap", False) or self.config.get("jacknife", False)
            else f"{self.config['out']}_predlocs.txt"
        )
        if save_preds_to_disk:
            pred_df.to_csv(outfile, index=False)

        if return_df:
            return pred_df

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
        self,
        genotypes,
        samples,
        window_start=0,
        window_size=5e5,
        window_stop=None,
        return_df=False,
        save_full_pred_matrix=True,
    ):
        # Store samples
        self.samples = samples

        # Get positions from zarr
        if not hasattr(self, "positions"):
            if not self.config.get("zarr"):
                raise ValueError(
                    "zarr path must be provided in config for windowed analysis"
                )
            callset = zarr.open_group(self.config["zarr"], mode="r")
            self.positions = callset["variants/POS"][:]

        if window_stop is None:
            window_stop = max(self.positions)

        windows = range(int(window_start), int(window_stop), int(window_size))

        # Initial training to set up model and data
        first_window = (self.positions >= int(window_start)) & (
            self.positions < int(window_start + window_size)
        )
        if sum(first_window) > 0:
            window_genos = genotypes[first_window, :, :]
            self.train(genotypes=window_genos, samples=samples)

        # Create lists to store predictions
        pred_dfs = []

        print("starting window analysis")
        for start in tqdm(windows):
            stop = start + int(window_size)
            in_window = (self.positions >= start) & (self.positions < stop)

            if sum(in_window) > 0:
                # Get genotypes for this window
                window_genos = genotypes[in_window, :, :]

                # Clear existing model
                self.model = None

                # Train on window data
                self.train(genotypes=window_genos, samples=samples)

                # Get predictions using self.predgen which is already properly formatted
                preds = self.predict(
                    return_df=True, save_preds_to_disk=not save_full_pred_matrix
                )

                if return_df:
                    # Rename columns to include window start
                    boot_preds = preds[["x", "y"]].copy()
                    boot_preds.columns = [f"x_win{start}", f"y_win{start}"]
                    pred_dfs.append(boot_preds)

                # Clear keras session
                keras.backend.clear_session()

        if return_df:
            # Concatenate all predictions and add sampleIDs
            all_predictions = pd.concat([preds[["sampleID"]], *pred_dfs], axis=1)

            if save_full_pred_matrix:
                all_predictions.to_csv(
                    f"{self.config['out']}_windows_predlocs.csv", index=False
                )
            return all_predictions

        return None

    def run_jacknife(
        self,
        genotypes,
        samples,
        prop=0.05,
        return_df=False,
        save_full_pred_matrix=True,
    ):
        """Run jacknife analysis by dropping SNPs.

        Args:
            genotypes: Array of genotype data
            samples: Sample IDs corresponding to genotypes
            prop (float, optional): Proportion of SNPs to drop in each replicate.
                Defaults to 0.05.
            return_df (bool, optional): Whether to return DataFrame of all predictions.
                Defaults to False.
            save_full_pred_matrix (bool, optional): Whether to save the full prediction matrix.
                Defaults to True.

        Returns:
            pandas.DataFrame or None: If return_df=True, returns DataFrame containing
                all predictions, with columns named 'x_0', 'y_0', 'x_1', 'y_1', etc.
                for each jacknife replicate. Row index contains sample IDs.
        """
        # Store samples
        self.samples = samples

        # Set jacknife flag in config
        self.config["jacknife"] = True

        # Set up prediction indices if not already done
        if not hasattr(self, "pred_indices"):
            # Get sample data
            sample_data = pd.read_csv(self.config["sample_data"], sep="\t")
            # Find samples without locations (NA in x or y)
            pred = sample_data.index[sample_data.x.isna() | sample_data.y.isna()].values
            # Convert to indices in the samples array
            self.pred_indices = np.where(
                np.isin(np.array(samples), sample_data.index[pred])
            )[0]

        # Create lists to store predictions
        pred_dfs = []
        preds = None

        # Initial training to set up model (but don't output predictions)
        self.train(genotypes=genotypes, samples=samples)

        print("starting jacknife resampling")
        af = []
        # Convert genotypes to allele counts first
        ac = genotypes.to_allele_counts()[:, :, 1]  # Get counts of alternate allele

        # Calculate allele frequencies
        for i in tqdm(range(ac.shape[0])):
            freq = np.sum(ac[i, :]) / (ac.shape[1] * 2)
            af.append(freq)
        af = np.array(af)

        for boot in tqdm(range(self.config.get("nboots", 50))):
            callbacks = self._create_callbacks(boot)
            pg = copy.deepcopy(self.predgen)

            sites_to_remove = np.random.choice(
                pg.shape[1], int(pg.shape[1] * prop), replace=False
            )

            for i in sites_to_remove:
                pg[:, i] = np.random.binomial(2, af[i], size=pg.shape[0])

            # Get predictions
            preds = self.predict(
                boot=boot,
                verbose=False,
                prediction_genotypes=pg,
                return_df=True,
                save_preds_to_disk=not save_full_pred_matrix,
            )

            # Rename columns to include boot number
            boot_preds = preds[["x", "y"]].copy()
            boot_preds.columns = [f"x_{boot}", f"y_{boot}"]
            pred_dfs.append(boot_preds)

        if return_df:
            # Concatenate all predictions and add sampleIDs
            all_predictions = pd.concat([preds[["sampleID"]], *pred_dfs], axis=1)

            if save_full_pred_matrix:
                all_predictions.to_csv(
                    f"{self.config['out']}_jacknife_predlocs.csv", index=False
                )
            return all_predictions

        return None

    def run_bootstraps(
        self,
        genotypes,
        samples,
        n_bootstraps=50,
        return_df=False,
        save_full_pred_matrix=True,
    ):
        # Store samples
        self.samples = samples

        # Set bootstrap flag in config
        self.config["bootstrap"] = True
        self.config["nboots"] = n_bootstraps

        # Initial training to set up model and data
        self.train(genotypes=genotypes, samples=samples)

        # Store original locations
        original_trainlocs = self.trainlocs
        original_testlocs = self.testlocs

        # Create lists to store predictions
        pred_dfs = []

        print("starting bootstrap resampling")

        for boot in tqdm(range(n_bootstraps)):
            # Set random seed
            np.random.seed(np.random.choice(range(int(1e6)), 1))

            # Create copies of data
            traingen2 = copy.deepcopy(self.traingen)
            testgen2 = copy.deepcopy(self.testgen)
            predgen2 = copy.deepcopy(self.predgen)

            # Resample sites with replacement
            site_order = np.random.choice(
                traingen2.shape[1], traingen2.shape[1], replace=True
            )

            # Reorder sites in all datasets
            traingen2 = traingen2[:, site_order]
            testgen2 = testgen2[:, site_order]
            predgen2 = predgen2[:, site_order]

            # Clear existing model
            self.model = None

            # Train on bootstrapped data with original locations
            self.train(
                genotypes=None,
                samples=samples,
                boot=boot,
                train_gen=traingen2,
                test_gen=testgen2,
                pred_gen=predgen2,
                train_locs=original_trainlocs,
                test_locs=original_testlocs,
            )

            # Get predictions
            preds = self.predict(
                boot=boot,
                verbose=False,
                prediction_genotypes=predgen2,
                return_df=True,
                save_preds_to_disk=not save_full_pred_matrix,
            )

            if return_df:
                # Rename columns to include boot number
                boot_preds = preds[["x", "y"]].copy()
                boot_preds.columns = [f"x_{boot}", f"y_{boot}"]
                pred_dfs.append(boot_preds)

            # Clear keras session
            keras.backend.clear_session()

        if return_df:
            # Concatenate all predictions and add sampleIDs
            all_predictions = pd.concat([preds[["sampleID"]], *pred_dfs], axis=1)

            if save_full_pred_matrix:
                all_predictions.to_csv(
                    f"{self.config['out']}_bootstrap_predlocs.csv", index=False
                )
            return all_predictions

        return None

    def train_holdout(self, genotypes, samples, k=10):
        """Train the model while holding out k samples with known locations."""
        # Store samples
        self.samples = samples

        print("\nDEBUG: Initial shapes:")
        print(f"genotypes: {genotypes.shape}")
        print(f"samples: {len(samples)}")

        # Get sample data file path
        sample_data_path = self.config.get("sample_data")
        if not sample_data_path:
            raise ValueError("sample_data file path must be provided in config")

        # Get sorted sample data and locations
        sample_data, locs = self.sort_samples(samples, sample_data_path)
        print(f"\nDEBUG: After sort_samples:")
        print(f"locs shape: {locs.shape}")

        # Get indices of samples with known locations
        known_idx = np.argwhere(~np.isnan(locs[:, 0]))
        known_idx = np.array([x[0] for x in known_idx])
        print(f"\nDEBUG: Known locations:")
        print(f"Number of samples with known locations: {len(known_idx)}")

        # Randomly select k samples to hold out
        holdout_idx = np.random.choice(known_idx, k, replace=False)
        mask = np.ones(len(locs), dtype=bool)
        mask[holdout_idx] = False

        print(f"\nDEBUG: Holdout info:")
        print(f"Number of holdout samples: {len(holdout_idx)}")
        print(f"Number of remaining samples: {np.sum(mask)}")

        # Filter SNPs
        filtered_genotypes = filter_snps(
            genotypes,
            min_mac=self.config.get("min_mac", 2),
            max_snps=self.config.get("max_SNPs"),
            impute=self.config.get("impute_missing", False),
        )
        print(f"\nDEBUG: After filter_snps:")
        print(f"filtered_genotypes shape: {filtered_genotypes.shape}")

        # Split remaining samples into train/test
        (
            train_idx,
            test_idx,
            self.traingen,
            self.testgen,
            train_locs,
            test_locs,
            pred_idx,
            pred_gen,
        ) = self._split_train_test(
            filtered_genotypes[:, mask],
            locs[mask],
            train_split=self.config.get("train_split", 0.9),
        )

        print(f"\nDEBUG: After split_train_test:")
        print(f"traingen shape: {self.traingen.shape}")
        print(f"testgen shape: {self.testgen.shape}")
        print(f"train_locs shape: {train_locs.shape}")
        print(f"test_locs shape: {test_locs.shape}")

        # Normalize locations using training data
        self.meanlong, self.sdlong, self.meanlat, self.sdlat, normalized_train_locs = (
            normalize_locs(train_locs)
        )

        # Store training and test data
        self.trainlocs = normalized_train_locs
        self.testlocs = np.array(
            [
                [
                    (x[0] - self.meanlong) / self.sdlong,
                    (x[1] - self.meanlat) / self.sdlat,
                ]
                for x in test_locs
            ]
        )

        # Store holdout data
        self.holdout_idx = holdout_idx
        self.holdout_gen = np.transpose(filtered_genotypes[:, holdout_idx])
        holdout_locs = locs[holdout_idx]
        self.holdout_locs = np.array(
            [
                [
                    (x[0] - self.meanlong) / self.sdlong,
                    (x[1] - self.meanlat) / self.sdlat,
                ]
                for x in holdout_locs
            ]
        )

        # Create new model (force recreation)
        self.model = create_network(
            input_shape=self.traingen.shape[1],
            width=self.config.get("width", 256),
            n_layers=self.config.get("nlayers", 8),
            dropout_prop=self.config.get("dropout_prop", 0.25),
        )

        callbacks = self._create_callbacks()

        self.history = self.model.fit(
            self.traingen,
            self.trainlocs,
            epochs=self.config.get("max_epochs", 5000),
            batch_size=self.config.get("batch_size", 32),
            shuffle=True,
            verbose=self.config.get("keras_verbose", 1),
            validation_data=(self.testgen, self.testlocs),
            callbacks=callbacks,
        )

        # Save training history
        hist_df = pd.DataFrame(self.history.history)
        hist_df.to_csv(f"{self.config['out']}_history.txt", sep="\t", index=False)

        return self.history
