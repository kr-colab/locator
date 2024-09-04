"""A shiny new locator implementation."""

import defopt
import numpy as np
import polars as pl
import keras
import attr
import zarr
from functools import cached_property
from tqdm import tqdm
import logging
from tensorflow_probability import distributions as tfd
from tensorflow import reshape as tf_reshape
import tensorflow as tf

logging.basicConfig(level=logging.INFO)

from collections import namedtuple

from typing import List
from bokeh.plotting import figure, show, save

TrainedModel = namedtuple("TrainedModel", ["history", "model"])


@attr.s(frozen=False)
class Locator:
    """Class for inference of sample location from genotypes.

    Parameters:
        vcf_path: path to a VCF file.
        sample_locs_path: path to a csv listing sample locations with columns
            sample_ID, x_coordinate, y_coordinate
        train_prop: fraction of samples to use in model training. Defaults to 80%.
        predict_samples: list of sample IDs to predict

    Methods:

    Properties:
        vcf: polars dataframe of the vcf
        samples: list of sample IDs
        ploidy: integer copy number, used for all sites
        allele_counts: polars dataframe of ALT allele counts by site. Non-genotyped sites are
            np.nan.
        site_af: list of ALT allele frequency for each site in the vcf. Missing data is np.nan.
        afs: polars dataframe of ALT allele frequencies in each sample (e.g. for ploidy=2,
            0=homozygous reference, 0.5 = heterozygous, 1=homozygous alternate)



    """

    vcf_path: str = attr.ib()
    sample_locs_path: str = attr.ib()
    predict_samples: List[str] = attr.ib()
    train_prop: float = attr.ib(default=0.8)
    seed: int = attr.ib(default=42)
    depth: int = attr.ib(default=6)
    width: int = attr.ib(default=128)
    patience: int = attr.ib(default=50)
    batch_size: int = attr.ib(default=16)
    out: str = attr.ib(default="locator")
    dropout_prop: float = attr.ib(default=0.2)
    max_epochs: int = attr.ib(default=int(1e5))

    ###############################################################################################
    ##################################### Data prep ###############################################
    ###############################################################################################
    def read_vcf(self) -> pl.DataFrame:
        """Load an uncompressed VCF as a polars dataframe."""
        logging.info("reading vcf")
        v = pl.read_csv(
            self.vcf_path, separator="\t", comment_prefix="##", has_header=True
        )
        return v.rename({"#CHROM": "CHROM"})

    @cached_property
    def vcf(self) -> pl.DataFrame:
        """Polars dataframe of the vcf."""
        return self.read_vcf()

    @cached_property
    def samples(self) -> List[str]:
        """list of sample IDs from the vcf.

        Returns:
            list sample IDs"""
        return self.vcf.columns[9:]

    @cached_property
    def ploidy(self) -> int:
        """Sample ploidy inferred from the VCF GT column.

        Assumes all sites have the same copy number as the first genotyped site in the first sample
        (i.e. no sex chromosomes or CNAs).

        Returns:
            integer
        """
        return len(self.vcf[self.samples[0]][0].replace("|", "/").split("/"))

    @cached_property
    def allele_counts(self) -> pl.DataFrame:
        """Polars dataframe of ALT allele counts with shape (n_sites,n_samples), including nans."""
        v = self.read_vcf()
        logging.info("loading genotypes")
        samples = v.columns[9 : len(v.columns)]
        allele_counts = {}
        for sample in tqdm(samples):
            allele_counts[sample] = [
                x.count("1") if not x.startswith(".") else np.nan for x in v[sample]
            ]
        return pl.DataFrame(allele_counts)

    @cached_property
    def site_af(self) -> pl.Series:
        """Polars series of global ALT allele frequencies by site, ignoring missing data.

        Returns:
            polars series of floats of allele frequency by site"""
        return self.allele_counts.map_rows(
            lambda a: np.nansum(a) / (self.ploidy * np.nansum(~np.isnan(a)))
        )

    @cached_property
    def afs(self) -> pl.DataFrame:
        """Polars dataframe of ALT allele frequencies with shape(n_sites,n_sample).

        Missing genotypes are filled with the global allele frequency. Invariant sites are dropped.

        Returns:
            polars dataframe of floats with shape (n_sites,n_samples).
        """
        ac = self.allele_counts.to_numpy() ## should figure out how to do this nicely in polars
        for i in range(ac.shape[0]):
            ac[i,np.isnan(ac[i,:])]=self.site_af[i]
        ac = pl.DataFrame(ac)
        ac.columns = self.allele_counts.columns
        return ac / self.ploidy

    @cached_property
    def sample_splits(self) -> dict:
        """Dictionary of sample names for the training and validation sets."""
        non_test_samples = [x for x in self.samples if x not in self.predict_samples]
        train = list(
            np.random.choice(
                non_test_samples,
                size=int(self.train_prop * len(non_test_samples)),
                replace=False,
            )
        )
        val = [x for x in non_test_samples if x not in train]
        return {"train": train, "val": val}

    @cached_property
    def train_af(self) -> np.ndarray:
        """Numpy array of genotype AF for prediction samples with shape (n_samples,n_sites)."""
        return np.array(self.afs.select(self.sample_splits["train"])).transpose()

    @cached_property
    def val_af(self) -> np.ndarray:
        """Numpy array of genotype AF for validation samples with shape (n_samples,n_sites)."""
        return np.array(self.afs.select(self.sample_splits["val"])).transpose()

    @cached_property
    def predict_af(self) -> np.ndarray:
        """Numpy array of genotype AF for prediction samples with shape (n_samples,n_sites)."""
        return np.array(self.afs.select(self.predict_samples)).transpose()

    @cached_property
    def sample_locs(self) -> pl.DataFrame:
        """Polars dataframe of sample locations with columns sample_id, x_coord, y_coord."""
        locs = pl.read_csv(
            self.sample_locs_path,
            new_columns=["sample", "x", "y"],
            dtypes=[pl.String, pl.Float32, pl.Float32],
            null_values=["NA", "na", "NaN", "NAN", "", "Inf", "-Inf"],
        )
        self.x_mean=locs['x'].mean()
        self.x_sd=locs['x'].std()
        self.y_mean=locs['y'].mean()
        self.y_sd=locs['y'].std()
        locs = locs.with_columns(
            x_norm=(pl.col('x')-self.x_mean)/self.x_sd,
            y_norm=(pl.col('y')-self.y_mean)/self.y_sd
        )
        return locs

    def get_sorted_locs(self, samples: List[str]) -> pl.DataFrame:
        """Get a set of sample locations sorted by the samples input.

        Args:
            samples: list of sample names.

        Returns:
            polars dataframe with columns sample,x,y
        """
        locs = []
        for sample in samples:
            locs.append(self.sample_locs.filter(pl.col("sample") == sample))
        return pl.concat(locs)

    def validate_names(self) -> bool:
        locs_in_gt=all([x in list(self.sample_splits.values()) for x in self.samples])
        gt_in_locs=all([x in self.samples for x in list(self.sample_splits.values())])
        if not locs_in_gt and gt_in_locs:
            logging.error("Error: non-matchine sample IDs in vcf and location inputs.")
        return locs_in_gt and gt_in_locs
    
    @cached_property
    def train_locs(self) -> np.ndarray:
        """Numpy array of x, y coords for training samples with shape (n_samples,2).

        Samples are sorted to match the column order of self.train_af.
        """
        return np.array(
            self.get_sorted_locs(self.sample_splits["train"]).select(["x_norm", "y_norm"])
        )

    @cached_property
    def val_locs(self) -> np.ndarray:
        """Numpy array of x, y coords for validation samples with shape (n_samples,2).

        Samples are sorted to match the column order of self.val_af.
        """
        return np.array(
            self.get_sorted_locs(self.sample_splits["val"]).select(["x_norm", "y_norm"])
        )

    ###############################################################################################
    ################################# Neural network setup ########################################
    ###############################################################################################
    def load_network(self):
        """Load a fully connected network of width x depth with dropout in the middle."""
        
        def reshape_to_covariance_matrix(ypred):
            # Extract components
            var1 = ypred[:, 0]  # Variance of the first variable
            var2 = ypred[:, 1]  # Variance of the second variable
            cov = ypred[:, 2]   # Covariance between the two variables
            
            # Create the covariance matrices
            cov_matrix = tf.stack([
                tf.stack([var1, cov], axis=1),
                tf.stack([cov, var2], axis=1)
            ], axis=1)
            
            return cov_matrix

        def mvn_loss_full(ytrue, ypred):
            means = ypred[:, 0:2]
            covs = reshape_to_covariance_matrix(ypred[:,2:5])
            d=tfd.MultivariateNormalFullCovariance(
                loc=means,
                covariance_matrix=covs,
                validate_args=True,
            )
            return keras.ops.mean(-d.log_prob(ytrue))
        
        def mvn_loss_diag(ytrue, ypred):
            means = ypred[:, 0:2]
            logsd = ypred[:,2:]
            d=tfd.MultivariateNormalDiag(
                loc=means,
                scale_diag=keras.ops.exp(logsd),
                validate_args=True,
            )
            return keras.ops.mean(-d.log_prob(ytrue))

        model = keras.Sequential()
        model.add(
            keras.layers.BatchNormalization(input_shape=(self.train_af.shape[1],))
        )
        for i in range(int(np.floor(self.depth / 2))):
            model.add(keras.layers.Dense(self.width, activation="elu"))
        model.add(keras.layers.Dropout(self.dropout_prop))
        for i in range(int(np.ceil(self.depth / 2))):
            model.add(keras.layers.Dense(self.width, activation="elu"))
        model.add(keras.layers.Dense(4))
        model.add(keras.layers.Dense(4))
        model.compile(optimizer="Adam", loss=mvn_loss_diag)
        return model

    @cached_property
    def callbacks(self):
        checkpointer = keras.callbacks.ModelCheckpoint(
            filepath=self.out + ".weights.h5",
            initial_value_threshold=1000,
            save_best_only=True,
            save_weights_only=True,
            monitor="val_loss",
            mode="min",
            save_freq="epoch",
        )
        earlystop = keras.callbacks.EarlyStopping(
            monitor="val_loss", min_delta=0, patience=self.patience
        )
        reducelr = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=int(self.patience / 6),
            mode="auto",
            min_delta=0,
            cooldown=0,
            min_lr=0,
        )
        return [checkpointer, earlystop, reducelr]

    @cached_property
    def fit(self):
        model = self.load_network()
        history = model.fit(
            self.train_af,
            self.train_locs,
            epochs=self.max_epochs,
            batch_size=self.batch_size,
            shuffle=True,
            validation_data=(self.val_af, self.val_locs),
            callbacks=self.callbacks,
        )
        model.load_weights(self.out + ".weights.h5")
        return TrainedModel(history.history, model)

    @cached_property
    def predict(self) -> pl.DataFrame:
        """Predict locations for self.predict_samples using a trained network.

        Args:
            trained_network: output of self.train_network.

        Returns:
            polars dataframe of predicted sample locations.
        """
        preds = self.fit.model.predict(self.predict_af)
        preds = pl.DataFrame(preds).with_columns(pl.Series(name="sample",values=self.predict_samples))
        preds.columns = ['pred_x','pred_y','log_sd_x','log_sd_y','sample']
        return preds
    
    @cached_property
    def training_predictions(self) -> pl.DataFrame:
        pred=pl.concat((pl.DataFrame(self.fit.model.predict(self.train_af)),
                        pl.DataFrame(self.fit.model.predict(self.val_af))))
        pred.columns=['pred_x','pred_y','log_sd_x','log_sd_y']
        pred = pred.with_columns(pl.Series(name="sample",
                                           values=[*self.sample_splits['train'],
                                                   *self.sample_splits['val']])
                                )
        truth=pl.concat((
            pl.DataFrame(self.train_locs).with_columns(set=pl.lit('training')),
            pl.DataFrame(self.val_locs).with_columns(set=pl.lit('validation')))
            )
        truth.columns=['true_x','true_y','set']
        return pl.concat((pred,truth),how='horizontal')
    
    ###############################################################################################
    #################################### postprocessing and plots #################################
    ###############################################################################################

    def plot_history(self, show_figure=False):
        """Plot model training history.

        Args:
            trained_network: output of self.train_network()

        Returns:
            None. If show=True, opens the plot in a web browser. If show=False, saves plot to
            self.out+training_history.html.
        """
        history = self.fit.history
        epochs = np.arange(1, len(history["val_loss"]))
        fig = figure(title="Model Training History")
        fig.line(
            x=epochs,
            y=history["loss"],
            color="orange",
            legend_label="training",
        )
        fig.line(
            x=epochs,
            y=history["val_loss"],
            legend_label="validation",
        )
        fig.xaxis.axis_label = "Epoch"
        fig.xaxis.axis_label = "Loss"
        if show_figure:
            show(fig)
        else:
            save(fig, self.out + ".locator_training_history.html")
        return None

    def plot_training_samples(self, show_figure=False, samples=None, predictions = False):
        """Generate a bokeh html plot of training and validation sample location predictions.

        Args:
            trained_network: output of self.train_network()

        Returns:
            None. If show=True, opens the plot in a web browser. If show=False, saves plot to
            self.out+training_sample_predictions.html."""
        fig = figure(title="Training and Validation Set Predictions")
        transparency = 0.5 if samples is None else 0.25
        training=self.training_predictions.filter(pl.col('set')=='training')
        validation=self.training_predictions.filter(pl.col('set')=='validation')
        fig.scatter(x=training['true_x'],
                    y=training['true_y'],
                    color='grey',
                    legend_label="Training Location",
                    alpha=transparency/2)
        for row in validation.iter_rows(named='True'):
            fig.line(x=[row['true_x'],row['pred_x']],
                     y=[row['true_y'],row['pred_y']],
                     color='grey',
                     alpha=transparency)
        fig.scatter(x=validation['true_x'],
                    y=validation['true_y'],
                    color='grey',
                    legend_label="Validation Location",
                    alpha=transparency)
        fig.scatter(x=validation['pred_x'],
                    y=validation['pred_y'],
                    color='steelblue',
                    legend_label="Validation Predicted Location",
                    alpha=transparency)
        if samples is not None:
            s=self.training_predictions.filter(pl.col("sample").is_in(samples))
            for row in s.iter_rows(named='True'):
                fig.line(x=[row['true_x'],row['pred_x']],
                        y=[row['true_y'],row['pred_y']],
                        color='grey',
                        alpha=1)
            fig.scatter(x=s['pred_x'],
                    y=s['pred_y'],
                    color='steelblue',
                    alpha=1)
            fig.ellipse(x=s['pred_x'],
                        y=s['pred_y'],
                        width=3.92*np.exp(s['log_sd_x']),
                        height=3.92*np.exp(s['log_sd_y']),
                        color='steelblue',
                        alpha=0.25)
        if predictions:
            fig.scatter(x=self.predict['pred_x'],
                    y=self.predict['pred_y'],
                    color='orangered',
                    legend_label='Predicted Locations',
                    alpha=1)
            # fig.ellipse(x=self.predict['pred_x'],
            #             y=self.predict['pred_y'],
            #             width=3.92*np.exp(self.predict['log_sd_x']),
            #             height=3.92*np.exp(self.predict['log_sd_y']),
            #             color='steelblue',
            #             alpha=0.25)
        if show_figure:
            show(fig)
        else:
            save(fig,self.out+".locator_training_predictions.html")
        return fig

    def generate_map_plot(self):
        """Generate a plot of predicted and true sample locations on a world map.

        Assumes sample locations are given as long, lat in a web mercator projection.

        Args:
            with_training: show training and validation sample locations + predictions on the map?

        Returns:
            None. If show=True, opens the plot in a web browser. If show=False, saves plot to
            self.out+locator_map.html.
        """
        return None


def locator(
    *,
    vcf: str,
    sample_locs: str,
    predict_samples: List[str],
    train_prop: float = 0.8,
    seed: int = 42,
    depth: int = 10,
    width: int = 256,
    patience: int = 50,
    batch_size: int = 32,
    out: str = "locator",
    dropout_prop: float = 0.2,
    max_epochs: int = int(1e5),
    plot: bool = True,
):
    """CLI runner function to fit a model and generate predictions for the test set.

    Args:
        vcf: path to an uncompressed vcf
        sample_locs: path to a csv of sample locations with columns sample,x,y
        predict_samples: list of samples to predict
        train_prop: fraction of non-prediction samples used for primary model training. Remaining
            samples are used for hyperparameter tuning (i.e. validation set).
        seed: random seed
        depth: network depth
        width: network width
        patience: epochs to wait after the last improvement in val_loss before dropping the
            learning rate
        batch_size: number of samples to run per minibatch
        out: stem path for output files
        dropout_prop: fraction of weights to zero in dropout in the middle of the network
        max_epochs: maximum epochs to train the model
        plot: generate html plots?

    Returns:
        polars dataframe of predicted locations for predict_samples
    """
    locator = Locator(
        vcf_path=vcf,
        sample_locs_path=sample_locs,
        predict_samples=predict_samples,
        train_prop=train_prop,
        seed=seed,
        depth=depth,
        width=width,
        patience=patience,
        batch_size=batch_size,
        out=out,
        dropout_prop=dropout_prop,
        max_epochs=max_epochs,
    )
    trained_network = locator.train_network()
    predictions = locator.predict_samples(trained_network)
    if plot:
        locator.plot_history(trained_network)
        locator.plot_training_samples(trained_network)

    predictions.write_csv(out + ".predicted_sample_locations.csv")

    return predictions


############### debug zone ################
# self = Locator(
#     vcf_path="/Users/cj/locator/data/test_genotypes2.vcf",
#     sample_locs_path="/Users/cj/locator/data/test_sample_data.samplexy.csv",
#     predict_samples=[f"msp_{x}" for x in range(50)],
#     train_prop=0.8,
#     patience=100,
#     depth=6,
#     width=256,
#     batch_size=32,
#     max_epochs=5000,
# )
# self.plot_training_samples(show_figure=True,
#                            samples=self.sample_splits['val'][10:20])

# pabu_locations=pl.read_csv("/Users/cj/locator/data/pabu_locations.csv")
# predict_samples=pabu_locations.filter(pabu_locations['Latitude']<27)['sampleID'].to_list()
# self = Locator(
#     vcf_path="/Users/cj/locator/data/pabu_test_genotypes.vcf",
#     sample_locs_path="/Users/cj/locator/data/pabu_locations.csv",
#     predict_samples=predict_samples,
#     train_prop=0.8,
#     patience=300,
#     depth=6,
#     width=64,
#     batch_size=16,
#     max_epochs=5000
# )
# self.plot_training_samples(show_figure=True,
#                            predictions=True)