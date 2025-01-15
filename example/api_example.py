# Example API for locator

from locator import Locator, plot_predictions


# Override some defaults
locator = Locator({"out": "my_analysis", "train_split": 0.9, "batch_size": 64})

# For your specific example, you might want:
locator = Locator(
    {
        "out": "windows",
        "sample_data": "data/test_sample_data.txt",
        "zarr": "data/test_genotypes.zarr",
    }
)

# Load and sort data
genotypes, samples = locator.load_genotypes(
    # vcf="data/test_genotypes.vcf.gz",
    zarr="data/test_genotypes.zarr",
)

# Train the model
# locator.train(genotypes=genotypes, samples=samples)

# do prediction
# all_predictions = locator.predict(verbose=True, return_df=True)

# do jacknife
all_predictions = locator.run_jacknife(genotypes, samples, return_df=True)

# do run_windows
# all_predictions = locator.run_windows(genotypes, samples, return_df=True)


## run bootstrap
# all_predictions = locator.run_bootstraps(
#     genotypes, samples, n_bootstraps=3, return_df=True
# )

plot_predictions(
    predictions=all_predictions, sample_data=samples, out_prefix="jacknife_example"
)


print(all_predictions.head())
