# Example API for locator

from locator import Locator


# Override some defaults
locator = Locator({"out": "my_analysis", "train_split": 0.9, "batch_size": 64})

# For your specific example, you might want:
locator = Locator({"out": "test_output", "sample_data": "data/test_sample_data.txt"})

# Load and sort data
genotypes, samples = locator.load_genotypes(
    vcf="data/test_genotypes.vcf.gz",
)

locator.sort_samples(
    samples=samples,
    sample_data_file="data/test_sample_data.txt",
)

# Train the model
locator.train(
    genotypes=genotypes,
    samples=samples,
)

# do prediction
locator.predict(verbose=True)

# do jacknife
locator.run_jacknife(genotypes, samples)

# plot the results
