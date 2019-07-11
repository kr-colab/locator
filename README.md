# Locator

Locator is a supervised machine learning method for predicting geographic location from
genotype or sequencing data. 

# Installation 

Locator requires python3 and the following packages:
allel, zarr, numpy, pandas, tensorflow, keras, and scipy. 

[[add links for installations and conda instructions]]
 
For large datasets or bootstrap uncertainty estimation we recommend you 
run Locator on a CUDA-enabled GPU (it will be 20-100x faster; Installation 
instructions can be found at https://www.tensorflow.org/install/gpu). However 
the program should run fine on CPU for smaller datasets. The test data takes 
~1 minute to run on CPUs for a 12-core laptop. 

# Examples

[[add test dataset and run parameters]]

This command will fit a cross-validation locator model to test data from a 
continuous-space SLiM simulation: 

python scripts/locator_dev.py --vcf data/test_genotypes.vcf.gz --sample_data data/test_sample_data.txt --out out/test


# Parameters
