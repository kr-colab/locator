`Locator` is a supervised machine learning method for predicting the geographic origin of a sample from
genotype or sequencing data. A manuscript describing it and its use can be found at https://elifesciences.org/articles/54507

# Installation 

The easiest way to install `locator` is to download the github repo and run the setup script. It's usually a good idea to do this in a new conda environment (https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) to avoid version conflicts with other software: 

```
conda create --name locator
conda activate locator
git clone https://github.com/kr-colab/locator.git
cd locator
pip install -r req.txt
```

We recommend running on a CUDA-enabled GPU (https://www.tensorflow.org/install/gpu).

# Overview
`locator` reads in a set of genotypes and locations, trains a neural network to approximate the relationship between them, and predicts locations for a set of samples held out from the training routine. Samples with known locations are split randomly into a training set (used to fit model parameters) and a validation set (used to tune hyperparameters of the optimizer and evaluate error after training). Predictions are then generated for all samples with unknown coordinates. By fitting multiple models to different regions of the genome or to bootstrapped subsets of the full SNP matrix, the approach can also estimate uncertainty in a location estimate. 

# Inputs
Genotypes can read in from .vcf, vcf.gz, .zarr, or a tab-delimited table with first column 'sampleID' and each entry giving the count of minor (or derived) alleles for an individual at a site. The current implementation expects diploid inputs. Please file an issue if you'd like to use Locator for other ploidies.    

Sample metadata should be a tab-delimited file with the first row:  

`sampleID	x	y`

Use NA or NaN for x and y values of samples with unknown locations. Metadata must include all samples in the genotypes file. 


# Examples

This command should fit a model to a simulated test dataset of 
~10,000 SNPs and 450 individuals and predict the locations of 50 validation samples. 

```
cd ~/locator
mkdir out/test
locator.py --vcf data/test_genotypes.vcf.gz --sample_data data/test_sample_data.txt --out out/test/test
```

It will produce 4 files in `out/test/`: 

test_predlocs.txt -- predicted locations  
test_history.txt -- training history  
test_params.json -- run parameters   
test_fitplot.pdf -- plot of training history   

See all parameters with `python scripts/locator.py --h`

## Uncertainty and Windowed Analysis
Generating multiple predictions by fitting separate models to windows across the genome allows estimates of uncertainty and intragenomic variation for an individual-level prediction. Using the `--windows` option will generate separate predictions for nonoverlapping windows of size `--window_size` (default 500,000bp).  

This option requires zarr input for fast chunked array access. We provide a wrapper function for scikit-allel's vcf_to_zarr() function in scripts/vcf_to_zarr.py.   

Convert the test data to zarr format and run a windowed analysis with:

```
python scripts/vcf_to_zarr.py --vcf data/test_genotypes.vcf.gz --zarr data/test_genotypes.zarr
mkdir out/test_windows/
locator.py --zarr data/test_genotypes.zarr --sample_data data/test_sample_data.txt --out out/test_windows/ --windows --window_size 250000
```
This should take around 5 minutes on a GPU. For analyses in humans, mosquitoes, and malaria parasites described in our paper, we used window sizes yielding 100,000-200,000 SNPs. 

Alternately, you run windowed analyses by subsetting a set of VCFs with tabix. We used this code to run windowed analyses across a set of Anopheles VCFs:
```
step=2000000
for chr in {2L,2R,3L,3R,X}
do
	echo "starting chromosome $chr"
	#get chromosome length
	header=`tabix -H /home/data_share/ag1000/phase1/ag1000g.phase1.ar3.pass.biallelic.$chr\.vcf.gz | grep "##contig=<ID=$chr,length="`
	length=`echo $header | awk '{sub(/.*=/,"");sub(/>/,"");print}'` 
	
	#subset vcf by region and run locator
	endwindow=$step
	for startwindow in `seq 1 $step $length`
	do 
		echo "processing $startwindow to $endwindow"
		tabix -h /home/data_share/ag1000/phase1/ag1000g.phase1.ar3.pass.biallelic.$chr\.vcf.gz \
		$chr\:$startwindow\-$endwindow > data/ag1000g/tmp.vcf
		
		python scripts/locator.py \
		--vcf data/ag1000g/tmp.vcf \
		--sample_data data/ag1000g/ag1000g.phase1.samples.locsplit.txt \
		--out out/ag1000g/$chr\_$startwindow\_$endwindow
		
		endwindow=$((endwindow+step))
		rm data/ag1000g/tmp.vcf
	done
done
```

## Bootstraps
You can also train replicate models on bootstrap samples of the full VCF (sampling SNPs with replacement) with the 
`--bootstrap` argument. To fit 5 bootstrap replicates, run:
```
mkdir out/bootstrap
locator.py --vcf data/test_genotypes.vcf.gz --sample_data data/test_sample_data.txt --out out/bootstrap/test --bootstrap --nboots 5
```
This is slow (you're fitting new models to each replicate), but should give a good idea of uncertainty in predicted locations. 

## Jacknife
Last, a quicker and probably worse estimate of uncertainty can also be generated by the `--jacknife` option. This uses a single trained model and generates predictions while treating a random 5% of sites as missing data. We recommend running bootstraps for "final" predictions instead, but for a quick look at uncertainty you can run jacknife samples with:
```
mkdir out/jacknife
locator.py --vcf data/test_genotypes.vcf.gz --sample_data data/test_sample_data.txt --out out/jacknife/test --jacknife --nboots 20
```

# Plotting and summarizing output
plot_locator.R is a command line script that calculates centroids from multiple locator predictions, estimates errors (if true locations for all samples are provided) and plots maps of locator output. It is intended for runs with multiple outputs (either windowed analyses, bootstraps, or jacknife replicates). Install the required packages by running 

```Rscript scripts/install_R_packages.R```

Calculate centroids and plot predictions for our jacknife predictions with:

```
Rscript scripts/plot_locator.R --infile out/test_windows/ --sample_data data/test_sample_data.txt --out out/test_ --map F
```

This will plot predictions and uncertainties for 9 randomly selected individuals to `/out/jacknife/test_windows.png`, and print the locations with peak kernal density ("kd_x/y") and the geographic centroids ("gc_x/y") across jacknife replicates to `out/jacknife/test_centroids.txt`. You can also calculate and plot validation error estimates by using the `--error` option if you provide a sample data file with true locations for all individuals. See all parameters with 

```
Rscript scripts/plot_locator.R --help
```

# Iterating over training/validation samples
For datasets with relatively few individuals, randomly splitting training and validation samples can sometimes result in different inferences across runs. One way to deal with this variation is to train replicate models while using different sets of samples for training and validation (by changing the random seed), then estimating final locations as the centroid across multiple predictions:
```
#loop over seeds and generate five predictions using different sets of training and validation samples
cd ~/locator
for i in {12345,54321,2349560,549657840,48576593}
do
  locator.py --vcf data/test_genotypes.vcf.gz --sample_data data/test_sample_data.txt --out out/test/test_seed_$i --seed $i
done

#generate plots and centroid estimates
Rscript scripts/plot_locator.R --infile out/test/ --sample_data data/test_sample_data.txt --out out/test/test --map F

```
The first command will train five separate `locator` models and generate predictions for the unknown individuals, and the second will calculate centroids (in `out/test/test_centroids.txt `and generate a plot showing the spread of predicted locations (`out/test/test_windows.png`). In the `test_centroids` file, the "kd_x/y" columns give the location with highest kernal density, while the "gc_x/y" columns give the geographic centroids. See the preprint for details. 

# Diagnosing Failures
We recommend all users read the paper (https://elifesciences.org/articles/54507) before using Locator to get an idea of when and how it can fail. In general, location prediction works better in populations with less dispersal and datasets with more SNPs. When run on populations with too much dispersal or too little data, Locator tends to predict the middle of the distribution of training points. This behavior can also occur when a species is strongly structured in only one direction -- for example, if there is a strong north-south cline in allele frequencies but no east-west variation, Locator will typically generate accurate latitude predictions but will guess the middle of the longitudinal range of training points. 

The best way to diagnose these failures is to note the validation performance statistics printed to screen at the end of each Locator training run: 
```
predicting locations...
R2(x)=0.9484760204379148
R2(y)=0.9596984359743175
mean validation error 3.7585447303960313
median validation error 3.3019781150072984

run time 0.6170202493667603 minutes
```
These values describe the correlation between predicted and true locations in each dimension for the set of validation samples used during model training. If one or both of the R^2 numbers is low, expect predictions on that dimension to collapse towards the mean. In our tests, error on the test set is typically very similar to that on the validation set, so the validation errors printed here should also give you a rough estimate of how far off predictions should be in your dataset. 


# License

This software is available free for all non-commercial use under the non-profit open software license v 3.0 (see LICENSE.txt).






