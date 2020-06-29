`--infile` : path to directory with .predlocs files for plotting

`--sample_data` : path to tab-delimited sample metadata file, formatted `sampleID x y`. if `map == 'T'`, should be WGS1984 x / y. if `--error == True`, must contain locations of all samples

`--out` : path to output directory (will be appended with filenames)

`--width` : width in inches; height scales with map size

`--sample` : sample ID to plot. if none is provided, a random sample will be plotted

`--error` : calculate and plot kernel density. requires sample metadata to include locations of all samples

`--map` : plot on world map


`python plot_locator.py --infile predlocs --sample_data metadata.txt --out out --sample 'sampleID' --error True`
