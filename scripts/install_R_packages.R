installed <- rownames(installed.packages())
required <- c("data.table","scales","raster","sp","MASS","rgeos","plyr","progress","argparse","ggplot2")
needed <- required[!required %in% installed]
for(i in needed) install.packages(i,repos='http://cran.us.r-project.org')