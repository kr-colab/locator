#plot output for one individual from a Locator run
suppressMessages(suppressWarnings(require(data.table)))
suppressMessages(suppressWarnings(require(scales)))
suppressMessages(suppressWarnings(require(raster)))
suppressMessages(suppressWarnings(require(sp)))
suppressMessages(suppressWarnings(require(MASS)))
suppressMessages(suppressWarnings(require(rgeos)))
suppressMessages(suppressWarnings(require(plyr)))
suppressMessages(suppressWarnings(require(progress)))
suppressMessages(suppressWarnings(require(argparse)))
suppressMessages(suppressWarnings(require(ggplot2)))

parser <- argparse::ArgumentParser(description="Plot summary of a set of locator predictions.")
parser$add_argument('--infile',help="path to folder with .predlocs files")
parser$add_argument('--sample_data',help="path to sample_data file (should be WGS1984 x / y if map=TRUE.")
parser$add_argument('--out',help="path to output (will be appended with _typeofplot.pdf)")
parser$add_argument('--width',default=5,type="double",help="width in inches of the output map. default = 5")
parser$add_argument('--height',default=4,type="double",help="height in inches of the output map. default = 4")
parser$add_argument('--samples',default=NULL,type="character",help="samples IDs to plot, separated by commas. e.g. sample1,sample2,sample3. No spaces. default = NULL")
parser$add_argument('--nsamples',default=9,help="if no --samples argument is provided, --nsamples random samples will be plotted. default = 9")
parser$add_argument('--ncol',default=3,type="integer",help="number of columns for multipanel plots (should evenly divide --nsamples). default = 3")
parser$add_argument('--error',default=FALSE,action="store_true",help="calculate error and plot summary? requires known locations for all samples. T / F. default = F")
parser$add_argument('--legend_position',default="bottom",help="legend position for summary plots if --error is True. Options:'bottom','right'. default = bottom")
parser$add_argument('--map',default="T",type="character",help="plot basemap? default = T")
parser$add_argument('--longlat',default=FALSE,action="store_true",help="set to TRUE if coordinates are x and y in decimal degrees for error in kilometers. default: FALSE. ")
parser$add_argument('--haploid',default=FALSE,action="store_true",help="set to TRUE if predictions are from locator_phased.py. Predictions will be plotted for each haploid chromosome separately. default: FALSE.")
parser$add_argument('--centroid_method',default="kd",help="Method for summarizing window/bootstrap predictions. Options 'gc' (take the centroid of window predictions with rgeos::gCentroid() ) or 'kd' (take the location of maximum density after kernal density estimation with mass::kde( )). default: kd")
args <- parser$parse_args()

infile <- args$infile
sample_data <- args$sample_data
out <- args$out
width <- args$width
height <- args$height
ncol <- args$ncol
dropout <- args$dropout
error <- args$error
samples <- args$samples
usemap <- args$map
haploid <- args$haploid
nsamples <- args$nsamples
centroid_method <- args$centroid_method

# infile <- "~/Downloads/locator/bootstraps/"
# sample_data <- "~/Downloads/locator/city_2.txt"
# out <- "~/Desktop/locator_plot_test"
# width <- 5
# height <- 4
# samples <- NULL
# nsamples<- 9
# ncol <- 3
# usemap <- T
# haploid <- F

# load("~/locator/data/cntrymap.Rdata")

kdepred <- function(xcoords,ycoords){
  try({
    density <- kde2d(xcoords,ycoords,n=500)
    max_index <- which(density[[3]] == max(density[[3]]), arr.ind = TRUE)
    kd_x <- density[[1]][max_index[1]]
    kd_y  <- density[[2]][max_index[2]]
    return(data.frame(kd_x,kd_y))
  },{
    kd_x <- mean(xcoords)
    kd_y <- mean(ycoords)
    return(data.frame(kd_x,kd_y))
  })
}

print("loading data")
if(grepl("predlocs.txt",infile)){
  pd <- fread(infile,data.table=F)
  names(pd) <- c('xpred','ypred','sampleID')
  files <- infile
} else {
  files <- list.files(infile,full.names = T)
  files <- grep("predlocs",files,value=T)
  pd <- fread(files[1],data.table=F)[0,1:3]
  for(f in files){
    a <- fread(f,data.table = F,header=T)[,1:3]
    pd <- rbind(pd,a)
  }
  names(pd) <- c('xpred','ypred','sampleID')
}

locs <- fread(sample_data,data.table=F)

if(!is.null(samples) && grepl(",",samples)){
  samples <- unlist(strsplit(samples,","))
} else if(is.null(samples)){
  samples <- sample(unique(pd$sampleID),nsamples,replace = F)
} else {
  samples <- args$samples
}

pd <- merge(pd,locs,by="sampleID")

if(error){
  print("calculating error")
  #get error for centroids and max kernel density locations
  bp <- ddply(pd,.(sampleID),function(e) {
    k <- kdepred(e$xpred,e$ypred)
    g <- as.data.frame(gCentroid(SpatialPoints(as.matrix(e[,c("xpred","ypred")]),proj4string = crs(proj4string(map)))))
    out <- unlist(c(g,k))
    names(out) <- c("gc_x","gc_y","kd_x","kd_y")
    return(out)
  })
  
  pd <- merge(pd,bp,by="sampleID")
  outsum <- pd[,c("sampleID","kd_x","kd_y","gc_x","gc_y")]
  outsum <- ddply(outsum,.(sampleID),function(e) e[1,])
  write.table(outsum,paste0(out,"_centroids.txt"),sep="\t",row.names=FALSE)
  
  plocs=as.matrix(pd[,c("kd_x","kd_y")])
  tlocs=as.matrix(pd[,c("x","y")])
  dists=sapply(1:nrow(plocs),function(e) spDistsN1(t(as.matrix(plocs[e,])),
                                                   t(as.matrix(tlocs[e,])),longlat = args$longlat))
  pd$dist_kd <- dists
  print(paste("mean kernel peak error =",mean(dists)))
  print(paste("median kernel peak error =",median(dists)))
  print(paste("90% CI for kernal peak error = ",quantile(dists,0.05),quantile(dists,0.95)))
  
  plocs=as.matrix(pd[,c("gc_x","gc_y")])
  tlocs=as.matrix(pd[,c("x","y")])
  dists=sapply(1:nrow(plocs),function(e) spDistsN1(t(as.matrix(plocs[e,])),
                                                   t(as.matrix(tlocs[e,])),longlat = args$longlat))
  pd$dist_gc <- dists
  print(paste("mean centroid error =",mean(dists)))
  print(paste("median centroid error ",median(dists)))
  print(paste("90% CI for centroid error = ",quantile(dists,0.05),quantile(dists,0.95)))
}


print("plotting")
pb <- progress_bar$new(total=length(samples))
png(paste0(out,"_windows.png"),width=width,height=height,res = 600,units = "in")
par(oma=c(0,0,0,0),mai=c(.15,.15,.15,.15),mgp=c(3,0.15,0))
if(length(samples)==1){
  layout(mat=matrix(c(1,2),byrow=T,nrow=2),heights = c(1,.5))
}
if(length(samples)==2){
  layout(mat=matrix(c(1,2,3,3),byrow=T,nrow=2),heights = c(1,.5))
} else if(length(samples)>=3){
  layout(mat=matrix(c(1:length(samples),rep(length(samples)+1,ncol)),
                    byrow=T,nrow=ceiling(length(samples)/ncol)+1),
         heights = c(rep(1,ceiling(length(samples)/ncol)),.5))
}
for(i in samples){
  print(i)
  sample <- pd[pd$sampleID==i,]
  if(usemap=="T"){
    plot(map,axes=T,cex.axis=0.5,tck=-0.03,border="white",
         xlim=c(min(na.omit(c(sample$xpred,sample$x)))-6,
                max(na.omit(c(sample$xpred,sample$x)))+6),
         ylim=c(min(na.omit(c(sample$ypred,sample$y)))-6,
                max(na.omit(c(sample$ypred,sample$y)))+6),
         col="grey",lwd=0.35)
  } else {
    plot(0,axes=T,cex.axis=0.5,tck=-0.03,
         xlim=c(min(na.omit(c(pd$xpred,pd$x)))-1,
                max(na.omit(c(pd$xpred,pd$x)))+1),
         ylim=c(min(na.omit(c(pd$ypred,pd$y)))-1,
                max(na.omit(c(pd$ypred,pd$y)))+1),
         col="white")
  }
  
  #title(paste(sample$population[1],sample$sampleID[1],sep=":"),cex.main=0.9,font.main=1)
  title(sample$sampleID[1],cex.main=0.8,font.main=1)
  box(lwd=1)
  pts <- SpatialPoints(as.matrix(data.frame(sample$xpred,sample$ypred)))
  try({
    kd <- kde2d(sample$xpred,sample$ypred,n = 80,
                lims = c(min(na.omit(c(sample$xpred,sample$x)))-15,
                         max(na.omit(c(sample$xpred,sample$x)))+15,
                         min(na.omit(c(sample$ypred,sample$y))-15),
                         max(na.omit(c(sample$ypred,sample$y)))+15))
    prob <- c(.95,.5,.1) #via https://stackoverflow.com/questions/16225530/contours-of-percentiles-on-level-plot
    dx <- diff(kd$x[1:2])
    dy <- diff(kd$y[1:2])
    sz <- sort(kd$z)
    c1 <- cumsum(sz) * dx * dy
    levels <- sapply(prob, function(x) {
      approx(c1, sz, xout = 1 - x)$y
    })
    levels <- levels[!is.na(levels)]
  },silent=TRUE)
  points(x=locs$x,y=locs$y,col="dodgerblue3",pch=1,cex=0.5,lwd=0.5)
  points(pts,pch=16,cex=0.4,col=alpha("black",0.7))
  try({
    contour(kd,levels=levels,drawlabels=T,labels=prob,add=T,
            labcex=0.32,lwd=0.5,axes=True,vfont=c("sans serif","bold"))
  },silent=TRUE)
  points(x=sample$x[1],y=sample$y[1],col="red3",pch=1,cex=.75)
  # if(!is.null(grep("FULL",files))){
  #   points(pts[grepl("FULL",files)],col="forestgreen",pch=1,cex=.8)
  # }
  #pb$tick()
}
plot(1, type = "n", axes=FALSE, xlab="", ylab="")
legend(x="top",
       legend=c("Training Locations","Sample Location","Predicted Locations"),
       col=c("dodgerblue3","red3","black"),
       pch=16,cex=1,pt.cex=2,bty='n',horiz=T,x.intersp = 1)
dev.off()


if(error){
  pdf(paste0(out,"_summary.pdf"),width=6,height=3.25,useDingbats = F)
  if(usemap=="T"){
    if(centroid_method=="gc"){
      truelocs <- ddply(pd,.(x,y),summarize,error=mean(dist_gc))
      locsn <- ddply(locs,.(x,y),summarize,n=length(sampleID))
      truelocs <- merge(truelocs,locsn,c("x","y"))
      map <- crop(map,c(min(na.omit(c(pd$xpred,pd$x)))-10,
                        max(na.omit(c(pd$xpred,pd$x)))+10,
                        min(na.omit(c(pd$ypred,pd$y)))-10,
                        max(na.omit(c(pd$ypred,pd$y)))+10))
      print(ggplot()+coord_map(projection = "mollweide",
                               xlim=c(min(na.omit(c(pd$xpred,pd$x)))-10,
                                      max(na.omit(c(pd$xpred,pd$x)))+10),
                               ylim=c(min(na.omit(c(pd$ypred,pd$y)))-10,
                                      max(na.omit(c(pd$ypred,pd$y)))+10))+
              theme_classic()+theme(axis.title = element_blank(),
                                    legend.title = element_text(size=8),
                                    legend.text=element_text(size=6),
                                    axis.text=element_text(size=6),
                                    # legend.box = "horizontal",
                                    legend.position = args$legend_position)+
              scale_color_distiller(palette = "RdYlBu",name="Mean Error\n(km)")+
              scale_size_continuous(name="Training\nSamples")+
              geom_polygon(data=fortify(map),aes(x=long,y=lat,group=group),fill="grey",color="white",lwd=0.2)+
              geom_point(data=truelocs,aes(x=x,y=y,color=error,size=n))+
              geom_segment(data=pd,aes(x=x,y=y,xend=gc_x,yend=gc_y),lwd=0.2)+
              geom_point(data=pd,aes(x=gc_x,y=gc_y),size=0.5,shape=1))
    } else if(centroid_method=="kd") {
      truelocs <- ddply(pd,.(x,y),summarize,error=mean(dist_kd))
      locsn <- ddply(locs,.(x,y),summarize,n=length(sampleID))
      truelocs <- merge(truelocs,locsn,c("x","y"))
      map <- crop(map,c(min(na.omit(c(pd$xpred,pd$x)))-10,
                        max(na.omit(c(pd$xpred,pd$x)))+10,
                        min(na.omit(c(pd$ypred,pd$y)))-10,
                        max(na.omit(c(pd$ypred,pd$y)))+10))
      print(ggplot()+coord_map(projection = "mollweide",
                               xlim=c(min(na.omit(c(pd$xpred,pd$x)))-10,
                                      max(na.omit(c(pd$xpred,pd$x)))+10),
                               ylim=c(min(na.omit(c(pd$ypred,pd$y)))-10,
                                      max(na.omit(c(pd$ypred,pd$y)))+10))+
              theme_classic()+theme(axis.title = element_blank(),
                                    legend.title = element_text(size=8),
                                    legend.text=element_text(size=6),
                                    axis.text=element_text(size=6),
                                    # legend.box = "horizontal",
                                    legend.position = args$legend_position)+
              scale_color_distiller(palette = "RdYlBu",name="Mean Error\n(km)")+
              scale_size_continuous(name="Training\nSamples")+
              geom_polygon(data=fortify(map),aes(x=long,y=lat,group=group),fill="grey50",color="white",lwd=0.2)+
              geom_point(data=locs,aes(x=x,y=y),shape=1,color="grey30",size=0.6,stroke=0.3)+
              geom_point(data=truelocs,aes(x=x,y=y,color=error,size=n))+
              geom_segment(data=pd,aes(x=x,y=y,xend=kd_x,yend=kd_y),lwd=0.2)+
              geom_point(data=pd,aes(x=kd_x,y=kd_y),size=0.5,shape=1))
    }
  } else {
    print(ggplot()+
            theme_classic()+theme(axis.title = element_blank(),
                                  legend.title = element_text(size=8),
                                  legend.text=element_text(size=6),
                                  axis.text=element_text(size=6),
                                  # legend.box = "horizontal",
                                  legend.position = args$legend_position)+
            scale_color_distiller(palette = "RdYlBu",name="Mean Error")+
            scale_size_continuous(name="Training\nSamples")+
            #geom_polygon(data=fortify(map),aes(x=long,y=lat,group=group),fill="grey",color="white",lwd=0.2)+
            geom_point(data=locs,aes(x=x,y=y),shape=1,color="grey50",size=0.6,stroke=0.3)+
            geom_point(data=truelocs,aes(x=x,y=y,color=error,size=n))+
            geom_segment(data=pd,aes(x=x,y=y,xend=gc_x,yend=gc_y),lwd=0.2)+
            geom_point(data=pd,aes(x=gc_x,y=gc_y),size=0.5,shape=1))
  }
 
  dev.off()
  
  #pd$dist <- apply(pd[,2:5],1,function(e) spDistsN1(matrix(e[1:2],ncol=2),matrix(e[3:4],ncol=2),longlat = TRUE))
  pdf(paste0(out,"_error_histogram.pdf"),width=3,height=2.5)
  print(ggplot(data=pd,aes(x=dist_gc))+
    theme_classic()+theme(axis.text=element_text(size=6),axis.title=element_text(size=8))+
    xlab("Test Error (km)")+ylab("n samples")+
    #scale_x_log10()+
    geom_histogram())
  dev.off()
}

  