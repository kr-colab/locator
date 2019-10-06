#plot output for one individual from a Locator run
require(data.table);require(scales);require(raster)
require(sp);require(MASS);require(rgeos);require(plyr)
require(progress);require(argparse);require(ggplot2)

parser <- argparse::ArgumentParser()
parser$add_argument('--infile')
parser$add_argument('--sample_data')
parser$add_argument('--out')
parser$add_argument('--width',default=5,type="double")
parser$add_argument('--height',default=4,type="double")
parser$add_argument('--samples',default=NULL)
parser$add_argument('--nsamples',default=3)
parser$add_argument('--ncol',default=3,type="integer")
parser$add_argument('--error',default="F")
parser$add_argument('--legend_position',default="bottom")
args <- parser$parse_args()

infile <- args$infile
sample_data <- args$sample_data
out <- args$out
width <- args$width
height <- args$height
ncol <- args$ncol
dropout <- args$dropout
error <- args$error
# indir <- "~/locator/out/ag1000g/windows"
# sample_data <- "~/locator/data/ag1000g/anopheles_samples_sp.txt"
# out <- "~/Desktop/testmap"
# width <- 5
# height <- 4
# samples <- c("AN0007-C","AN0014-C","AB0190-C","AB0211-C","AB0206-C","AJ0088-C")
# infile <- "~/Desktop/droptest_predlocs.txt"
# AN0007-C,AN0014-C,AB0190-C,AB0211-C,AB0206-C,AJ0088-C


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
  names(pd) <- c('xpred','ypred','sampleID','prediction')
} else {
  files <- list.files(infile,full.names = T)
  files <- grep("predlocs",files,value=T)
  pd <- fread(files[1],data.table=F)[0,]
  for(f in files){
    a <- fread(f,data.table = F)#[-1,-1]
    pd <- rbind(pd,a)
  }
  names(pd) <- c('xpred','ypred','sampleID')
}


if(!is.null(args$samples) && grepl(",",args$samples)){
  samples <- unlist(strsplit(args$samples,","))
} else if(is.null(args$samples)){
  samples <- sample(unique(pd$sampleID),args$nsamples,replace = F)
} else {
  samples <- args$samples
}

locs <- fread(sample_data,data.table=F)
pd <- merge(pd,locs,by="sampleID")

if(error!="F"){
  print("calculating error")
  #get error for centroids and max kernel density locations
  bp <- ddply(pd,.(sampleID),function(e) {
    k <- kdepred(e$xpred,e$ypred)
    g <- as.data.frame(gCentroid(SpatialPoints(as.matrix(e[,c("xpred","ypred")]))))
    out <- unlist(c(g,k))
    names(out) <- c("gc_x","gc_y","kd_x","kd_y")
    return(out)
  })
  pd <- merge(pd,bp,by="sampleID")
  plocs=as.matrix(pd[,c("kd_x","kd_y")])
  tlocs=as.matrix(pd[,c("longitude","latitude")])
  dists=sapply(1:nrow(plocs),function(e) spDistsN1(t(as.matrix(plocs[e,])),
                                                   t(as.matrix(tlocs[e,])),longlat = T))
  pd$dist_kd <- dists
  print(paste("mean kernel peak error =",mean(dists)))
  print(paste("median kernel peak error =",median(dists)))
  print(paste("90% CI for kernal peak error = ",quantile(dists,0.05),quantile(dists,0.95)))
  
  plocs=as.matrix(pd[,c("gc_x","gc_y")])
  tlocs=as.matrix(pd[,c("longitude","latitude")])
  dists=sapply(1:nrow(plocs),function(e) spDistsN1(t(as.matrix(plocs[e,])),
                                                   t(as.matrix(tlocs[e,])),longlat = T))
  pd$dist_gc <- dists
  print(paste("mean centroid error =",mean(dists)))
  print(paste("median centroid error ",median(dists)))
  print(paste("90% CI for centroid error = ",quantile(dists,0.05),quantile(dists,0.95)))
  
}



load("~/locator/data/cntrymap.Rdata")

print("plotting")
pb <- progress_bar$new(total=length(samples))
png(paste0(out,"_windows.png"),width=width,height=height,res = 600,units = "in")
par(oma=c(0,0,0,0),mai=c(.15,.15,.15,.15),mgp=c(3,0.15,0))
if(length(samples)==2){
  layout(mat=matrix(c(1,2,3,3),byrow=T,nrow=2),heights = c(1,.5))
} else if(length(samples)>=3){
  layout(mat=matrix(c(1:length(samples),rep(length(samples)+1,ncol)),
                    byrow=T,nrow=ceiling(length(samples)/ncol)+1),
         heights = c(rep(1,ceiling(length(samples)/ncol)),.5))
}
for(i in samples){
  sample <- subset(pd,sampleID==i)
  plot(map,axes=T,cex.axis=0.5,tck=-0.03,
       xlim=c(min(na.omit(c(sample$xpred,sample$longitude)))-6,
              max(na.omit(c(sample$xpred,sample$longitude)))+6),
       ylim=c(min(na.omit(c(sample$ypred,sample$latitude)))-6,
              max(na.omit(c(sample$ypred,sample$latitude)))+6),
       col="grey",border="white",lwd=0.35)
  title(sample$sampleID[1],cex.main=0.75,font.main=1)
  box(lwd=1)
  pts <- SpatialPoints(as.matrix(data.frame(sample$xpred,sample$ypred)))
  try({
    kd <- kde2d(sample$xpred,sample$ypred,n = 80,
                lims = c(min(na.omit(c(sample$xpred,sample$longitude)))-15,
                         max(na.omit(c(sample$xpred,sample$longitude)))+15,
                         min(na.omit(c(sample$ypred,sample$latitude))-15),
                         max(na.omit(c(sample$ypred,sample$latitude)))+15))
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
  points(x=locs$longitude,y=locs$latitude,col="dodgerblue3",pch=16,cex=0.5,lwd=0.2)
  points(pts,pch=16,cex=0.45,col=alpha("black",0.7))
  try({
    contour(kd,levels=levels,drawlabels=T,labels=prob,add=T,
            labcex=0.3,lwd=0.5,axes=True,vfont=c("sans serif","bold"))
  },silent=TRUE)
  points(x=sample$longitude[1],y=sample$latitude[1],col="red3",pch=1,cex=.8)
  points(pts[grepl("FULL",files)],col="forestgreen",pch=1,cex=.8)
  #pb$tick()
}
plot(1, type = "n", axes=FALSE, xlab="", ylab="")
legend(x="top",
       legend=c("Training Locations","Sample Location","Predicted Locations"),
       col=c("dodgerblue3","red3","black"),
       pch=16,cex=.7,pt.cex=2.25,bty='n',horiz=T)
dev.off()


truelocs <- ddply(pd,.(longitude,latitude),summarize,error=mean(dist_gc))
locsn <- ddply(locs,.(longitude,latitude),summarize,n=length(sampleID))
truelocs <- merge(truelocs,locsn,c("longitude","latitude"))

pdf(paste0(out,"_summary.pdf"),width=6,height=3.25,useDingbats = F)
map <- crop(map,c(min(na.omit(c(pd$xpred,pd$longitude)))-10,
                  max(na.omit(c(pd$xpred,pd$longitude)))+10,
                  min(na.omit(c(pd$ypred,pd$latitude)))-10,
                  max(na.omit(c(pd$ypred,pd$latitude)))+10))
ggplot()+coord_map(projection = "mollweide",
                   xlim=c(min(na.omit(c(pd$xpred,pd$longitude)))-10,
                          max(na.omit(c(pd$xpred,pd$longitude)))+10),
                   ylim=c(min(na.omit(c(pd$ypred,pd$latitude)))-10,
                          max(na.omit(c(pd$ypred,pd$latitude)))+10))+
  theme_classic()+theme(axis.title = element_blank(),
                        legend.title = element_text(size=8),
                        legend.text=element_text(size=6),
                        axis.text=element_text(size=6),
                       # legend.box = "horizontal",
                        legend.position = args$legend_position)+
  scale_color_distiller(palette = "RdYlBu",name="Mean Error\n(km)")+
  scale_size_continuous(name="Training\nSamples")+
  geom_polygon(data=fortify(map),aes(x=long,y=lat,group=group),fill="grey",color="white",lwd=0.2)+
  geom_point(data=truelocs,aes(x=longitude,y=latitude,color=error,size=n))+
  geom_segment(data=pd,aes(x=longitude,y=latitude,xend=gc_x,yend=gc_y),lwd=0.2)+
  geom_point(data=pd,aes(x=gc_x,y=gc_y),size=0.5,shape=1)
dev.off()
  
#pd$dist <- apply(pd[,2:5],1,function(e) spDistsN1(matrix(e[1:2],ncol=2),matrix(e[3:4],ncol=2),longlat = TRUE))
pdf(paste0(out,"_error_histogram.pdf"),width=3,height=2.5)
ggplot(data=pd,aes(x=dist_gc))+
  theme_classic()+theme(axis.text=element_text(size=6),axis.title=element_text(size=8))+
  xlab("Test Error (km)")+ylab("n samples")+
  #scale_x_log10()+
  geom_histogram()
dev.off()
  
  