#slim_locator_plots
setwd("~/locator/")
library(data.table);library(ggplot2);library(magrittr);library(sp);library(rgeos);library(cowplot)
theme_set(theme_classic()+theme(title=element_text(size=8),
                                axis.text=element_text(size=7),
                                axis.title=element_text(size=8),
                                legend.title=element_text(size=8),
                                legend.text=element_text(size=7),
                                strip.background = element_blank(),
                                strip.text=element_text(size=8)))

hist <- fread("out/sigma_0.45_dense5_100000_1005_history.txt")
hist$epoch <- 1:nrow(hist)
ggplot(data=hist[5:nrow(hist),],aes(x=epoch,y=val_loss))+
  geom_line(lwd=0.3)+
  scale_y_log10()


#################### single panel map plot ###################
infile <- "out/sigma_0.63_dense5_100000_4957_predlocs.txt"
locs <- fread("data/sigma_0.65_rand500_samples.txt",data.table = F)

pred <- fread(infile,data.table = F)[-1,-1]
colnames(pred)[ncol(pred)] <- "sampleID"

xcoords <- as.matrix(pred[,seq(1,ncol(pred)-1,2)])
ycoords <- as.matrix(pred[,seq(2,ncol(pred)-1,2)])
b <- data.frame(matrix(ncol=2,nrow=0))
for(i in 1:nrow(xcoords)){
  meanx <- mean(xcoords[i,])
  meany <- mean(ycoords[i,])
  b[i,] <- c(meanx,meany)
}
names(b) <- c("x","y")
b$sampleID <- pred$sampleID

pd <- merge(locs,b,by="sampleID")
names(b) <- names(locs)
b$type <- "predicted"
locs$type <- "true"
lpd <- rbind(locs,b)

tlocs=as.matrix(pd[,c("longitude","latitude")])
plocs=as.matrix(pd[,c("x","y")])
dists=sapply(1:nrow(plocs),function(e) spDistsN1(t(as.matrix(plocs[e,])),
                                                 t(as.matrix(tlocs[e,]))))
mean_dist=mean(dists,na.rm=T)
median_dist=median(dists,na.rm=T)

p1 <- ggplot(data=pd)+
  ggtitle("neighborhood size = 25\n100000 SNPs")+
  theme(axis.title=element_blank(),
        legend.position = "bottom",
        title=element_text(size=7),
        legend.box.margin = margin(-15,0,0,0),
        legend.background = element_blank())+
  coord_equal()+
  scale_shape_manual(values=c(16,21))+
  geom_point(data=subset(lpd,!sampleID %in% pred$sampleID),aes(x=longitude,y=latitude),
             col="grey",size=1,shape=1)+
  geom_segment(aes(x=x,y=y,xend=longitude,yend=latitude),lwd=0.25)+
  geom_point(data=subset(lpd,sampleID %in% pred$sampleID),
             aes(x=longitude,y=latitude,shape=type),size=1,fill="white",col="black")
p1 <- p1+guides(shape=guide_legend(override.aes = list(size=4)))
p2 <- ggplot(data=data.frame(dists/50),aes(x=dists))+
  geom_density(fill="grey",color=NA)+
  geom_vline(aes(xintercept=median(dists)))+
  theme(axis.title.y=element_blank(),axis.text.y=element_blank(),axis.ticks.y = element_blank())+
  xlab("Error (% landscape width)")

pdf("fig/slim_map_sigma063_100KSNPs.pdf",useDingbats = F,width=3.5,height = 3.5)
print(p1)
# ggdraw()+
#   draw_plot(p1,0,.3,1,.7)+
#   draw_plot(p2,0.05,0,.8,.3)
dev.off()

print(median_dist)/50


######################### multipanel map plot ##########################
infiles <- c("out/sigma_0.45_dense5_100_3745_predlocs.txt",
             "out/sigma_0.45_dense5_1000_3745_predlocs.txt",
             "out/sigma_0.45_dense5_10000_3745_predlocs.txt",
             "out/sigma_0.45_dense5_100000_3745_predlocs.txt",
             "out/sigma_0.45_dense5_300000_3745_predlocs.txt")
pd2 <- data.frame(sampleID=NA,longitude=NA,latitude=NA,x=NA,y=NA,params=NA)[0,]
lpd2 <- data.frame(longitude=NA,latitude=NA,sampleID=NA,type=NA,params=NA)[0,]
history <- hist <- fread(gsub("_predlocs.txt","_history.txt",infiles[1]))
history$epoch <- NA;history$infile <- NA
history <- history[0,]
for(infile in infiles){
  pred <- fread(infile,data.table = F)[-1,-1]
  colnames(pred)[ncol(pred)] <- "sampleID"
  
  xcoords <- as.matrix(pred[,seq(1,ncol(pred)-1,2)])
  ycoords <- as.matrix(pred[,seq(2,ncol(pred)-1,2)])
  b <- data.frame(matrix(ncol=2,nrow=0))
  for(i in 1:nrow(xcoords)){
    meanx <- mean(xcoords[i,])
    meany <- mean(ycoords[i,])
    b[i,] <- c(meanx,meany)
  }
  names(b) <- c("x","y")
  b$sampleID <- pred$sampleID
  
  locs <- fread("data/sigma_0.45_rand500_samples.txt",data.table = F)
  pd <- merge(locs,b,by="sampleID")
  names(b) <- names(locs)
  b$type <- "predicted"
  locs$type <- "true"
  lpd <- rbind(locs,b)
  pd$params <- infile
  lpd$params <- infile
  pd2 <- rbind(pd2,pd)
  lpd2 <- rbind(lpd2,lpd)
  
  hist <- fread(gsub("_predlocs.txt","_history.txt",infile))
  hist$epoch <- 1:nrow(hist)
  hist$infile <- infile
  history <- rbind(history,hist)
}
pd2$nsnps <- sapply(strsplit(pd2$params,"_"),function(e) paste(e[4],"SNPs"))
lpd2$nsnps <- sapply(strsplit(lpd2$params,"_"),function(e) paste(e[4],"SNPs"))
history$nsnps <- sapply(strsplit(history$infile,"_"),function(e) paste(e[4],"SNPs"))
p1 <- ggplot(data=pd2)+
  facet_wrap(~nsnps,nrow=1)+
  theme(axis.title=element_blank(),
        #legend.position = "bottom",
        title=element_text(size=7),
        legend.box.margin = margin(-15,0,0,0),
        legend.background = element_blank())+
  coord_equal()+
  scale_shape_manual(values=c(16,21),name="Location")+
  geom_point(data=subset(lpd2,!sampleID %in% pred$sampleID),aes(x=longitude,y=latitude),
             col="grey",size=1,shape=1)+
  geom_segment(aes(x=x,y=y,xend=longitude,yend=latitude),lwd=0.25)+
  geom_point(data=subset(lpd2,sampleID %in% pred$sampleID),
             aes(x=longitude,y=latitude,shape=type),size=1,fill="white",col="black")
p1 <- p1+guides(shape=guide_legend(override.aes = list(size=4)))
p2 <- ggplot(data=subset(history,epoch>5),aes(x=epoch,y=val_loss))+
  theme(axis.text.x=element_text(angle=45,hjust=1))+
  facet_wrap(~nsnps,nrow=1)+
  ylab("Mean Squared Error")+xlab("Training Epoch")+
  scale_y_log10()+
  #scale_x_log10()+
  geom_line(lwd=0.3,col="grey")+
  geom_smooth(fill=NA,col="black",lwd=0.5)

pdf("fig/slim_map_sigma045_nSNPs_2frame.pdf",useDingbats = F,width=6.5,height = 3.5)
ggdraw()+
  draw_plot(p1,0.0275,0.4,.975,.6)+
  draw_plot(p2,0,0,.85,.47)
dev.off()

######################## history plots and mean absolute error summaries  #########################
files <- list.files("out",full.names = T)
files <- grep("sigma",files,value = T)
files <- grep("predlocs",files,value = T)
locfiles <- c("data/sigma_0.45_rand500_samples.txt",
              "data/sigma_0.63_rand500_samples.txt",
              "data/sigma_1.29_rand500_samples.txt")
#f <-files[1]
for(f in files){
  sigma <- strsplit(f,"_") %>% unlist() %>% .[2] %>% as.numeric()
  network <- strsplit(f,"_") %>% unlist() %>% .[3]
  nsnps <- strsplit(f,"_") %>% unlist() %>% .[4] %>% as.numeric()
  seed <- strsplit(f,"_") %>% unlist() %>% .[5] %>% as.numeric()

  locs <- fread(locfiles[grep(as.character(sigma),locfiles)])
  pred <- fread(f,data.table = F)[-1,-1]
  colnames(pred)[ncol(pred)] <- "sampleID"
  xcoords <- as.matrix(pred[,seq(1,ncol(pred)-1,2)])
  ycoords <- as.matrix(pred[,seq(2,ncol(pred)-1,2)])
  b <- data.frame(matrix(ncol=2,nrow=0))
  for(i in 1:nrow(xcoords)){
    meanx <- mean(xcoords[i,])
    meany <- mean(ycoords[i,])
    b[i,] <- c(meanx,meany)
  }
  names(b) <- c("x","y")
  b$sampleID <- pred$sampleID
  pd <- merge(locs,b,by="sampleID")
  dists <- sapply(1:nrow(b),function(e) spDistsN1(as.matrix(pd[e,c("longitude","latitude")]),
                                         as.matrix(pd[e,c("x","y")])))
  pd$median_error <- median(dists)
  pd$sigma <- sigma
  pd$network <- network
  pd$nsnps <- nsnps
  pd$seed <- seed
  if (f==files[1]){
    pd2 <- pd
  } else {
    pd2 <- rbind(pd2,pd) 
  }
}

pd3 <- ddply(pd2,.(sigma,network,nsnps,seed),function(e) e$median_error[1])
pd3$nsnps <- format(pd3$nsnps,scientific = F)
pd3$dispersal <- paste("mean dispersal =",(pd3$sigma/50)*100,"%\nlandscape width")


pdf("fig/slim_error_summary_by_network.pdf",width=6.5,height=2.25,useDingbats = F)
ggplot(data=pd3,aes(x=nsnps,y=(V1/50)*100,col=network))+
  xlab("N SNPs")+ylab("Median Error\n(% Landscape width)")+
  theme(axis.text.x=element_text(angle=45,hjust=1))+
  scale_y_log10(limits=c(3,35))+
  scale_color_manual(values=c("orangered","steelblue3","gold2"))+
  facet_wrap(~dispersal)+
  geom_boxplot()
dev.off()

