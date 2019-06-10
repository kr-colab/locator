#process model predictions for dropout setting performance
library(data.table);library(plyr);library(magrittr);library(ggplot2);library(progress)
setwd("~/locator/out/")
files <- list.files()
#files <- files[grepl("2L13|2L15|2L010|2L14100K",files)]
files <- files[!grepl("LS",files)]
files <- files[!grepl("ag1000g",files)]
files <- files[grepl("anopheles",files)]

kdepred <- function(xcoords,ycoords){
  long <- c();lat <- c()
  for(i in 1:nrow(xcoords)){ #run 2d kernel density estimate and assign predicted locations to site with highest density
    if(length(unique(xcoords[i,]))>10 & length(unique(ycoords[i,]))>10){ #prevent errors when predictions have near 0 spread (usually low dropout...)
      density <- kde2d(xcoords[i,],ycoords[i,],n=100,
                       lims = c(-21,51,-18,19))
      max_index <- which(density[[3]] == max(density[[3]]), arr.ind = TRUE)
      long <- append(long,density[[1]][max_index[1]])
      lat <- append(lat,density[[2]][max_index[2]]) 
    } else{
      long <- append(long,mean(xcoords[i,]))
      lat <- append(lat,mean(ycoords[i,]))
    }
  }
  return(data.frame(long,lat))
}

locs <- fread("~/locator/data/anopheles_samples_sp.txt")
pb <- progress_bar$new(total = length(files))
for(j in files){
  pred <- data.frame(fread(j))[-1,-1]
  colnames(pred)[ncol(pred)] <- "sampleID"
  xcoords <- as.matrix(pred[,seq(1,ncol(pred)-1,2)])
  ycoords <- as.matrix(pred[,seq(2,ncol(pred)-1,2)])
  if(grepl("dense4",j)){
      best_pred <- data.frame(xcoords[,1],ycoords[,1])
  } else {
      best_pred <- kdepred(xcoords,ycoords)
    }
  best_pred <- cbind(best_pred,pred[,ncol(pred)])
  colnames(best_pred) <- c("longitude","latitude","sampleID")
  pd <- merge(best_pred,locs,by="sampleID",all=T)
  plocs=as.matrix(pd[!is.na(pd$longitude.x),c("longitude.x","latitude.x")])
  tlocs=as.matrix(pd[!is.na(pd$longitude.x),c("longitude.y","latitude.y")])
  dists=sapply(1:nrow(plocs),function(e) spDistsN1(t(as.matrix(plocs[e,])),
                                                   t(as.matrix(tlocs[e,])),longlat = T))
  mean_dist=mean(dists,na.rm=T)
  median_dist=median(dists,na.rm=T)
  
  tmp <- strsplit(j,"_")[[1]]
  model <- tmp[3]
  dataset <- tmp[2]
  dropout <- tmp[4]
  run <- tmp[5]
  ls <- tmp[6]
  
  if(j==files[1]){
    out <- data.frame(mean_dist,median_dist,model,dataset,dropout,run,ls)
  } else {
    out <- rbind(out,data.frame(mean_dist,median_dist,model,dataset,dropout,run,ls))
  }
  pb$tick()
}
out$model <- factor(out$model,levels=c("dense1","dense2","dense3","dense4","dense5","dense6"))
out$dropout <- factor(as.numeric(as.character(out$dropout)))
out2 <- subset(out,dataset %in% c("2L13","2L15"))
out2$dataset <- mapvalues(out2$dataset,from=c("2L13","2L15"),to=c("30,000 SNPs","300,000 SNPs"))
#out2 <- out
#out$dropout <- as.character(out$dropout)
#out$dataset <- factor(out$dataset,levels=c("2L13","2L15","2L010","2L13.5","2L14100K"))
#out$dataset <- mapvalues(out$dataset,from=levels(out$dataset),to=c("30kSNPs","300kSNPs",
#                                                                   "900kSNPs","100KSNPs",
#                                                                   "100KSNPs_shuffled"))
#out$dropout <- factor(out$dropout,levels=c(0.5,0.25,0.1))

p <- ggplot(data=out2,aes(x=model,fill=dropout,y=median_dist))+
  facet_wrap(~dataset)+
  theme(panel.grid.major=element_line(size=0.2,color="grey"),
        axis.text=element_text(size=7),
        axis.text.x=element_text(size=7,angle=45,hjust=1))+
  scale_fill_manual(values=c("steelblue3","orangered","gold2"))+
  ylab("Median Error (km)")+xlab("Network")+
  #scale_y_log10(limits=c(10,1200))+
  ylim(0,600)+
  geom_point(shape=21,stroke=0.1,position=position_jitter(width=0.25),size=2,alpha=0.75)
  #geom_smooth()
  #geom_boxplot(lwd=0.1,outlier.size=.5)
  #geom_violin()

pdf("~/locator/fig/anopheles_network_comparison",width=5,height=2.5,useDingbats = F)
print(p)
dev.off()

