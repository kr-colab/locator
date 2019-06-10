#locator pabu plots
library(ggplot2);library(magrittr);library(data.table);library(rgeos)
setwd("~/locator/")

theme_set(theme_classic()+theme(title=element_text(size=8),
                                axis.text=element_text(size=7),
                                axis.title=element_text(size=8),
                                legend.title=element_text(size=8),
                                legend.text=element_text(size=7),
                                strip.background = element_blank(),
                                strip.text=element_text(size=8)))

pabu2 <- fread("data/pabu_full_data.csv")
files <- list.files("out",full.names = T) %>% 
  grep("pabu_c91h60*",.,value = T) %>% 
    grep("predlocs",.,value=T)
#files <- files[2]
for(j in files){
  pred <- data.frame(fread(j))[-1,-1]
  colnames(pred)[ncol(pred)] <- "sampleID"
  if(j==files[1]){
    xcoords <- as.matrix(pred[,seq(1,ncol(pred)-1,2)])
    ycoords <- as.matrix(pred[,seq(2,ncol(pred)-1,2)])
  } else {
    xcoords <- cbind(xcoords,pred[,seq(1,ncol(pred)-1,2)])
    ycoords <- cbind(ycoords,pred[,seq(2,ncol(pred)-1,2)])
  }
}
xcoords <- as.matrix(xcoords)
ycoords <- as.matrix(ycoords)
colnames(pred)[ncol(pred)] <- "sampleID"
#best_pred <- kdepred(xcoords,ycoords)
best_pred <- data.frame(meanx=NA,meany=NA)[-1,]
for(i in 1:nrow(xcoords)){
  pts <- SpatialPoints(coords=as.matrix(data.frame(xcoords[i,],ycoords[i,])))
  center <- as.data.frame(gCentroid(pts))
  best_pred[i,] <- center
}
best_pred <- cbind(best_pred,pred[,ncol(pred)])
names(best_pred) <- c("long","lat","sampleID")
pd <- merge(pabu2,best_pred,by="sampleID")

map <- map_data("world")
map2 <- map_data("state")
p <- ggplot()+coord_map()+
  theme(axis.title.x = element_blank(),
        axis.title.y=element_blank(),
        legend.position=c(0.15,0.25),
        legend.background = element_blank())+
  scale_color_manual(values=c("orangered","steelblue3"),name="")+
  xlim(-112,-75)+ylim(12,38)+
  geom_path(data=map2,aes(x=long,y=lat,group=group),col="grey",lwd=0.2)+
  geom_path(data=map,aes(x=long,y=lat,group=group),col="black",lwd=0.2)+
  geom_segment(data=pd,aes(x=Longitude,xend=long,y=Latitude,yend=lat),lwd=0.2,col="grey")+
  geom_point(data=best_pred,aes(x=long,y=lat),shape=1,size=1)+
  geom_point(data=pabu2,aes(x=Longitude,y=Latitude,col=season),shape=1,size=3)

pdf("fig/pabu_map.pdf",width=5,height=3.5,useDingbats = F)
print(p)
dev.off()
print(p)

#undertainty plots
plot_uncertainty <- function(sampleID=NULL){
  if(is.null(sampleID)){
    ind <- sample(pred$sampleID,1,T)
  }
  xcoords2 <- xcoords[pred$sampleID %in% ind,]
  ycoords2 <- ycoords[pred$sampleID %in% ind,]
  indpreds <- data.frame(xcoords2,ycoords2,ind)
  
  map <- map_data("state")
  p <- ggplot()+
    coord_map(xlim = c(min(indpreds$xcoords2)-5,max(indpreds$xcoords2)+5),
              ylim = c(min(indpreds$ycoords2)-5,max(indpreds$ycoords2)+5))+
    theme(axis.title=element_blank())+
    ggtitle(ind)+
    geom_path(data=map,aes(x=long,y=lat,group=group),lwd=0.5,color="black")+
    scale_fill_viridis_c(name="Density",option = "B",guide=F)+
    stat_density_2d(data=indpreds,aes(x=xcoords2,y=ycoords2,fill = ..level..),
                    geom="polygon",alpha=0.5,n = 200)+
    geom_point(data=best_pred[best_pred$sampleID==ind,],aes(x=long,y=lat),
               fill="white",stroke=0.1,shape=21,color="black",size=2)
  return(p)
}
map <- map_data("state")
p1 <- plot_uncertainty()
p2 <- plot_uncertainty()
p3 <- plot_uncertainty()
p4 <- plot_uncertainty()
p5 <- plot_uncertainty()
p6 <- plot_uncertainty()
p7 <- plot_uncertainty()
p8 <- plot_uncertainty()
p9 <- plot_uncertainty()

pdf("fig/pabu_uncertainty_grid.pdf",width=6,height=6,useDingbats = F)
plot_grid(p1,p2,p3,p4,p5,p6,p7,p8,p9)
dev.off()




