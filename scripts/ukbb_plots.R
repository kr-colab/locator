#biobank maps
library(data.table);library(pbapply);library(magrittr);library(ggplot2)
samples <- fread("~/locator/data/ukb_100000inds_sample_data.txt")
ggplot(data=samples,aes(x=longitude,y=latitude))+stat_bin_hex()

pred <- fread("~/locator/out/ukb_100kinds_dense0_predlocs.txt",data.table = F)[-1,-1]
xcoords <- as.matrix(pred[,seq(1,ncol(pred)-1,2)])
ycoords <- as.matrix(pred[,seq(2,ncol(pred)-1,2)])
best_pred <- data.frame(meanx=NA,meany=NA)[-1,]
for(i in 1:nrow(xcoords)){
  meanx <- mean(xcoords[i,])
  meany <- mean(ycoords[i,])
  best_pred[i,] <- data.frame(meanx,meany)
}
#best_pred <- kdepred(xcoords,ycoords)
best_pred <- cbind(best_pred,pred[,ncol(pred)])
names(best_pred) <- c("long","lat","sampleID")
pd <- merge(samples,best_pred,by="sampleID")

ggplot()+
  geom_point(data=pd,aes(x=longitude,y=latitude),col="red")+
  geom_segment(data=pd,aes(x=longitude,xend=long,y=latitude,yend=lat),lwd=0.1)+
  geom_point(data=pd,aes(x=long,y=lat),col="blue")

ggplot(data=pd,aes(x=long,y=longitude))+geom_point()
ggplot(data=pd,aes(x=lat,y=latitude))+geom_point()

lm(long~longitude,data=pd) %>% summary()
lm(lat~latitude,data=pd) %>% summary()

# uncertainty plots .......
i <- 40
sampleID <- pred$sampleID[i]
colnames(pred)[ncol(pred)] <- "sampleID"
xcoords <- as.matrix(pred[,seq(1,ncol(pred)-1,2)])
ycoords <- as.matrix(pred[,seq(2,ncol(pred)-1,2)])
ind <- sampleID
indpreds <- data.frame(x=xcoords[pred$sampleID==ind,],y=ycoords[pred$sampleID==ind,])
indpreds <- SpatialPoints(as.matrix(indpreds),proj4string = CRS("+proj=tmerc +lat_0=49 +lon_0=-2 +k=0.9996012717 +x_0=400000 +y_0=-100000 +ellps=airy +towgs84=446.448,-125.157,542.06,0.15,0.247,0.842,-20.489 +units=m +no_defs"))
indpreds <- spTransform(indpreds,CRS("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"))
indpreds <- as.data.frame(indpreds)
colnames(indpreds) <- c("longitude","latitude")
x=pd[pd$sampleID==best_pred$sampleID[i],"longitude"]
y=pd[pd$sampleID==best_pred$sampleID[i],"latitude"]
trueloc <- data.frame(x,y)
trueloc <- SpatialPoints(trueloc,proj4string = CRS("+proj=tmerc +lat_0=49 +lon_0=-2 +k=0.9996012717 +x_0=400000 +y_0=-100000 +ellps=airy +towgs84=446.448,-125.157,542.06,0.15,0.247,0.842,-20.489 +units=m +no_defs"))
trueloc <- spTransform(trueloc,CRS("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"))
trueloc <- as.data.frame(trueloc)
map <- map_data("world")
ggplot()+
  coord_map(xlim = c(min(indpreds$longitude)-5,max(indpreds$longitude)+5),
            ylim = c(min(indpreds$latitude)-5,max(indpreds$latitude)+5))+
  theme_classic()+
  geom_polygon(data=map,aes(x=long,y=lat,group=group),lwd=0.5,fill="grey",color="white")+
  scale_fill_viridis_c(name="Density",option = "B",guide=F)+
  geom_point(data=indpreds,aes(x=longitude,y=latitude),alpha=0.25)+
  geom_point(data=trueloc,aes(x=longitude,y=latitude),col="red")


pred[,ncol(pred)] <- as.character(pred[,ncol(pred)])
best_pred$sampleID <- as.character(best_pred$sampleID)
p <- plot_uncertainty(pred,sampleID=best_pred$sampleID[1],
                 x=pd[pd$sampleID==best_pred$sampleID[1],"longitude"],
                 y=pd[pd$sampleID==best_pred$sampleID[1],"latitude"],
                 best_pred)
print(p)
