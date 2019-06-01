#anopheles locator plots
library(ggplot2);library(magrittr);library(data.table)
setwd("~/locator/")

pred <- data.frame(fread("stats/anopheles_predlocs.txt"))
test <- data.frame(fread("stats/anopheles_testlocs.txt"))
colnames(pred) <- c("pred_long","pred_lat")
colnames(test) <- c("true_long","true_lat")
pd <- cbind(pred,test)
pred$type <- "prediction"
test$type <- "truth"
colnames(pred) <- c("long","lat","type")
colnames(test) <- c("long","lat","type")
longpd <- rbind(pred,test)

refsamples <- fread("data/anopheles_samples_sp.txt")
refsamples <- subset(refsamples,refsamples$species=="gambiae")

map <- map_data("world")
p <- ggplot()+coord_map()+
  theme(axis.title.x = element_blank(),
        axis.title.y=element_blank(),
        legend.position=c(0.2,0.25))+
  scale_color_manual(values=c("steelblue3","orangered"),name="")+
  xlim(min(longpd$long)-3,max(longpd$long)+3)+
  ylim(min(longpd$lat)-3,max(longpd$lat)+3)+
  geom_path(data=map,aes(x=long,y=lat,group=group),col="grey30",lwd=0.1)+
  geom_segment(data=pd,aes(x=true_long,xend=pred_long,y=true_lat,yend=pred_lat),lwd=0.25)+
  #geom_point(data=refsamples,aes(x=longitude,y=latitude),shape=1,size=1,col='black')+
  geom_point(data=longpd,aes(x=long,y=lat,col=type),shape=1,size=1)
p <- p+guides(color=guide_legend(keyheight=0.12,keywidth=0.1,default.unit = "inch"))
pdf("fig/gambiae_map.pdf",width=3.5,height=2,useDingbats = F)
print(p)
dev.off()


