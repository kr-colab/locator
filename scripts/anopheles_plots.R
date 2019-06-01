#anopheles locator plots
library(ggplot2);library(magrittr);library(data.table);library(plyr);library(sp);library(cowplot)
setwd("~/locator/")

pred <- data.frame(fread("out/anopheles_predlocs.txt"))[-1,-1]
colnames(pred) <- c("longitude","latitude","sampleID")
samples <- fread("data/anopheles_samples_sp.txt")
pd <- merge(pred,samples,by="sampleID",all=T)
pd$species[is.na(pd$species)] <- "unknown"
trainlocs <- ddply(pd,.(longitude.y,latitude.y,species),summarize,n=length(species))

#mean error
plocs=as.matrix(pd[!is.na(pd$longitude.x),c("longitude.x","latitude.x")])
tlocs=as.matrix(pd[!is.na(pd$longitude.x),c("longitude.y","latitude.y")])
dists=sapply(1:nrow(plocs),function(e) spDistsN1(t(as.matrix(plocs[e,])),
                                           t(as.matrix(tlocs[e,])),longlat = T))
mean(dists)

map <- map_data("world")
p <- ggplot()+coord_map()+
  theme(axis.title.x = element_blank(),
        axis.title.y=element_blank(),
        axis.text=element_text(size=7),
        legend.position=c(0,0.25),
        legend.box = "horizontal",
        legend.title=element_text(size=7),
        legend.text=element_text(size=6),
        legend.spacing = unit(0,"mm"))+
  scale_color_manual(values=c("steelblue3","orangered","gold2"),name="")+
  scale_fill_manual(values=c("steelblue3","orangered","gold2"),name="species")+
  scale_size_continuous(name="samples",breaks=c(2,10,20),labels=c(10,50,100))+
  xlim(min(pd$longitude.y)-4,max(pd$longitude.y)+4)+
  ylim(min(pd$latitude.y)-4,max(pd$latitude.y)+4)+
  geom_path(data=map,aes(x=long,y=lat,group=group),col="grey50",lwd=0.2)+
  geom_segment(data=pd,aes(x=longitude.x,xend=longitude.y,y=latitude.x,yend=latitude.y),lwd=0.2)+
  geom_point(data=pd,aes(x=longitude.x,y=latitude.x,fill=species),size=1,shape=21,stroke=0.02)+
  geom_point(data=trainlocs,aes(x=longitude.y,y=latitude.y,fill=species,size=n/5),shape=21,stroke=0.1,alpha=0.3)
p <- p+guides(fill=guide_legend(keyheight=0.12,
                                 keywidth=0.1,
                                 default.unit = "inch",
                                 override.aes=list(size=3)),
              size=guide_legend(keyheight=0.12,
                                 keywidth=0.1,
                                 default.unit = "inch"))

pdf("fig/gambiae_map.pdf",width=3.25,height=2,useDingbats = F)
print(p)
dev.off()

p <- ggplot()+coord_map()+
  theme(axis.title.x = element_blank(),
        axis.title.y=element_blank(),
        axis.text=element_text(size=7),
        legend.position="bottom",
        legend.box = "vertical",
        legend.title=element_text(size=7),
        legend.text=element_text(size=6,face="italic"),
        legend.spacing = unit(0,"mm"),
        legend.box.margin = margin(-20,0,0,25),
        legend.spacing.y=unit(-.2,"inch"))+
  scale_color_manual(values=c("steelblue3","orangered","gold2"),name="",guide=F)+
  scale_fill_manual(values=c("steelblue3","orangered","gold2"),name="Species:")+
  scale_size_continuous(name="Samples:",breaks=c(2,10,20),labels=c(10,50,100))+
  xlim(min(pd$longitude.y)-4,max(pd$longitude.y)+4)+
  ylim(min(pd$latitude.y)-4,max(pd$latitude.y)+4)+
  geom_path(data=map,aes(x=long,y=lat,group=group),col="grey50",lwd=0.2)+
  geom_segment(data=pd,aes(x=longitude.x,xend=longitude.y,y=latitude.x,yend=latitude.y),lwd=0.2)+
  geom_point(data=pd,aes(x=longitude.x,y=latitude.x,fill=species),size=1,shape=21,stroke=0.01)+
  geom_point(data=trainlocs,aes(x=longitude.y,y=latitude.y,col=species,size=n/5),shape=1,stroke=0.3)
p <- p+guides(fill=guide_legend(keyheight=0.12,
                                keywidth=0.1,
                                default.unit = "inch",
                                override.aes=list(size=3)),
              size=guide_legend(keyheight=0.12,
                                keywidth=0.1,
                                default.unit = "inch"))

p2 <- ggplot(data=data.frame(dists),aes(x=dists))+
  theme(axis.title=element_text(size=7,vjust=3),
        axis.text=element_text(size=6),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        axis.title.y=element_blank())+
  #scale_x_continuous(breaks=c(0,750,1500))+
  geom_density(lwd=0.3,fill="grey")+
  xlab("Error (km)")+
  geom_vline(aes(xintercept=median(dists)),lwd=0.35,col="darkgreen")


pdf("fig/gambiae_combo_map.pdf",width=3.5,height=2.5,useDingbats = F)
ggdraw()+
  draw_plot(p,0,0,1,1)+
  draw_plot(p2,0.1,.25,.3,.45)
dev.off()

print(median(dists))
