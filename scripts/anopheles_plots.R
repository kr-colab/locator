#anopheles locator plots
library(ggplot2);library(magrittr);library(data.table);
library(plyr);library(sp);library(cowplot);library(MASS);library(progress)
setwd("~/locator/")

theme_set(theme_classic()+
          theme(axis.text=element_text(size=6),
                axis.title=element_text(size=8),
                legend.title=element_text(size=8),
                legend.text=element_text(size=6),
                strip.background=element_blank(),
                strip.text=element_text(size=8)))

kdepred <- function(xcoords,ycoords){
  long <- c();lat <- c()
  for(i in 1:nrow(xcoords)){ #run 2d kernel density estimate and assign predicted locations to site with highest density
    if(sd(xcoords[i,])>0.01 & sd(ycoords[i,])>0.01){ #prevent errors when predictions have near 0 spread (usually low dropout...)
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

pred <- data.frame(fread("out/anopheles_2L15_dense5_100000_5930_predlocs.txt"))[-1,-1]
colnames(pred)[ncol(pred)] <- "sampleID"
xcoords <- as.matrix(pred[,seq(1,ncol(pred)-1,2)])
ycoords <- as.matrix(pred[,seq(2,ncol(pred)-1,2)])
long <- c();lat <- c()
best_pred <- kdepred(xcoords,ycoords)
best_pred <- cbind(best_pred,pred[,ncol(pred)])
colnames(best_pred) <- c("longitude","latitude","sampleID")
samples <- fread("data/anopheles_samples_sp.txt")
pd <- merge(best_pred,samples,by="sampleID",all=T)
pd$species[is.na(pd$species)] <- "unknown"
trainlocs <- ddply(pd,.(longitude.y,latitude.y,species),summarize,n=length(species))

#median error
plocs=as.matrix(pd[!is.na(pd$longitude.x),c("longitude.x","latitude.x")])
tlocs=as.matrix(pd[!is.na(pd$longitude.x),c("longitude.y","latitude.y")])
dists=sapply(1:nrow(plocs),function(e) spDistsN1(t(as.matrix(plocs[e,])),
                                                 t(as.matrix(tlocs[e,])),longlat = T))

map <- map_data("world")
p <- ggplot()+coord_map(xlim = c(min(pd$longitude.x,na.rm=T)-5,max(pd$longitude.x,na.rm=T)+5),
                        ylim = c(min(pd$latitude.x,na.rm=T)-5,max(pd$latitude.x,na.rm=T)+5))+
  theme(axis.line=element_blank(),
        axis.ticks=element_blank(),
        axis.text=element_blank(),
        axis.title = element_blank(),
        #axis.text=element_text(size=7),
        legend.position="bottom",
        legend.box = "vertical",
        legend.title=element_text(size=7),
        legend.text=element_text(size=6,face="italic"),
        legend.spacing = unit(0,"mm"),
        legend.box.margin = margin(-12,0,0,20),
        legend.spacing.y=unit(-.1,"inch"),
        legend.background = element_blank())+
  scale_color_manual(values=c("steelblue3","orangered","gold2"),name="",guide=F)+
  scale_fill_manual(values=c("steelblue3","orangered","gold2"),name="Species:")+
  scale_size_continuous(name="Samples:",breaks=c(2,10,20),labels=c(10,50,100))+
  geom_polygon(data=map,aes(x=long,y=lat,group=group),fill="grey75",col="white",lwd=0.2)+
  geom_segment(data=pd,aes(x=longitude.x,xend=longitude.y,y=latitude.x,yend=latitude.y),lwd=0.2)+
  geom_point(data=pd,aes(x=longitude.x,y=latitude.x,fill=species),size=1.75,shape=21,stroke=0.01,alpha=0.7)+
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
        axis.title.y=element_blank(),
        panel.background = element_blank(),
        plot.background = element_blank())+
  #scale_x_continuous(breaks=c(0,750,1500))+
  geom_density(lwd=0.3,fill="grey")+
  xlab("Error (km)")+
  geom_vline(aes(xintercept=median(dists)),lwd=0.35)

pdf("fig/gambiae_combo_map.pdf",width=3.5,height=2.5,useDingbats = F)
ggdraw()+
  draw_plot(p,0,0,1,1)+
  draw_plot(p2,0.05,.2,.325,.45)
dev.off()
print(paste("mean error=",mean(dists),"km"))
print(paste("median error=",median(dists),"km"))

#visualize distribution of predictions for one individual
plot_uncertainty <- function(pred,sampleID,x,y,best_pred){
  xcoords <- as.matrix(pred[,seq(1,ncol(pred)-1,2)])
  ycoords <- as.matrix(pred[,seq(2,ncol(pred)-1,2)])
  ind <- sampleID
  indpreds <- data.frame(x=xcoords[pred$sampleID==ind,],y=ycoords[pred$sampleID==ind,])
  indpreds$sampleID <- ind
  colnames(indpreds) <- c("longitude","latitude")
  map <- map_data("world")
  p <- ggplot()+coord_map(xlim = c(min(indpreds$longitude)-7,max(indpreds$longitude)+7),
                     ylim = c(min(indpreds$latitude)-7,max(indpreds$latitude)+7))+
    theme(axis.title=element_blank())+
    geom_polygon(data=map,aes(x=long,y=lat,group=group),lwd=0.5,fill="grey",color="white")+
    scale_fill_viridis_c(name="Density",option = "B",guide=F)+
    stat_density_2d(data=indpreds,aes(x=longitude,y=latitude,fill = ..level..),
                    geom="polygon",alpha=0.5,n = 500)+
    geom_point(data=best_pred[best_pred$sampleID==ind,],aes(x=longitude,y=latitude),
               fill="white",stroke=0.1,shape=21,color="black",size=2)+
    geom_point(data=data.frame(x,y),aes(x=x,y=y),col="black")+
    annotate(geom="segment",x=x,y=y,lwd=0.5,
            xend=best_pred[best_pred$sampleID==ind,"longitude"],
            yend=best_pred[best_pred$sampleID==ind,"latitude"])
  #p <- p+guides(fill=guide_colorbar(barwidth = unit(3,"mm"),barheight = unit(15,"mm")))
}

plots <- list();inds <- sample(1:nrow(pred),9) #something weird with 161?
for(i in 1:9){
  p <- plot_uncertainty(pred,pred$sampleID[inds[i]],
                        pd[pd$sampleID==pred$sampleID[inds[i]],"longitude.y"],
                        pd[pd$sampleID==pred$sampleID[inds[i]],"latitude.y"],
                        best_pred)
  plots[[i]] <- p
}
pdf("fig/anopheles_locator_uncertainty_grid.pdf",width=6,height=6,useDingbats = F)
cowplot::plot_grid(plots[[1]],plots[[2]],plots[[3]],plots[[4]],
                   plots[[5]],plots[[6]],plots[[7]],plots[[8]],plots[[9]],ncol = 3)
dev.off()

