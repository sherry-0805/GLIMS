# use: R CMD BATCH --args brca_preprocessing.R

# data preprocessing
library(data.table)
library(magrittr)
library(dplyr)
library(stringr)
library(igraph)
library(tidyr)

node<-read.csv('./data/network/brca/max_connected_graph_nodes.csv')
colnames(node)[2]<-'gene'
probeMap <- read.table("./data/DATA_TCGA/gencode.v22.annotation.gene.probeMap", sep="\t",header=T,stringsAsFactors = FALSE,check.names=FALSE)

setwd('./data/DATA_TCGA/brca')
# load_sample
sample<-read.csv('./sample.csv',sep=';')
sample[1:3,1:3]
index<-sample$expr %in% sample$cn
sample1<-sample$expr[index]
index<-sample1 %in% sample$mut
sample1<-sample1[index]
index<-sample1 %in% sample$methy
sample1<-sample1[index]
can_sample<-data.frame(sample1)
colnames(can_sample)[1]<-'sample'
write.csv(can_sample,'./can_sample.csv',row.names = FALSE)

# expression
expr<-fread("./expr/TCGA-BRCA.htseq_fpkm.tsv.gz", h=T, check.names=F)
colnames(expr)[1]<-'Ensembl_ID'
expr_get <- expr %>%
  inner_join(probeMap, by=c("Ensembl_ID"="id") ) %>%
  select (gene, starts_with("TCGA"))
expr1<-expr_get
expr1$gene<-toupper(expr1$gene)
index<-expr1$gene %in% node$gene
expr2<-expr1[index,]
nrow(expr2)
index<-duplicated(expr2$gene)
expr2<-expr2[!index,]
expr<-expr2
# get cancer data
index<-colnames(expr) %in% can_sample$sample
expr1<-data.frame(expr)
expr1<-expr1[,index]
gene<-expr$gene
expr1<-data.frame(gene,expr1)
ncol(expr1)
nrow(expr1)
expr_can<-expr1
write.csv(expr1,'./expr/tcga_brca_expr.csv',row.names=FALSE)
# get normal data
sample_nor<-colnames(expr)
sample_nor<-data.frame(sample_nor)
number<-c(1:nrow(sample_nor))
sample_all<-data.frame(number,sample_nor)
sample_all$number<-0
sample_all$number<-str_count(sample_all$sample_nor,"11A")
index<-sample_all$number > 0
expr1<-data.frame(expr)
expr1<-expr1[,index]
gene<-expr$gene
expr1<-data.frame(gene,expr1)
write.csv(expr1,'./expr/tcga_brca_expr_normal.csv',row.names=FALSE)
expr_nor<-expr1
#calculate fold change
mean<-c(1:nrow(expr_nor))
expr_nor<-data.frame(mean,expr_nor)
expr_can<-data.frame(mean,expr_can)
expr_nor$mean<-rowMeans(expr_nor[3:ncol(expr_nor)])
expr_can$mean<-rowMeans(expr_can[3:ncol(expr_can)])
expr_cancer<-expr_can[,c(1,2)]
expr_nor<-expr_nor[,c(1,2)]
colnames(expr_cancer)[1]<-"mean_cancer"
colnames(expr_nor)[1]<-"mean_nor"
FC<-left_join(expr_cancer,expr_nor,by=c("gene"))
FC<-FC[,c("gene","mean_cancer","mean_nor")]
rate<-c(1:nrow(FC))
FC<-cbind(FC,rate)
FC$rate<-(FC$mean_cancer+0.00001)/(FC$mean_nor+0.00001)
log_v<-c(1:nrow(FC))
FC<-cbind(FC,log_v)
FC$log_v<-log2(FC$rate)
log_abs<-c(1:nrow(FC))
FC<-cbind(FC,log_abs)
FC$log_abs<-abs(FC$log_v)
write.csv(FC,"./expr/tcga_brca_expr_FC_detail.csv", row.names = FALSE)
FC<-FC[,c(1,6)]
colnames(FC)[2]<-"FC"
write.csv(FC,"./expr/tcga_brca_expr_FC.csv", row.names = FALSE)


# mutation
mut<-fread("./mut/TCGA-BRCA.muse_snv.tsv.gz",h=T, check.names=F)
data<-data.frame(mut)
index<-data$Sample_ID %in% can_sample$sample
data<-data[index,]
# 通过邻接矩阵处理mut数据
data1<-data[ ,c(1,2)]
data2<-graph.data.frame(as.matrix(data1))
data3<-get.adjacency(data2,sparse=FALSE)
data4<-data3[,-c(1:nrow(can_sample))]
data5<-data4[-c(nrow(can_sample)+1:nrow(data4)),]
data6<-t(data5)
mut<-data.frame(data6)
gene<-c(1:nrow(mut))
mut<-data.frame(gene,mut)
mut$gene<-rownames(mut)
index<-mut$gene %in% node$gene
mut<-mut[index,]
mut<-left_join(node,mut,by=c('gene'))
mut[is.na(mut)]<-0
mut<-mut[,-c(1)]
write.csv(mut,"./mut/brca_mut.csv",row.names=FALSE)

# freq
freq<-c(1:nrow(mut))
mut1<-cbind(freq,mut)
count<-c(1:nrow(mut1))
mut1<-cbind(count,mut1)
for(i in 1:nrow(mut1)){mut1$count[i]=rowSums(mut1[i,]==0)}
mut1$count<-(ncol(mut)-1)-mut1$count
mut1$freq<-mut1$count/ncol(mut)
freq_data<-mut1[,c('gene','freq')]
write.csv(freq_data,'./mut/tcga_brca_freq.csv',row.names=FALSE)

# copy number
cn<-fread('./cn/TCGA-BRCA.gistic.tsv.gz',h=T, check.names=F)
cn1<-cn[,colnames(cn) %in% can_sample$sample,with=FALSE]
colnames(cn)[1]<-'Ensembl_ID'
Ensembl_ID<-cn$Ensembl_ID
cn2<-data.frame(Ensembl_ID,cn1)
cn3 <- cn2 %>%
  inner_join(probeMap, by=c("Ensembl_ID"="id") ) %>%
  select (gene, starts_with('TCGA'))
cn3$gene<-toupper(cn3$gene)
index<-cn3$gene %in% node$gene
cn4<-cn3[index,]
cn<-cn4
index<-duplicated(cn$gene)
cn1<-cn[!index,]
cn2<-left_join(node,cn1,by=c('gene'))
cn2[is.na(cn2)]<-0
cn2<-cn2[,-c(1)]
cn<-cn2
write.csv(cn,'./cn/brca_cn.csv',row.names=FALSE)


# methylation
methy<-fread('./methy/TCGA-BRCA.methylation450.tsv.gz',h=T, check.names=F)
colnames(methy)[1]<-'cg'
cg<-methy$cg
methy1<-methy[,colnames(methy) %in% can_sample$sample,with=FALSE]
methy2<-data.frame(cg,methy1)
probe<-fread('./methy/illuminaMethyl450_hg38_GDC',h=T, check.names=F)
colnames(probe)[1]<-'id'
methy_get <-methy2 %>%
  inner_join(probe, by=c("cg"="id")) %>%
  select(gene, starts_with("TCGA"))
methy<-methy_get
nrow(methy)

mean<-c(1:nrow(methy))
methy1<-cbind(mean,methy)
methy1$mean = rowSums(methy1[,3:ncol(methy1)])
index<-is.na(methy1$mean)
methy2<-methy1[!index,]
nrow(methy2)
methy3<-methy2 %>%
  group_by(gene) %>% 
  filter(mean==max(mean))

number<-c(1:nrow(methy3))
methy3<-cbind(number,methy3) 
colnames(methy3)[1]<-'number'
methy3$number<-str_count(methy3$gene,",") 
max(methy3$number)
methy<-methy3
write.csv(methy,"./methy/methy_step1.csv",row.names=FALSE)
methy1<-methy %>%
  filter(number>1) 
methy2<-methy %>% 
  filter(number==1) 
methy3<-methy %>% 
  filter(number==0)
methy4<-separate(methy2,gene,into=c("gene1","gene2"),sep=",")
methy5<-methy4[,-c(4)]
methy4<-methy4[,-c(3)] 
colnames(methy4)[3]<-"gene"
colnames(methy5)[3]<-"gene"
methy6<-rbind(methy4,methy5)
methy3<-rbind(methy3,methy6)

methy4<-separate(methy1,gene,into=c("gene1","gene2","gene3","gene4","gene5","gene6","gene7", "gene8","gene9","gene10","gene11","gene12","gene13","gene14","gene15","gene16","gene17", "gene18","gene19","gene20","gene21","gene22"),sep=",")
for (i in 1:10){ 
  methy5<-methy4[,-c(4:((13-i)*2))] 
  methy6<-methy4[,-c(3,5:((13-i)*2))] 
  colnames(methy5)[3]<-"gene" 
  colnames(methy6)[3]<-"gene" 
  methy4<-methy4[,-c(3,4)] 
  if (i==1){methy7<-rbind(methy5,methy6)}
  if (i!=1){methy5<-rbind(methy5,methy6) 
  methy7<-rbind(methy5,methy7)}
}
methy5<- methy4[,-c(4)]
methy6<-methy4[,-c(3)]
colnames(methy5)[3]<-"gene"
colnames(methy6)[3]<-"gene"
methy5<-rbind(methy5,methy6)
methy7<-rbind(methy5,methy7)
nrow(methy7)
index<-is.na(methy7$gene)
methy7<-methy7[!index,]
methy3<-rbind(methy3,methy7)
methy<-methy3
methy1<-methy[!is.na(methy$gene),]
nrow(methy1)

methy2<-methy1 %>%
  group_by(gene) %>%
  filter(mean==max(mean))
nrow(methy2)
index<-methy2$gene %in% node$gene
methy3<-methy2[index,]
methy1<-left_join(node,methy3,by=c("gene"))
methy1[is.na(methy1)]<-0
methy<-methy1[,-c(1,3,4)]
write.csv(methy,"./methy/brca_methy.csv",row.names=FALSE)








