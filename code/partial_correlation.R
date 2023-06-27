library(tidyr)
library(dplyr)
library(tidyr)
library(readxl)

# R CMD BATCH --args ./brca/20220423/A3/brca_a3_pc.R

# BRCA
# A3
# This is an example of calculating the partial correlation coefficient between predicted BRCA genes and A3 alternative splicing.

setwd('./partial_cor/brca')
pre_gene <- read.csv('./data/brca_gene.csv')
as <- read_excel('./data/brca_as.xls', sheet = "a3")
as_psi <- read_excel('./data/as_psi.xlsx', sheet = 'brca_tcga')
expr <- read.csv('./data/tcga_brca_expr.csv')

index <- as_psi$AS %in% as$AS
as_psi <- as_psi[index,]
as_psi <- data.frame(as_psi)
var <- apply(as_psi[, 4:ncol(as_psi)], 1, var)
as_psi <- cbind(as_psi, var)

nrow(as)
index <- as_psi$var > 0.005
as_psi <- as_psi[index,]
index <- as$AS %in% as_psi$AS
as <- as[index,]
nrow(as)

GeneName <- 1:nrow(as)
map <- data.frame(GeneName, gene = as$gene, AS = as$AS)
map$GeneName <- pre_gene$GeneName[1]

for (i in 2:nrow(pre_gene)){
  GeneName <- 1:nrow(as)
  map1 <- data.frame(GeneName, gene = as$gene, AS = as$AS)
  map1$GeneName <- pre_gene$GeneName[i]
  map <- rbind(map, map1)
}

index <- map$gene %in% expr$gene
map <- map[index,]
write.csv(map, './A3/brca_gene_a3_map.csv', row.names = FALSE)
map <- read.csv('./A3/brca_gene_a3_map.csv')
expr1 <- expr[, -1]
rownames(expr1) <- expr$gene
as_psi <- read_excel('./data/as_psi.xlsx', sheet = 'brca_tcga')
index <- as$AS %in% as_psi$AS
as <- as[index,]
as_psi1 <- as_psi[, 3:ncol(as_psi)]
rownames(as_psi1) <- paste(as_psi$gene_name, as_psi$AS, sep = '_')
sample <- substr(colnames(expr1), 1, 15)
colnames(expr1) <- sample
index <- colnames(as_psi1) %in% colnames(expr1)
as_psi1 <- as_psi1[, index]
index <- colnames(expr1) %in% colnames(as_psi1)
expr1 <- expr1[, index]
# Adjust the sample order
expr1 <- expr1[, order(colnames(expr1))]
as_psi1 <- as_psi1[, order(colnames(as_psi1))]
order_label <- ifelse(colnames(expr1) == colnames(as_psi1), 1, 0)
order_label

# The calculation of e1g2
e1g2 <- map
gene_AS <- paste(e1g2$gene, e1g2$AS, sep = '_')
e1g2 <- data.frame(e1g2, gene_AS)
cor_results <- apply(e1g2, 1, function(row) {
  g2 <- row["GeneName"]
  e1 <- row["gene_AS"]
  index_g2 <- rownames(expr1) == g2
  index_e1 <- rownames(as_psi1) == e1
  c_r <- cor(as.numeric(as_psi1[index_e1, ]), as.numeric(expr1[index_g2, ]), method = "pearson")
  p <- cor.test(as.numeric(as_psi1[index_e1, ]), as.numeric(expr1[index_g2, ]), method = "pearson")$p.value
  
  data.frame(GeneName = g2, gene_AS = e1, cor_r = c_r, pvalue = p)
})
e1g2_cor <- do.call(rbind, cor_results)
write.csv(e1g2_cor, './A3/e1g2_cor.csv', row.names = FALSE)

# The calculation of e1g1
e1g1 <- map[, c('gene', 'AS')]
gene_AS <- paste(e1g1$gene, e1g1$AS, sep = '_')
e1g1 <- data.frame(e1g1, gene_AS)
e1g1 <- e1g1[!duplicated(e1g1$gene_AS), ]
cor_results <- apply(e1g1, 1, function(row) {
  g1 <- row["gene"]
  e1 <- row["gene_AS"]
  index_g1 <- rownames(expr1) == g1
  index_e1 <- rownames(as_psi1) == e1
  c_r <- cor(as.numeric(as_psi1[index_e1, ]), as.numeric(expr1[index_g1, ]), method = "pearson")
  p <- cor.test(as.numeric(as_psi1[index_e1, ]), as.numeric(expr1[index_g1, ]), method = "pearson")$p.value
  data.frame(gene = g1, gene_AS = e1, cor_r = c_r, pvalue = p)
})
e1g1_cor <- do.call(rbind, cor_results)
write.csv(e1g1_cor, './A3/e1g1_cor.csv', row.names = FALSE)

# The calculation of g2g1
g2g1 <- map[, c('GeneName', 'gene')]
g2g1 <- data.frame(g2g1, test = 1:nrow(g2g1))
g2g1 <- g2g1[!duplicated(paste(g2g1$GeneName, g2g1$gene, sep = '_')), ]
cor_results <- apply(g2g1, 1, function(row) {
  g2 <- row["GeneName"]
  g1 <- row["gene"]
  index1 <- rownames(expr1) == g1
  index2 <- rownames(expr1) == g2
  c_r <- cor(as.numeric(expr1[index2, ]), as.numeric(expr1[index1, ]), method = "pearson")
  p <- cor.test(as.numeric(expr1[index2, ]), as.numeric(expr1[index1, ]), method = "pearson")$p.value
  data.frame(GeneName = g2, gene = g1, cor_r = c_r, pvalue = p)
})
g2g1_cor <- do.call(rbind, cor_results)
write.csv(g2g1_cor, './A3/g2g1_cor.csv', row.names = FALSE)

# The calculation of pc
all_cor <- map
all_cor <- data.frame(all_cor, test = 1:nrow(all_cor))
all_cor$test <- paste(all_cor$GeneName, all_cor$gene, all_cor$AS, sep = '_')
e1g2_cor <- e1g2_cor[, c('gene_AS', 'cor_r')]
all_cor <- left_join(all_cor, e1g2_cor, by = c('test'))
colnames(all_cor)[5] <- 'e1g2_cor'
e1g1_cor <- e1g1_cor[, c('gene_AS', 'cor_r')]
all_cor <- left_join(all_cor, e1g1_cor, by = c('gene_AS'))
colnames(all_cor)[6] <- 'e1g1_cor'
gene_map_cor <- gene_map_cor[, c('test', 'cor_r')]
all_cor <- left_join(all_cor, gene_map_cor, by = c('test'))
colnames(all_cor)[7] <- 'g2g1_cor'
write.csv(all_cor, './A3/all_cor.csv', row.names = FALSE)
pc <- read.csv('./A3/all_cor.csv')
pc$a <- pc$e1g2_cor - pc$e1g1_cor * pc$g2g1_cor
pc$b <- sqrt((1 - pc$e1g1_cor^2) * (1 - pc$g2g1_cor^2))
pc$partial_cor <- pc$a / pc$b
pc <- pc[!is.na(pc$partial_cor), ]
pc$partial_cor_abs <- abs(pc$partial_cor)
write.csv(pc, './A3/partial_cor.csv', row.names = FALSE)



