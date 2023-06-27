library(tidyr)
library(dplyr)
library(tidyr)
library(readxl)

# the construction of cancer-related genes co-regulation network

pre_gene <- read.csv('./predicted_genes/brca_gene.csv')
edges <- data.frame(GeneA = rep(pre_gene$GeneName, each = nrow(pre_gene)),
                    GeneB = rep(pre_gene$GeneName, times = nrow(pre_gene)))
edges <- edges[edges$GeneA != edges$GeneB, ] # Remove self-loops
edges$weight <- 0
write.csv(edges, './edges.csv', row.names = FALSE)

edges <- read.csv('./edges.csv')
file_paths <- c('./A3/partial_cor.csv',
                './A5/partial_cor.csv',
                './AL/partial_cor.csv',
                './AF/partial_cor.csv',
                './SE/partial_cor.csv',
                './RI/partial_cor.csv',
                './MX/partial_cor.csv')

pc <- data.frame()
for (path in file_paths) {
  df <- read.csv(path)
  index <- df$partial_cor_abs > 0.2
  df <- df[index, ]
  pc <- rbind(pc, df)
}

# Count the number of AS events linked to each gene
gene_as <- pc %>%
  count(GeneName) %>%
  rename(GeneA = GeneName) %>%
  rename(GeneA_as = n)

# Construction of cancer-related co-splicing network
co_network<-left_join(edges,gene_as,by=c('GeneA'))
colnames(co_network)[4]<-'GeneA_as'
colnames(gene_as)[1]<-'GeneB'
co_network<-left_join(co_network,gene_as,by=c('GeneB'))
colnames(co_network)[5]<-'GeneB_as'

# Count the number of common AS events
for (i in 1:nrow(co_network)) {
  gene_a <- co_network$GeneA[i]
  gene_b <- co_network$GeneB[i]
  as_a <- pc[pc$GeneName == gene_a, ]
  as_b <- pc[pc$GeneName == gene_b, ]
  as_c <- as_a[as_a$AS %in% as_b$AS, ]
  n <- nrow(as_c)
  co_network$common_as[i] <- n
}

# Calculation of edge weight
co_network$weight <- co_network$common_as * 2 / (co_network$GeneA_as + co_network$GeneB_as)
co_network[is.na(co_network)] <- 0
co_network <- co_network[co_network$GeneA != co_network$GeneB, ]
index <- co_network$weight > 0.3
co_network <- co_network[index, ]
write.csv(co_network, './cancer_genes_co_regulation_network.csv', row.names = FALSE)



