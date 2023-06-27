import argparse
import gzip
import pandas as pd
import zipfile
import networkx as nx
import os
import pandas as pd


parser = argparse.ArgumentParser(description='cancer type')
parser.add_argument('-c', '--cancer_type', dest='cancer_type', default='gbm', type=str)
# gbm, brca, luad, pancancer
args = parser.parse_args()
cancer_type = args.cancer_type

# load the expression data
def obtain_tcga_genes():
    probeMap = pd.read_table('./data/DATA_TCGA/gencode.v22.annotation.gene.probeMap', sep='\t')
    if cancer_type!='pancancer':
        expr_path = './data/DATA_TCGA/'+cancer_type+'/expr/TCGA-'+str.upper(cancer_type)+'.htseq_fpkm.tsv.gz'
    else:
        expr_path = './data/DATA_TCGA/pancancer/expr/GDC-PANCAN.htseq_fpkm-uq.tsv.gz'
    
    print('load expression data from', expr_path)
    with gzip.open(expr_path, 'rb') as f:
        expr = pd.read_csv(f, sep='\t')
        f.close()
        print('load expression data')
    expr_gene_ensembl = expr['Ensembl_ID']
    expr_gene = pd.merge(expr_gene_ensembl, probeMap, left_on='Ensembl_ID', right_on='id')
    tcga_expr_gene = expr_gene['gene'].str.upper()
    tcga_expr_gene = pd.DataFrame(tcga_expr_gene, columns=['gene'])
    return tcga_expr_gene


# load origin ppi network
def load_ppi_network():
    with zipfile.ZipFile('./data/network/PICKLE_HUMAN_3_2_PPI_Network_default.zip', 'r') as z:
        f = z.open(z.namelist()[0])
        ppi = pd.read_table(f, sep='\t')
        f.close()
        z.close()
    ppi = ppi[['InteractorA', 'InteractorB']]
    ppi_ann = pd.read_table('./data/network/GeneID_Conversion.txt', sep='\t')
    ppi_ann = ppi_ann[['Gene_ID', 'Gene_Symbol']]
    ppi_a = pd.merge(ppi, ppi_ann, left_on='InteractorA', right_on='Gene_ID')
    ppi_a = ppi_a.rename(columns={'Gene_Symbol': 'GeneA'})
    ppi_b = pd.merge(ppi_a, ppi_ann, left_on='InteractorB', right_on='Gene_ID')
    ppi_b = ppi_b.rename(columns={'Gene_Symbol': 'GeneB'})
    ppi_net = ppi_b[['GeneA', 'GeneB']]
    ppi_net['GeneA'] = ppi_net['GeneA'].str.upper()
    ppi_net['GeneB'] = ppi_net['GeneB'].str.upper()
    return ppi_net


# obtain max connected graph
def merge_ppi_tcga_genes(tcga_genes, ppi_net):
    ppi_net1 = pd.merge(ppi_net, tcga_genes, left_on=['GeneA'], right_on=['gene'], how='inner')
    ppi_net2 = pd.merge(ppi_net1, tcga_genes, left_on=['GeneB'], right_on=['gene'], how='inner')
    ppi = ppi_net2[['gene_x', 'gene_y']]
    ppi = ppi.rename(columns={'gene_x': 'GeneA'})
    ppi = ppi.rename(columns={'gene_y': 'GeneB'})
    # drop the duplicates
    ppi = ppi.drop_duplicates()
    return ppi


# obtain max connected graph
def obtain_max_connected_graph(ppi):
    edge_list = []
    node_set = set()
    for i in range(ppi.shape[0]):
        y0 = ppi.iloc[i,0]
        y1 = ppi.iloc[i,1]
        node_set.add(y0)
        node_set.add(y1)
        edge = (y0, y1)
        edge_list.append(edge)
    T = nx.Graph()
    T.add_edges_from(edge_list)
    C = sorted(nx.connected_components(T), key=len, reverse=True)
    connected_max = C[0]
    print('The number of genes in max connected graph is', len(C[0]))
    connected_g = pd.DataFrame(connected_max, columns=['gene'])
    return connected_g


def obtain_max_connected_graph_edges(nodes, ppi):
    ppi1 = pd.merge(ppi, nodes, left_on=['GeneA'], right_on=['gene'], how='inner')
    ppi2 = pd.merge(ppi1, nodes, left_on=['GeneB'], right_on=['gene'], how='inner')
    max_g = ppi2[['gene_x', 'gene_y']]
    max_g = max_g.rename(columns={'gene_x': 'GeneA'})
    max_g = max_g.rename(columns={'gene_y': 'GeneB'})
    return max_g


tcga_genes = obtain_tcga_genes()
ppi_net = load_ppi_network()
ppi = merge_ppi_tcga_genes(tcga_genes, ppi_net)
connected_g = obtain_max_connected_graph(ppi)
graph_nodes_dir = './data/network/'+cancer_type
if not os.path.isdir(graph_nodes_dir):
    os.makedirs(graph_nodes_dir)
    print('Create cancer container dir', graph_nodes_dir)
else:
    pass
connected_g.to_csv(graph_nodes_dir+'/max_connected_graph_nodes.csv')
max_g_edges = obtain_max_connected_graph_edges(connected_g, ppi)
max_g_edges.to_csv(graph_nodes_dir+'/connected_graph.csv')

print('Finish')
