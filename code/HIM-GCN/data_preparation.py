import os, sys
import argparse
import h5py
import pandas as pd
import networkx as nx
import numpy as np
import time
from sklearn import preprocessing

import data_preprocessing_utils as utils


use_expression_data = True
use_expr_fc = True
use_expr = True
use_mutation_data = True
use_mutfreq = True
use_mut = True
use_cn_data = True
use_methy_data = True
use_pca = True
use_nmf = False
use_minmax = True

parser = argparse.ArgumentParser(description='cancer type')
parser.add_argument('-c', '--cancer_type',dest='cancer_type',default='pancancer',type=str)
args = parser.parse_args()
cancer_type = args.cancer_type

ppi = pd.read_csv('./data/network/'+cancer_type+'/connected_graph.csv', header=0)
ppi_graph = nx.from_pandas_edgelist(df=ppi, source='GeneA', target='GeneB', edge_attr=None)
ppi = nx.to_pandas_adjacency(G=ppi_graph)

if use_expression_data:
    if use_expr_fc:
        expr_fc = pd.read_csv('./data/DATA_TCGA/'+cancer_type+'/expr/tcga_'+cancer_type+'_expr_FC.csv', header=0)
        expr_fc.set_index('gene', inplace=True)
    else:
        pass
    if use_expr:
        expr = pd.read_csv('./data/DATA_TCGA/'+cancer_type+'/expr/tcga_'+cancer_type+'_expr.csv', header=0)
        expr.set_index('gene', inplace=True)
    else:
        pass
else:
    pass

if use_mutation_data:
    if use_mutfreq:
        mutfreq = pd.read_csv('./data/DATA_TCGA/'+cancer_type+'/mut/tcga_' + cancer_type + '_freq.csv', header=0)
        mutfreq.set_index('gene', inplace=True)
    else:
        pass
    if use_mut:
        mut = pd.read_csv('./data/DATA_TCGA/'+cancer_type+'/mut/' + cancer_type + '_mut.csv', header=0)
        mut.set_index('gene', inplace=True)
    else:
        pass
else:
    pass

if use_cn_data:
    cn = pd.read_csv('./data/DATA_TCGA/'+cancer_type+'/cn/' + cancer_type + '_cn.csv', header=0)
    cn.set_index('gene', inplace=True)
else:
    pass

if use_methy_data:
    methy = pd.read_csv('./data/DATA_TCGA/'+cancer_type+'/methy/' + cancer_type + '_methy.csv', header=0)
    methy.set_index('gene', inplace=True)
else:
    pass

expr_fc_reindex = expr_fc.reindex(ppi.index, fill_value=0)
expr_reindex = expr.reindex(ppi.index, fill_value=0)
mutfreq_reindex = mutfreq.reindex(ppi.index, fill_value=0)
mut_reindex = mut.reindex(ppi.index, fill_value=0)
cn_reindex = cn.reindex(ppi.index, fill_value=0)
methy_reindex = methy.reindex(ppi.index, fill_value=0)

print((expr_fc_reindex.index == expr_reindex.index).all())
print((expr_fc_reindex.index == mutfreq_reindex.index).all())
print((mutfreq_reindex.index == mut_reindex.index).all())
print((mut_reindex.index == cn_reindex.index).all())
print((cn_reindex.index == methy_reindex.index).all())

if use_minmax:
    scaler = preprocessing.MinMaxScaler()
    expr_fc = scaler.fit_transform(np.abs(expr_fc_reindex))
    mutfreq = scaler.fit_transform(mutfreq_reindex)
else:
    expr_fc = np.array(expr_fc_reindex)
    mutfreq = np.array(mutfreq_reindex)

if use_pca:
    expr = utils.build_pca(expr_reindex, components=0.98)
    mut = utils.build_pca(mut_reindex, components=0.7)
    cn = utils.build_pca(cn_reindex, components=0.7)
    methy = utils.build_pca(methy_reindex, components=0.98)
else:
    expr = utils.build_nmf(expr_reindex, components=40)
    mut = utils.build_nmf(mut_reindex, components=20)
    cn = utils.build_nmf(cn_reindex, components=20)
    methy = utils.build_nmf(methy_reindex, components=40)

expr_c = np.concatenate((expr, expr_fc), axis=1)
mut_c = np.concatenate((mut, mutfreq), axis=1)
nodes = utils.obtain_gene_with_ensembl_id(ppi.index)
nodes.head()
nodes.columns = ['gene']

pos_cancer_genes = utils.obtain_positive_cancer_genes(nodes, cancer_type=cancer_type)
neg_normal_genes = utils.obtain_negative_normal_genes(nodes)

if cancer_type=='pancancer':
    neg_normal_genes = neg_normal_genes
else:
    n = len(pos_cancer_genes)
    neg_normal_genes = neg_normal_genes.sample(n=n*10)

pos_genes = nodes.gene.isin(pos_cancer_genes).values.reshape(-1,1)
neg_genes = nodes.gene.isin(neg_normal_genes['gene']).values.reshape(-1,1)
print(pos_genes.sum())
print(neg_genes.sum())

nodes['ID'] = nodes.index
nodes = nodes[['ID', 'gene']]
t = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
container_dir = './code/HIM-GCN/data_container/'+cancer_type
if not os.path.isdir(container_dir):
    os.makedirs(container_dir)
    print('Create cancer container dir', container_dir)
data_path = './code/HIM-GCN/data_container/'+cancer_type+'/'+cancer_type+'_data_'+t+'.h5'
print('The data path of container is', data_path)

data_container = h5py.File(data_path, 'w')
data_container.create_dataset('network', data=ppi.values, shape=ppi.shape)
data_container.create_dataset('gene_name', data=nodes, dtype=h5py.special_dtype(vlen=str))
data_container.create_dataset('expr', data=expr_c, shape=expr_c.shape)
data_container.create_dataset('mut', data=mut_c, shape=mut_c.shape)
data_container.create_dataset('cn', data=cn, shape=cn.shape)
data_container.create_dataset('methy', data=methy, shape=methy.shape)
data_container.create_dataset('pos', data=pos_genes, shape=pos_genes.shape)
data_container.create_dataset('neg', data=neg_genes, shape=neg_genes.shape)
data_container.close()














