import mygene
import os, sys, h5py
import pandas as pd
from sklearn.decomposition import NMF, PCA


# definite the structure of NMF and PCA
def build_nmf(file, components=20):
    file_value = file.values
    nmf = NMF(n_components=components, init=None, l1_ratio=0, alpha=0.01)
    file_nmf = nmf.fit_transform(file_value)
    return file_nmf


def build_pca(file, components=0.7):
    file_value = file.values
    pca = PCA(n_components=components)
    file_pca = pca.fit_transform(file_value)
    return file_pca


# definite a function to obtain genes with ensembl id
def obtain_gene_with_ensembl_id(nodes):
    my_info = mygene.MyGeneInfo()
    info = my_info.querymany(nodes, scopes='symbol, refseg, uniprot', fields='ensembl.gene',species='human', returnall=True)
    res = []
    for query_res in info['out']:
        gene = query_res['query']
        if 'ensembl' in query_res:
            if type(query_res['ensembl']) is list:
                ensembl_id = query_res['ensembl'][0]['gene']
            else:
                ensembl_id = query_res['ensembl']['gene']
        else:
            ensembl_id = None
        query_result = [gene, ensembl_id]
        res.append(query_result)
    gene_with_ensembl = pd.DataFrame(res, columns=['Symbol', 'Ensembl_ID']).set_index('Ensembl_ID')
    gene_with_ensembl.dropna(axis=0, inplace=True)
    gene_with_ensembl.drop_duplicates(inplace=True)
    return gene_with_ensembl


# definite a function to get the positive cancer genes
def obtain_positive_cancer_genes(nodes, cancer_type='pancancer'):
    if cancer_type=='pancancer':
        # obtain known cancer genes from NCG
        ncg = pd.read_table('./data/cancer_gene/NCG/cancergenes_list.txt',sep='\t')
        ncg_known_cancer_genes = list(ncg.iloc[:,0].dropna(axis=0))
        # obtain cancer genes from COSMIC
        cosmic = pd.read_csv('./data/cancer_gene/COSMIC/cancer_gene_census.csv', header=0)
        cosmic_cancer_genes = list(cosmic['Gene Symbol'])
        cancer_genes = set(ncg_known_cancer_genes + cosmic_cancer_genes)
        l = len(cancer_genes)
        print("The number of known cancer genes in ncg and cosmic is", l)

    else:
        cosmic = pd.read_csv('./data/cancer_gene/COSMIC/cancer_gene_census.csv', header=0)
        if cancer_type == 'gbm':
            # for the cancer gene selection of GBM, taking oligodendroglioma, glioma, and glioblastoma | GBM into consideration
            index_somatic = cosmic['Tumour Types(Somatic)'].str.contains('glioma|glioblastoma|GBM|oligodendroglioma',na=False)
            index_germline = cosmic['Tumour Types(Germline)'].str.contains('glioma|glioblastoma|GBM|oligodendroglioma',na=False)
            # remove the genes related with paraganglioma
            index_paragbm_somatic = cosmic['Tumour Types(Somatic)'].str.contains('paraganglioma', na=False)
            index_paragbm_germline = cosmic['Tumour Types(Germline)'].str.contains('paraganglioma', na=False)
            gbm = cosmic.loc[index_somatic | index_germline]
            paragbm = cosmic.loc[index_paragbm_somatic | index_paragbm_germline]
            cancer_genes = pd.concat([gbm, paragbm, paragbm]).drop_duplicates(keep=False)
            cancer_genes = list(cancer_genes['Gene Symbol'])
            l = len(cancer_genes)
            print("The number of known cancer genes in gbm is", l)

        if cancer_type == 'luad':
            # for the cancer gene selection of LUAD, taking lung adenocarcinoma, lung, and lung cancer into consideration
            index_somatic = cosmic['Tumour Types(Somatic)'].str.contains('lung adenocarcinoma|lung|lung cancer',na=False)
            index_germline = cosmic['Tumour Types(Germline)'].str.contains('lung adenocarcinoma|lung|lung cancer',na=False)
            # remove the genes related with small cell lung carcinoma
            index_paragbm_somatic = cosmic['Tumour Types(Somatic)'].str.contains('small cell lung carcinoma', na=False)
            index_paragbm_germline = cosmic['Tumour Types(Germline)'].str.contains('small cell lung carcinoma',na=False)
            luad = cosmic.loc[index_somatic | index_germline]
            small_l = cosmic.loc[index_paragbm_somatic | index_paragbm_germline]
            cancer_genes = pd.concat([luad, small_l, small_l]).drop_duplicates(keep=False)
            cancer_genes = list(cancer_genes['Gene Symbol'])
            l = len(cancer_genes)
            print("The number of known cancer genes in luad is", l)

        if cancer_type == 'brca':
            # for the cancer gene selection of BRCA, taking breast into consideration
            index_somatic = cosmic['Tumour Types(Somatic)'].str.contains('breast', na=False)
            index_germline = cosmic['Tumour Types(Germline)'].str.contains('breast', na=False)
            cancer_genes = cosmic.loc[index_somatic | index_germline]
            cancer_genes = list(cancer_genes['Gene Symbol'])
            l = len(cancer_genes)
            print("The number of known cancer genes in luad is", l)

    positive_cancer_genes = nodes[nodes.gene.isin(cancer_genes)].gene
    l = len(positive_cancer_genes)
    print("The total number of positive cancer genes in", cancer_type,"is", l)
    return positive_cancer_genes


def obtain_negative_normal_genes(nodes):
    # obtain known and candidate cancer genes from NCG
    ncg = pd.read_table('./data/cancer_gene/NCG/cancergenes_list.txt', sep='\t')
    ncg_known_cancer_genes = list(ncg.iloc[:, 0].dropna(axis=0))
    ncg_candidate_cancer_genes = list(ncg.iloc[:, 1].dropna(axis=0))
    ncg_genes = list(set(ncg_known_cancer_genes + ncg_candidate_cancer_genes))
    l = len(ncg_genes)
    print("The number of known and candidate cancer genes from NCG is", l)

    # obtain cancer genes from COSMIC
    cosmic = pd.read_csv('./data/cancer_gene/COSMIC/cancer_gene_census.csv', header=0)
    cosmic_genes = list(cosmic['Gene Symbol'])
    l = len(cosmic_genes)
    print("The number of cancer genes from COSMIC is", l)

    # obtain disease related genes from OMIM
    omim = pd.read_table('./data/cancer_gene/OMIM/genemap2.txt', sep='\t', comment="#", header=None)
    omim_genes = []
    for i in range(0, len(omim)):
        omim_disease_gene = omim.iloc[i, 6].split(',')
        omim_genes += omim_disease_gene
    omim_genes = list(set(omim_genes))
    l = len(omim_genes)
    print("The number of disease genes from OMIM is", l)

    # obtain cancer pathway related genes from KEGG
    kegg = pd.read_csv('./data/cancer_gene/KEGG/KEGG_genes_in_pathways_in_cancer.txt', header=0)
    kegg_genes = list(kegg.iloc[1:len(kegg), 0])
    l = len(kegg_genes)
    print("The number of cancer pathway related genes from KEGG is", l)

    cancer_disease_genes = list(set(ncg_genes + cosmic_genes + omim_genes + kegg_genes))
    print("The number of total cancer and disease related genes is", len(cancer_disease_genes))
    negative_normal_genes = nodes[~nodes.gene.isin(cancer_disease_genes)]
    print("The number of total negative normal genes is", len(negative_normal_genes))

    return negative_normal_genes



