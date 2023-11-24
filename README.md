# GLIMS
This is the source code of "GLIMS: A two-stage gradual learning method for cancer genes prediction using multi-omics data and co-splicing network". This project presents a comprehensive approach for cancer gene prediction by integrating multiple omics data types and protein-protein interaction (PPI) networks. GLIMS adopts a two-stage progressive learning strategy: (1) a hierarchical graph neural network framework known as HIM-GCN is employed in a semi-supervised manner to predict candidate cancer genes by integrating multi-omics data and PPI networks; (2) the initial predictions are further refined by incorporating co-splicing network information using an unsupervised approach, which serves to prioritize the identified cancer genes.

# Dependencies
python(version=3.7.9) ; 
tensorflow (version=1.15.0) ; numpy (version=1.19.1); pandas (version=1.2.4) ; scikit-learn (version=0.24.2) ; scipy (version=1.6.2) ; h5py (version=2.10.0) ; networkx (version=2.5.1) ; mygene (version=3.2.2); gcn.

Considering the compatibility issues between gcn and tensorFlow, we have provided a gcn library in ```./code/HIM-GCN/GCN```. The original GCN code can be found at 'https://github.com/tkipf/gcn'.

# Guided Tutorial
Considering the complete dataset is rather large, you can download the full files from "[here]". These files include not only all the code but also the necessary ```.h5``` file for cancer gene prediction, containing the PPI network, multi-omics gene features, and the required training labels.

Here, we illustrate the usage of the model using BRCA as an example.
```
# Construction of data container of cancer
python ./code/preprocessing_data/build_max_connected_graph.py -c brca
# -c gbm/brca/luad/pancancer

R CMD BATCH --args ./code/preprocessing_data/brca_preprocessing.R
# --args gbm_preprocessing.R/brca_preprocessing.R/luad_preprocessing.R/pancancer_preprocessing.R

python ./code/HIM-GCN/data_preparation.py  -c brca
# -c gbm/brca/luad/pancancer

# Model training
python ./code/HIM-GCN/train_himgcn_cv.py -cv 3 -e 1500 -d ./code/HIM-GCN/data_container/brca_test_data.h5
```

```brca_test_data.h5``` is an example input file that contains a subnetwork composed of 378 genes and multi-omics features for each gene, and the HIM-GCN outputs the likelihood of all input genes being cancer genes after training. The ```partial_correlation.R``` script is used to calculate the partial correlation coefficients between the cancer gene candidates and AS events. The ```co_regulation_network.R``` script is used to construct a comprehensive cancer-related co-splicing network. Finally, the PageRank algorithm is applied to re-rank the candidates.


[here]: https://zenodo.org/records/10202473?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjcxNTJmNzA2LTZmNDItNGExNS05YzY3LWIwYzc3YTU5NzZkZiIsImRhdGEiOnt9LCJyYW5kb20iOiIwNjMwNTcxNjFkYTIxOGJlYjkyYzI4YjE5YzBmMGFlNyJ9.bbmH69dNa_c8g4jUKQC_4AeQnZ75eRLdvRIIzhMS6a70j8dbooJ7ghzSqq27-j0s7-2hB3X21-3XBwywXV5E-A


