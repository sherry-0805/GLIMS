# GLIMS
This is the source code of "GLIMS: A two-stage gradual learning method for cancer genes prediction using multi-omics data and co-splicing network". This project presents a comprehensive approach for cancer gene prediction by integrating multiple omics data types and protein-protein interaction (PPI) networks. GLIMS adopts a two-stage progressive learning strategy: (1) a hierarchical graph neural network framework known as HIM-GCN is employed in a semi-supervised manner to predict candidate cancer genes by integrating multi-omics data and PPI networks; (2) the initial predictions are further refined by incorporating co-splicing network information using an unsupervised approach, which serves to prioritize the identified cancer genes.

# Dependencies
python(version=3.7.9) ; 
tensorflow (version=1.15.0) ; numpy (version=1.19.1); pandas (version=1.2.4) ; scikit-learn (version=0.24.2) ; scipy (version=1.6.2) ; h5py (version=2.10.0) ; networkx (version=2.5.1) ; mygene (version=3.2.2)
