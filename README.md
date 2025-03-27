# **scNET: Learning Context-Specific Gene and Cell Embeddings by Integrating Single-Cell Gene Expression Data with Protein-Protein Interaction Information**

## **Overview**

Recent advances in single-cell RNA sequencing (scRNA-seq) techniques have provided unprecedented insights into tissue heterogeneity. However, gene expression data alone often fails to capture changes in cellular pathways and complexes, which are more discernible at the protein level. Additionally, analyzing scRNA-seq data presents challenges due to high noise levels and zero inflation. In this study, we propose a novel approach to address these limitations by integrating scRNA-seq datasets with a protein-protein interaction (PPI) network. Our method employs a unique bi-graph architecture based on graph neural networks (GNNs), enabling the joint representation of gene expression and PPI network data. This approach models gene-to-gene relationships under specific biological contexts and refines cell-cell relations using an attention mechanism, resulting in new gene and cell embeddings. We provide comprehensive evaluations to demonstrate the effectiveness of our method.

![Overview of the scNET Method](https://raw.githubusercontent.com/madilabcode/scNET/bb9385a9945e34e1e2500c8173baf5c8ece91f79/images/scNET.jpg)
## Download via PIP
`pip install scnet`

## Download via git
To clone the repository, use the following command:
`git clone https://github.com/madilabcode/scNET`

We recommend using the provided Conda environment located at ./Data/scNET-env.yaml.
cd scNET
conda env create -f ./Data/scNET-env.yaml

## import scNET
`import scNET`

## API
To train scNET on scRNA-seq data, first load an AnnData object using Scanpy, then initialize training with the following command:

`scNET.run_scNET(obj, pre_processing_flag=False, human_flag=False, number_of_batches=3, split_cells= True, max_epoch=250, model_name = project_name)`

with the following args:

* **obj (AnnData, optional)**: AnnData obj.

* **pre_processing_flag (bool, optional)**: If True, perform pre-processing steps.

* **human_flag (bool, optional)**: Controls gene name casing in the network.

* **number_of_batches (int, optional)**: Number of mini-batches for the training.

* **split_cells (bool, optional)**: If True, split by cells instead of edges during training.

* **n_neighbors (int, optional)**: Number of neighbors for building the adjacency graph.

* **max_epoch (int, optional)**: Max number of epochs for model training.

* **model_name (str, optional)**: Identifier for saving the model outputs.

* **save_model_flag (bool, optional)**: If True, save the trained model.


### Retrieve embeddings and model outputs with:

`embedded_genes, embedded_cells, node_features , out_features =  scNET.load_embeddings(project_name)`

where:
* **embedded_genes (np.ndarray)**: Learned gene embeddings.
  
* **embedded_cells (np.ndarray)**: Learned cell embeddings.
  
* **node_features (pd.DataFrame)**: Original gene expression matrix.
  
* **out_features (np.ndarray)**: Reconstructed gene expression matrix
  

### Create a new AnnData object using model outputs:

`recon_obj = scNET.create_reconstructed_obj(node_features, out_features, obj)`

### Construct a co-embedded network using the gene embeddings:
`scNET.build_co_embeded_network(embedded_genes, node_features)`
## Tutorial

For a basic usage example of our framework, please refer to the following notebook:
[scNET Example Notebook](https://colab.research.google.com/github/madilabcode/scNET/blob/main/scNET.ipynb)

