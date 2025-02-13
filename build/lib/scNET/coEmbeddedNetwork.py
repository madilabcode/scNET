from scipy.stats import spearmanr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import torch 
from sklearn.cluster import KMeans
#import umap.plot
import networkx as nx 
from networkx.algorithms import community
import networkx.algorithms.community as nx_comm
from sklearn.metrics import precision_recall_curve, auc
import scNET.Utils as ut 
import gseapy as gp
import os

from sklearn.metrics import average_precision_score, roc_auc_score, adjusted_rand_score
import scanpy as sc

import gseapy as gp
import warnings
warnings.filterwarnings('ignore')


device = torch.device("cpu")
cp = {
  '0': '#1f77b4',
  '1': '#aec7e8',
  '2': '#ff7f0e',
  '3': '#ffbb78',
  '4': '#2ca02c',
  '5': '#98df8a',
  '6': '#d62728',
  '7': '#ff9896',
  '8': '#9467bd',
  '9': '#c5b0d5',
  '10': '#8c564b',
  '11': '#c49c94',
  '12': '#e377c2',
  '13': '#f7b6d2',
  '14': '#7f7f7f',
  '15': '#c7c7c7',
  '16': '#bcbd22',
  '17': '#dbdb8d',
  '18': '#17becf',
  '19': '#9edae5',
  '20': '#1f77b4',
  '21': '#ff7f0e',
  '22': '#2ca02c',
  '23': '#d62728',
  '24': '#9467bd',
  '25': '#8c564b',
  '26': '#e377c2',
  '27': '#7f7f7f',
  '28': '#bcbd22',
  '29': '#17becf'
}

def load_embeddings(proj_name):
    '''
    Loads the embeddings and gene expression data for a given project.

    Args:
        proj_name (str): The name of the project.

    Returns:
        tuple: A tuple containing:
            - embedded_genes (np.ndarray): Learned gene embeddings.
            - embedded_cells (np.ndarray): Learned cell embeddings.
            - node_features (pd.DataFrame): Original gene expression matrix.
            - out_features (np.ndarray): Reconstructed gene expression matrix.
    '''
    embeded_genes = ut.load_obj(r"./Embedding/row_embedding_" + proj_name)
    embeded_cells = ut.load_obj(r"./Embedding/col_embedding_" + proj_name)
    node_features = pd.read_csv(r"./Embedding/node_features_" + proj_name,index_col=0)
    out_features = ut.load_obj(r"./Embedding/out_features_" + proj_name)
    return embeded_genes, embeded_cells, node_features, out_features


def create_reconstructed_obj(node_features, out_features, orignal_obj=None):
  '''
    Creates an AnnData object from reconstructed gene expression data, normalizes it, and computes PCA, neighbors, clustering, and UMAP.

    Args:
        node_features (pd.DataFrame): The original gene expression matrix with genes as columns and cells as rows.
        out_features (np.ndarray): The reconstructed gene expression matrix.
        original_obj (AnnData, optional): The original AnnData object, if available, to copy cell metadata (obs) from. Defaults to None.

    Returns:
        AnnData: An AnnData object containing the reconstructed gene expression data with PCA, neighbors, Leiden clustering, and UMAP embeddings computed.
    '''
  embd = pd.DataFrame(out_features,index=node_features.columns[:out_features.shape[0]], columns=node_features.index)

  embd = (embd - embd.min()) / (embd.max() - embd.min())

  adata = sc.AnnData(embd)
  if not orignal_obj is None:
    adata.obs = orignal_obj.obs[:embd.shape[0]]

  sc.tl.pca(adata, svd_solver='arpack')
  sc.pp.neighbors(adata, n_neighbors=10, n_pcs=15)
  sc.tl.leiden(adata,resolution=0.5)
  sc.tl.umap(adata)
  return adata


def calculate_marker_gene_aupr(adata, marker_genes=['Cd4','Cd14',"P2ry12","Ncr1"]\
                              , cell_types=[['CD4 Tcells'], ['Macrophages'], ['Microglia'],["NK"]]):
  '''
    Calculates the Area Under the Precision-Recall curve (AUPR) for specified marker genes in identifying specific cell types.

    Args:
        adata (AnnData): The annotated data matrix (AnnData object) containing gene expression data and cell type information.
        marker_genes (list of str, optional): A list of marker genes to evaluate. Defaults to ['Cd4', 'Cd8a', 'Cd14', 'P2ry12', 'Ncr1'].
        cell_types (list of lists, optional): A list of lists where each sublist contains the cell types associated with the corresponding marker gene. Defaults to [['CD4 Tcells'], ['CD8 Tcells', 'NK'], ['Macrophages'], ['Microglia'], ['NK']].

    Returns:
        None: The function prints the AUPR score for each marker gene and the corresponding cell type(s).
        
  '''                              
  for marker_gene, cell_type in zip(marker_genes, cell_types):
      gene_expression = adata[:, marker_gene].X.toarray().flatten()
      binary_labels = (adata.obs["Cell Type"].isin(cell_type)).astype(int)

      precision, recall, _ = precision_recall_curve(binary_labels, gene_expression)
      aupr = auc(recall, precision)

      print(f"AUPR for {marker_gene} in identifying {cell_type[0]}: {aupr:.4f}")


def pathway_enricment(adata, groupby="seurat_clusters", groups=None):
  '''
    Performs pathway enrichment analysis using KEGG pathways for differentially expressed genes in specific groups.

    Args:
        adata (AnnData): The annotated data matrix (AnnData object) containing gene expression data and cell clustering/grouping information.
        groupby (str, optional): The key in `adata.obs` to group cells by for differential expression analysis. Defaults to "seurat_clusters".
        groups (list, optional): A list of specific groups (clusters or cell types) to analyze. If None, all unique groups in `adata.obs[groupby]` are used. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - de_genes_per_group (dict): A dictionary where keys are group names and values are DataFrames of differentially expressed genes for each group.
            - significant_pathways (dict): A dictionary where keys are group names and values are DataFrames of significant KEGG pathways with adjusted p-values for each group.
            - filtered_kegg (dict): A dictionary of KEGG pathways filtered for genes present in the dataset.
            - enrichment_results (dict): A dictionary where keys are group names and values are full enrichment results from Enrichr for each group.

    Method:
        - The function retrieves KEGG pathways and filters them based on the genes present in `adata`.
        - Differentially expressed (DE) genes are identified for each group using the Wilcoxon rank-sum test.
        - Pathway enrichment analysis is performed using Enrichr, based on the DE genes.
        - Pathways with adjusted p-values below 0.05 are considered significant.
  '''
  adata.var.index = adata.var.index.str.upper()
  kegg_gene_sets = gp.get_library('KEGG_2021_Human')

  filtered_kegg = {pathway: [gene for gene in genes if gene in adata.var.index]
                  for pathway, genes in kegg_gene_sets.items()}

  filtered_kegg = {pathway: genes + ["t1"] for pathway, genes in filtered_kegg.items() if len(genes) > 0}


  if groups is None:
    groups = adata.obs[groupby].unique()

  sc.tl.rank_genes_groups(adata, groupby=groupby, method='wilcoxon')

  de_genes_per_group = {}
  for group in groups:
      dedf = sc.get.rank_genes_groups_df(adata, group=group)
      dedf.names = dedf.names.str.upper()
      genes = dedf[(dedf['logfoldchanges'] > 0) & (dedf["pvals_adj"] <  0.05)]
      de_genes_per_group[group] = dedf[(dedf['logfoldchanges'] > 0) & (dedf["pvals_adj"] <  0.05)]

  enrichment_results = {}
  significant_pathways = {}
  significance_threshold = 0.05

  for group, genes in de_genes_per_group.items():

      try:
        genes = genes['names'].values
        enr = gp.enrichr(gene_list=(genes.tolist() + ["t1"]),
                        gene_sets=filtered_kegg,
                        background=list(adata.var.index) + ["t1"],
                        organism='Human',  
                        outdir=None)
      except:
        continue

      significant = enr.results[enr.results['Adjusted P-value'] < significance_threshold]

      enrichment_results[group] = enr.results
      significant_pathways[group] = significant[['Term', 'Adjusted P-value']]


  return de_genes_per_group, significant_pathways, filtered_kegg , enrichment_results


def plot_de_pathways(significant_pathways,enrichment_results):
  '''
    Plots a heatmap of the -log10(Adjusted P-value) for significant pathways across multiple datasets.

    Args:
        significant_pathways (dict): A dictionary where keys are dataset names (or groups), and values are DataFrames containing significant pathways and their adjusted p-values.
        enrichment_results (dict): A dictionary where keys are dataset names (or groups), and values are DataFrames containing full pathway enrichment results, including adjusted p-values for each pathway.

    Returns:
        None: The function generates and displays a heatmap showing the significance (-log10(Adjusted P-value)) of the top 20 pathways across different datasets.

  '''

  data_dict = significant_pathways
  combined_df = pd.DataFrame()

  for _, df in enrichment_results.items():
      top5_df = df.sort_values(by='Adjusted P-value').head(20)
      for dataset_name, df2 in enrichment_results.items():
        df2 = df2.loc[df2.Term.isin(top5_df.Term)]
        df2['Dataset'] = dataset_name
        combined_df = pd.concat([combined_df, df2])

  combined_df['Unique Term'] = combined_df['Term']

  combined_df['-log10(Adjusted P-value)'] = -np.log(combined_df['Adjusted P-value'])

  # Pivot the data to make a matrix suitable for a heatmap
  pivot_df = combined_df.drop_duplicates().pivot(index="Unique Term", columns="Dataset", values="-log10(Adjusted P-value)")
  pivot_df.fillna(0,inplace=True)
  plt.figure(figsize=(10, 30))
  g = sns.clustermap(pivot_df, annot=False, cmap="YlGnBu", linewidths=.5,figsize=(15,25))
  plt.title('Heatmap of Pathway Significance by Dataset', fontsize=18)
  g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), fontsize=15,rotation=45)
  g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=15)
  g.ax_heatmap.set_xlabel('Dataset', fontsize=15)
  g.ax_heatmap.set_ylabel('Pathway Term', fontsize=15)
  plt.tight_layout()


def plot_gene_umap_clustring(embedded_rows):
    means_embedd = KMeans(n_clusters=20, random_state=42).fit(embedded_rows)
    obj = sc.AnnData(embedded_rows)
    obj.obs["cluster"] = means_embedd.labels_
    obj.obs["cluster"] = obj.obs.cluster.astype(str)
    sc.pp.neighbors(obj, n_neighbors=12)
    sc.tl.leiden(obj)
    sc.tl.umap(obj)
    sc.pl.umap(obj, color="cluster",palette=cp)
    return means_embedd.labels_ 


def build_co_embeded_network(embedded_rows,node_fetures, threshold=99):
    '''
    Builds a co-embedded network from the given embedded rows using a correlation-based thresholding approach and detects communities using the Louvain algorithm.

    Args:
        embedded_rows (np.ndarray): A matrix of embeddings (e.g., gene embeddings) where each row corresponds to an entity (e.g., gene or cell).
        node_fetures (pd.DataFrame): A DataFrame containing features or identifiers for the nodes, where the index corresponds to the entities in `embedded_rows`.
        threshold (int, optional): The percentile threshold to use when binarizing the correlation matrix. Defaults to 99.

    Returns:
        tuple: A tuple containing:
            - graph (networkx.Graph): The co-embedded network created from the thresholded correlation matrix.
            - mod (float): The modularity score of the network, indicating the strength of the community structure.

    Method:
        - Computes the absolute Pearson correlation matrix between the rows of `embedded_rows`.
        - Applies a threshold (specified by the `threshold` percentile) to binarize the correlation matrix.
        - Constructs a graph where nodes are connected if their correlation exceeds the threshold.
        - Applies the Louvain algorithm to detect communities within the graph.
        - Calculates the modularity score for the detected communities.
        - Relabels the graph nodes using the identifiers from `node_fetures`.
   '''
    corr = np.corrcoef(embedded_rows)
    corr = np.abs(corr)  
    np.fill_diagonal(corr,0)
    mat = (np.abs(corr) > np.percentile(corr, threshold)).astype(np.int64)
    graph = nx.from_numpy_array(mat)
    comm = nx_comm.louvain_communities(graph,resolution=1, seed=42)
    mod = nx_comm.modularity(graph, comm)
    map_nodes = {list(graph.nodes)[i]:node_fetures.index[i] for i in range(len(node_fetures.index))}
    graph = nx.relabel_nodes(graph,map_nodes)
    return graph, mod 

def p_cluster_enr(clusters,all_genes, GO_custom):
    '''
    Calculates the number of clusters that are significantly enriched for Gene Ontology (GO) terms.

    Args:
        clusters (array-like): An array or list of cluster labels for each gene.
        all_genes (array-like): A list of all gene names corresponding to the genes clustered.
        GO_custom (dict): A dictionary of GO terms and their associated genes. Keys are GO term names, and values are lists of genes associated with each GO term.

    Returns:
        int: The number of clusters with significant GO term enrichment (Adjusted P-value ≤ 0.05).

    Method:
        - Creates a DataFrame associating each gene with its cluster.
        - Iterates over each unique cluster:
            - If the cluster contains at least 5 genes, performs GO term enrichment analysis using Enrichr.
            - Stores the minimum Adjusted P-value from the enrichment results.
        - Determines which clusters have a minimum Adjusted P-value ≤ 0.05.
        - Returns the count of such clusters.
    '''
    df =  pd.DataFrame({"genes" : all_genes, "cluster": clusters})
    results = []
    for cluster in df.cluster.drop_duplicates():
        genes = df.loc[df.cluster == cluster,"genes"]
        if len(genes) >= 5:
            enr = gp.enrichr(gene_list=list(genes), # or "./tests/data/gene_list.txt",
                        gene_sets=GO_custom,
                        background=all_genes, #"hsapiens_gene_ensembl",
                        outdir=None,
                        verbose=True
                    )
            results.append(enr.results["Adjusted P-value"].min())
        else:
            results.append(1)
        
    results = np.array(results) <= 0.05
    return results.sum()


def run_p_enr_on_range_clusterts(embd,refs, min_n_clusters=60,max_n_clusters=130):
    '''
    Runs GO term enrichment analysis over a range of cluster counts and plots the percentage of clusters that are significantly enriched.

    Args:
        embd (np.ndarray or pd.DataFrame): The embedding matrix (e.g., gene embeddings) where rows correspond to genes.
        refs (list of pd.DataFrame): A list of reference expression matrices (e.g., raw counts or other reference data) for comparison.
        min_n_clusters (int, optional): The minimum number of clusters to evaluate. Defaults to 60.
        max_n_clusters (int, optional): The maximum number of clusters to evaluate. Defaults to 130.

    Returns:
        tuple:
            - results_embedd (list): A list containing the proportion of significantly enriched clusters for each cluster count in `embd`.
            - results_refs (list of lists): A list where each element is a list containing the proportion of significantly enriched clusters for each cluster count in the corresponding reference in `refs`.

    Method:
        - Converts gene names to uppercase for consistency.
        - Retrieves the GO Biological Process 2021 gene sets and filters them to include only genes present in the dataset.
        - For a range of cluster counts (from `min_n_clusters` to `max_n_clusters` in steps of 10):
            - Clusters the embeddings and references using KMeans clustering.
            - Calculates the number of clusters with significant GO term enrichment using `p_cluster_enr`.
            - Stores the proportion of significantly enriched clusters (number of significant clusters divided by total clusters).
        - Plots a bar chart showing the percentage of significantly enriched clusters versus the number of clusters for both embeddings and references.

    '''
    results_embedd = []
    exp_genes = (refs[0] > 0).sum(axis=1) > refs[0].shape[1] * 0.1

    all_genes = refs[0].loc[exp_genes].index

    all_genes = list(map(lambda x: x.upper(), all_genes))
    results_refs = [[] for ref in refs]
    GO_custom = gp.get_library("GO_Biological_Process_2021")
    
    GO_custom = {key: list(filter(lambda x: x in all_genes, value)) for key, value in GO_custom.items()}
    GO_custom = {key: value for key, value in GO_custom.items() if len(value) > 0 }

    for k in range(min_n_clusters,max_n_clusters,10):
        means_embedd = KMeans(n_clusters=k, random_state=42).fit(embd[exp_genes.values])
        results_embedd.append(p_cluster_enr(means_embedd.labels_, all_genes, GO_custom) / k)
        for i in  range(len(refs)):
            means_ref = KMeans(n_clusters=k, random_state=42).fit(refs[i].loc[exp_genes])
            results_refs[i].append(p_cluster_enr(means_ref.labels_, all_genes, GO_custom) / k)
    
    k = [ i for i in range(min_n_clusters,max_n_clusters,10)]
    df = pd.DataFrame({"n_clusters":k + k, "Percentage of GO significant enriched clusters": results_embedd+ list(results_refs[0])  , "type": ["scNET" for i in k] + ["Counts" for i in k]})#+ ["scLINE" for i in k]})
    plt.figure(figsize=(12, 10),  dpi = 600)
    sns.set_context("paper", font_scale=2.5)   
    ax = sns.barplot(x="n_clusters", y="Percentage of GO significant enriched clusters", hue="type", data=df,palette=["darkturquoise", "lightsalmon"] )
    plt.show()
    return results_embedd, results_refs


def crate_kegg_annot(all_genes):
    '''
      Creates a binary annotation matrix for KEGG pathways, indicating gene-pathway memberships for a given set of genes.

      Args:
          all_genes (list): A list of genes to be annotated with KEGG pathway membership.

      Returns:
          pd.DataFrame: A binary DataFrame where rows correspond to genes and columns to KEGG pathways. A value of 1 indicates that a gene is part of the pathway, and 0 otherwise.
    '''
    KEGG_custom = gp.get_library("KEGG_2021_Human")
    filtered_kegg = {pathway: [gene for gene in genes if gene in all_genes]
                  for pathway, genes in KEGG_custom.items()}
    array = [ (gene, key) for key in filtered_kegg for gene in filtered_kegg[key] ]
    kegg_df = pd.DataFrame(array)
    df = pd.DataFrame(0, index=all_genes, columns=filtered_kegg.keys())

    for key, values in filtered_kegg.items():
        for value in values:
            df.loc[value, key] = 1
    
    return df
    
def calculate_aupr(pred, vec, test_vec):
    pred_test = list(map(lambda x: pred[list(vec.index).index(x)], test_vec.index))
    return average_precision_score(test_vec.values, pred_test)

def make_term_predication(graphs, term_vec):
    '''
    Propagates gene-term predictions across multiple graphs and evaluates their performance using AUPR.

    Args:
        graphs (list of networkx.Graph): A list of graphs (e.g., gene co-expression networks) to propagate term information.
        term_vec (pd.Series): A binary vector indicating whether each gene is associated with a specific KEGG term.

    Returns:
        list: A list of AUPR scores for each graph's predictions.

    Method:
        - Splits the term vector into training and testing sets.
        - Uses propagation on each graph to predict term associations for genes.
        - Evaluates predictions using AUPR.
    '''
    train_vec = term_vec.sample(frac=0.7)
    test_vec = term_vec[~term_vec.index.isin(train_vec.index)]
    test_pos = test_vec[test_vec == 1]
    test_neg = test_vec[test_vec == 0].sample(test_pos.shape[0])
    test_vec = test_vec[list(test_pos.index) + list(test_neg.index)]
    vec = term_vec.copy()
    vec *= list(map(lambda x:  train_vec[x] if x in train_vec.index else float(0), vec.index))
    result_aupr = []
    for graph in graphs:
        w = nx.to_pandas_adjacency(graph)
        w = w.loc[term_vec.index, term_vec.index]
        train_vec = vec.copy()
        pred = ut.propagation(train_vec.values, w)
        result_aupr.append([calculate_aupr(pred , term_vec, test_vec)])
    return result_aupr

def predict_kegg(gene_embedding, ref):
    '''
    Predicts KEGG pathway memberships using gene embeddings and reference data, and evaluates the performance using AUPR.

    Args:
        gene_embedding (np.ndarray): The matrix of gene embeddings.
        ref (pd.DataFrame): A reference dataset containing gene expression or other relevant features.

    Returns:
        pd.DataFrame: A DataFrame containing the AUPR scores for predictions from the gene embeddings and reference data.

    Method:
        - Annotates genes with KEGG pathway memberships using `crate_kegg_annot`.
        - Filters KEGG pathways to include those with at least 40 gene members.
        - Constructs co-embedded networks from both the embeddings and reference data.
        - Uses propagation to predict pathway memberships for each graph.
        - Evaluates the predictions using AUPR and plots the results.
    '''
    ref.index = list(map(lambda x: x.upper(),ref.index))
    annot = crate_kegg_annot(ref.index)
    annot_threshold = annot.sum()>=40
    annot_threshold = annot_threshold[annot_threshold == True].sort_values(ascending=False).head(50)
    graph_embedded,_ = build_co_embeded_network(gene_embedding,ref)
    graph_ref,_ =build_co_embeded_network(ref,ref)
    kegg_pred = [make_term_predication([graph_embedded,graph_ref], annot[term]) for term in annot_threshold.index]
   
    kegg_pred = np.array(kegg_pred).squeeze()
    df = pd.DataFrame({"AUPR" : kegg_pred.T.reshape(-1), "Method": ["scNET" for i in range(kegg_pred.shape[0])]  +  ["Counts" for i in range(kegg_pred.shape[0])]})

    fig, ax = plt.subplots(figsize=[10,7])
    fig.set_dpi(600)

    custom_palette =  ['darkturquoise', 'lightsalmon']

    sns.boxenplot(ax=ax, data=df,x="Method", y="AUPR", palette=custom_palette)    
    sns.set_theme(style='white',font_scale=1.5)
    plt.show()
    return df

def plot_umap_cells(cell_embedding):    
    obj = sc.AnnData(cell_embedding)
    sc.pp.neighbors(obj, n_neighbors=12)
    sc.tl.leiden(obj)
    sc.tl.umap(obj)
    sc.pl.umap(obj, color="leiden",palette=cp)