from scipy.stats import spearmanr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import torch 
from sklearn.cluster import KMeans
#import umap.plot
import umap.umap_
import networkx as nx 
from networkx.algorithms import community
import networkx.algorithms.community as nx_comm
from sklearn.metrics import precision_recall_curve, auc
import Utils as ut 
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
    embeded_genes = ut.load_obj(r"./Embedding/row_embedding_" + proj_name)
    embeded_cells = ut.load_obj(r"./Embedding/col_embedding_" + proj_name)
    node_features = pd.read_csv(r"./Embedding/node_features_" + proj_name,index_col=0)
    out_features = ut.load_obj(r"./Embedding/out_features_" + proj_name)
    return embeded_genes, embeded_cells, node_features, out_features


def create_reconstructed_obj(node_features, out_features, orignal_obj=None):
  embd = pd.DataFrame(out_features,index=node_features.columns[:out_features.shape[0]], columns=node_features.index)

  embd = (embd - embd.min()) / (embd.max() - embd.min())

  adata = sc.AnnData(embd)
  if not orignal_obj is None:
    adata.obs = orignal_obj.obs[:embd.shape[0]]

  sc.tl.pca(adata, svd_solver='arpack')
  sc.pp.neighbors(adata, n_neighbors=10, n_pcs=15)
  sc.tl.leiden(adata,resolution=0.6)
  sc.tl.umap(adata)
  return adata


def cal_marker_gene_aupr(adata, marker_genes=['Cd4', 'Cd8a', 'Cd14',"P2ry12","Ncr1"]\
                              , cell_types=[['CD4 Tcells'], ["CD8 Tcells","NK"], ['Macrophages'], ['Microglia'],["NK"]]):
  for marker_gene, cell_type in zip(marker_genes, cell_types):
      gene_expression = adata[:, marker_gene].X.toarray().flatten()
      binary_labels = (adata.obs["Cell Type"].isin(cell_type)).astype(int)

      precision, recall, _ = precision_recall_curve(binary_labels, gene_expression)
      aupr = auc(recall, precision)

      print(f"AUPR for {marker_gene} in identifying {cell_type[0]}: {aupr:.4f}")


def pathway_enricment(adata, groupby="seurat_clusters", groups=None):
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
                        background=list(adata.var.index) + ["t1"],# You can change this to other gene sets
                        organism='Human',  # Specify organism as Mouse
                        outdir=None)
      except:
        continue

      significant = enr.results[enr.results['Adjusted P-value'] < significance_threshold]

      enrichment_results[group] = enr.results
      significant_pathways[group] = significant[['Term', 'Adjusted P-value']]

#  for group, pathways in significant_pathways.items():
#      print(f"Significant pathways for cluster {group}:")
#      print(pathways)
#      print("\n")

  return de_genes_per_group, significant_pathways, filtered_kegg , enrichment_results


def plot_de_pathways(significant_pathways,enrichment_results):
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
    #sc.pp.pca(obj)
    sc.pp.neighbors(obj, n_neighbors=12)
    sc.tl.leiden(obj)
    sc.tl.umap(obj)
    sc.pl.umap(obj, color="cluster",palette=cp)
    return means_embedd.labels_ 


def build_co_embeded_network(embedded_rows,node_fetures, threshold=99):
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


def run_p_enr_on_range_clusterts(embd, refs,all_genes, min_n_clusters=20,max_n_clusters=80):
    results_embedd = []
    results_refs = [[] for ref in refs]
    GO_custom = gp.get_library("GO_Biological_Process_2021")

    GO_custom = {key: list(filter(lambda x: x in all_genes, value)) for key, value in GO_custom.items()}
    GO_custom = {key: value for key, value in GO_custom.items() if len(value) > 0 }

    for k in range(min_n_clusters,max_n_clusters,10):
        means_embedd = KMeans(n_clusters=k, random_state=42).fit(embd)
        results_embedd.append(p_cluster_enr(means_embedd.labels_, all_genes, GO_custom) / k)
        for i in  range(len(refs)):
            means_ref = KMeans(n_clusters=k, random_state=42).fit(refs[i])
            results_refs[i].append(p_cluster_enr(means_ref.labels_, all_genes, GO_custom) / k)
    
    k = [ i for i in range(min_n_clusters,max_n_clusters,10)]
    df = pd.DataFrame({"n_clusters":k + k, "Percentage of GO significant enriched clusters": results_embedd+ list(results_refs[0])  , "type": ["scNET" for i in k] + ["Counts" for i in k]})#+ ["scLINE" for i in k]})
    plt.figure(figsize=(12, 10),  dpi = 600)
    sns.set_context("paper", font_scale=2.5)   
    ax = sns.barplot(x="n_clusters", y="Percentage of GO significant enriched clusters", hue="type", data=df,palette=["darkturquoise", "lightsalmon"] )
    plt.show()
    return results_embedd, results_refs


def crate_kegg_annot(all_genes):
    KEGG_custom = gp.get_library("KEGG_2021_Human")
    filtered_kegg = {pathway: [gene for gene in genes if gene in all_genes]
                  for pathway, genes in KEGG_custom.items()}
    array = [ (gene, key) for key in filtered_kegg for gene in filtered_kegg[key] ]
    kegg_df = pd.DataFrame(array)
    df = pd.DataFrame(0, index=all_genes, columns=filtered_kegg.keys())

    # Iterate through the dictionary to update the DataFrame
    for key, values in filtered_kegg.items():
        for value in values:
            df.loc[value, key] = 1
    
    return df
    
def calc_roc(pred, vec, test_vec):
    pred_test = list(map(lambda x: pred[list(vec.index).index(x)], test_vec.index))
    return roc_auc_score(test_vec.values, pred_test)

def calculate_aupr(pred, vec, test_vec):
    pred_test = list(map(lambda x: pred[list(vec.index).index(x)], test_vec.index))
    return average_precision_score(test_vec.values, pred_test)

def make_term_predication(graphs, term_vec):
    train_vec = term_vec.sample(frac=0.7)
    test_vec = term_vec[~term_vec.index.isin(train_vec.index)]
    test_pos = test_vec[test_vec == 1]
    test_neg = test_vec[test_vec == 0].sample(test_pos.shape[0])
    test_vec = test_vec[list(test_pos.index) + list(test_neg.index)]
    vec = term_vec.copy()
    vec *= list(map(lambda x:  train_vec[x] if x in train_vec.index else float(0), vec.index))
    results_roc = []
    result_aupr = []
    for graph in graphs:
        w = nx.to_pandas_adjacency(graph)
        w = w.loc[term_vec.index, term_vec.index]
        train_vec = vec.copy()
        pred = ut.propagation(train_vec.values, w)
        results_roc.append([calc_roc(pred , term_vec, test_vec)])
        result_aupr.append([calculate_aupr(pred , term_vec, test_vec)])
    return result_aupr

def predict_kegg(gene_embedding, ref):
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