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
import Utils as ut 
import gseapy as gp
import os

from sklearn.metrics import average_precision_score, roc_auc_score, adjusted_rand_score
import scanpy as sc

import gseapy as gp


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
    embeded_genes = ut.load_obj(r"./Embedding/row_embedding" + proj_name)
    embeded_cells = ut.load_obj(r"./Embedding/col_embedding" + proj_name)
    node_feature = pd.read_csv(r"./Embedding/node_features" + proj_name,index_col=0)
    return embeded_genes, embeded_cells, node_feature

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
    KEGG_custom = {key:list(filter(lambda x: x in all_genes, value)) for key, value in KEGG_custom.items()}
    array = [ (gene, key) for key in KEGG_custom for gene in KEGG_custom[key] ]
    kegg_df = pd.DataFrame(array)
    df = pd.DataFrame(0, index=all_genes, columns=KEGG_custom.keys())

    # Iterate through the dictionary to update the DataFrame
    for key, values in KEGG_custom.items():
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
    vec = term_vec.copy()
    vec *= list(map(lambda x:  train_vec[x] if x in train_vec.index else float(0), vec.index))
    results = []
    for graph in graphs:
        w = nx.to_pandas_adjacency(graph)
        w = w.loc[term_vec.index, term_vec.index]
        train_vec = vec.copy()
        pred = ut.propagation(train_vec.values, w)
        results.append([calc_roc(pred , term_vec, test_vec)])
    return results

def predict_kegg(gene_embedding, ref):
    annot = crate_kegg_annot(ref.index)
    annot_threshold = annot.sum()>=40
    annot_threshold = annot_threshold[annot_threshold == True].sort_values(ascending=False).head(50)
    graph_embedded,_ = build_co_embeded_network(gene_embedding,ref)
    graph_ref,_ =build_co_embeded_network(ref,ref)
    kegg_pred = [make_term_predication([graph_embedded,graph_ref], annot[term]) for term in annot_threshold.index]
   
    kegg_pred = np.array(kegg_pred).squeeze()
    df = pd.DataFrame({"ROCAUC" : kegg_pred.T.reshape(-1), "Method": ["scNET" for i in range(kegg_pred.shape[0])]  +  ["Counts" for i in range(kegg_pred.shape[0])]})

    fig, ax = plt.subplots(figsize=[10,7])
    fig.set_dpi(600)

    custom_palette =  ['darkturquoise', 'lightsalmon']

    sns.boxenplot(ax=ax, data=df,x="Method", y="ROCAUC", palette=custom_palette)    
    sns.set_theme(style='white',font_scale=1.5)
    plt.show()
    return df

def plot_umap_cells(cell_embedding):    
    obj = sc.AnnData(cell_embedding)
    sc.pp.neighbors(obj, n_neighbors=12)
    sc.tl.leiden(obj)
    sc.tl.umap(obj)
    sc.pl.umap(obj, color="leiden",palette=cp)