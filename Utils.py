import torch
import numpy as np
import pandas as pd 
import scanpy as sc
import networkx as nx
from scipy.stats import ranksums, ttest_ind, spearmanr
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import average_precision_score
import pickle 
from MultyGraphModel import BiGraphAutoEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch_geometric.utils import  convert
import mygene
import os 
import random
os.environ["R_HOME"] = r"C:\Program Files\R\R-4.2.1"
os.environ['path'] += r";C:\Program Files\R\R-4.2.1"
from rpy2.robjects import r
from rpy2 import robjects as ro
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import default_converter
from sklearn.metrics import adjusted_rand_score
import rpy2.robjects.pandas2ri as rpyp
from sklearn.metrics import average_precision_score, roc_auc_score, adjusted_rand_score

r('''library(EGAD)
data(GO.human)''')

alpha  = 0.9
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
epsilon = 0.0001

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def normW(W):
    sum_rows = pd.DataFrame(W.sum(axis=1)) + epsilon
    sum_rows = sum_rows @ sum_rows.T
    sum_rows **= 1/2
    return W / sum_rows



def calculate_propagation_matrix(W, epsilon = 0.0001):
   # device = torch.device("cpu")
    S =  []
    W = normW(W)
    W = torch.tensor(W.values).to(device)
    for index in range(W.shape[0]):
        y = torch.zeros(W.shape[0],dtype=torch.float32).to(device)
        y[index] = 1
        f = y.clone()
        flag = True

        while(flag):
            next_f = (alpha*(W@f) + (1-alpha)*y).to(device)
        
            if torch.linalg.norm(next_f - f) <= epsilon:
                flag = False
            else:
              #  print(torch.linalg.norm(next_f - f))
                f = next_f
        S.append(f)
    return torch.concat(S).view(W.shape)

def propagate_all_genes(W,exp):
    S = calculate_propagation_matrix(W)
    prop_exp = torch.tensor(exp.values).to(device).T
    prop_exp = S @ prop_exp
    prop_norm = S @ torch.ones_like(prop_exp)
    prop_exp /= prop_norm
    prop_exp = pd.DataFrame(prop_exp.T.detach().cpu().numpy(),index = exp.index, columns = exp.columns)
    return prop_exp

def one_step_propagation(W,F):
    W = torch.tensor(normW(W).values, dtype= torch.float32)
    F = torch.tensor(F,dtype= torch.float32)
    prop_exp = (alpha)*W@F + (1-alpha)*F
    prop_norm  = (alpha)*W@torch.ones_like(F) + (1-alpha)*torch.ones_like(F)
    return prop_exp/prop_norm

def add_noise(obj,alpha = 0.0, drop_out = False):
    obj_noise = obj.raw.to_adata()
    #obj_noise.X = (1-alpha) *obj_noise.X + alpha*np.random.randn(*obj.X.shape)
    if drop_out:
        obj_noise.X = obj_noise.X * np.random.binomial(1,(1-alpha),obj.X.shape)
    else:
        obj_noise.X = ((1-alpha) *obj_noise.X + alpha*np.random.randn(*obj.X.shape)).astype(np.float32)
    obj_noise.var["highly_variable"] = True    
    sc.tl.pca(obj_noise, svd_solver='arpack',use_highly_variable = False)
    sc.pp.neighbors(obj_noise,n_pcs=20, n_neighbors=50)
    obj_noise.raw = obj_noise

    return obj_noise

def wilcoxon_enrcment_test(up_sig, down_sig, exp):
    gene_exp = exp.loc[exp.index.isin(up_sig)]
    if down_sig is None:     
        backround_exp = exp.loc[~exp.index.isin(up_sig)]
    else:
        backround_exp = exp.loc[exp.index.isin(down_sig)]
        
    rank = ranksums(backround_exp,gene_exp,alternative="less")[1] # rank expression of up sig higher than backround
    rank = 1 if rank > 0.05 else rank 
    return -1 * np.log(rank)


# ---------------------------
# calculates the signature of the data
#
# returns scores vector of signature calculated per cell
# ---------------------------
def signature_values(exp, up_sig, down_sig=None):
    up_sig = pd.DataFrame(up_sig).squeeze()
    # first letter of gene in upper case
    up_sig = up_sig.apply(lambda x: x[0].upper() + x[1:].lower())
    # keep genes in sig that appear in exp data
    up_sig = up_sig[up_sig.isin(exp.index)]

    if down_sig is not None:
        down_sig = pd.DataFrame(down_sig).squeeze()
        down_sig = down_sig.apply(lambda x: x[0].upper() + x[1:].lower())
        down_sig = down_sig[down_sig.isin(exp.index)]
    
    return exp.apply(lambda cell: wilcoxon_enrcment_test(up_sig, down_sig, cell), axis=0)

def run_signature_on_obj(obj, up_sig, down_sig=None, conn_flag =  True, umap_flag = True,  slot=None, idents=None):
    exp = obj.raw.to_adata().to_df().T
    sigs_scores = signature_values(exp, up_sig, down_sig) # wilcoxon score
    if conn_flag:
        graph = obj.obsp["connectivities"].toarray()
    else:
        graph = obj.obsp["distances"].toarray()
    sigs_scores = propagation(sigs_scores,graph)
    obj.obs["SigScore"] = sigs_scores
    # color_map = "jet"
    if umap_flag:
        sc.pl.umap(obj, color=["SigScore"],color_map="magma")
    else:
        sc.pl.tsne(obj, color=["SigScore"],color_map="magma")
    # sc.pl.embedding(obj, basis="umap", color="SigScore", color_map = "jet")
    if not slot is None:
        stat_test(obj, slot, idents)
    return obj


def stat_test(obj, slot, idents):
    sub1 = obj[obj.obs[slot] == idents[0]].obs.SigScore
    sub2 = obj[obj.obs[slot] == idents[1]].obs.SigScore

    pvalue = ttest_ind(sub1,sub2)[1]
    sns.boxenplot(obj.obs, x=slot,y="SigScore")
    plt.title(f"Signature P-Value = {round(pvalue,5)}")
    plt.show()
    

# ---------------------------
# Y - scores vector of cells
# W - Adjacency matrix
#
# f_t = alpha * (W * f_(t-1)) + (1-alpha)*Y
#
# returns f/f1
# ---------------------------
def propagation(Y, W):
    W = normW(W)
    f = np.array(Y)
    Y = np.array(Y)
   # f2 = calculate_propagation_matrix(W) @ Y

    W = np.array(W.values)
    
    Y1 = np.ones(Y.shape, dtype=np.float64)
    f1 = np.ones(Y.shape, dtype=np.float64)
    flag = True

    while(flag):
        next_f = alpha*(W@f) + (1-alpha)*Y
        next_f1 = alpha*(W@f1) + (1-alpha)*Y1
    
        if np.linalg.norm(next_f - f) <= epsilon and np.linalg.norm(next_f1 - f1) <= epsilon:
            flag = False
        else:
            #print(np.linalg.norm(next_f - f))
            #print(np.linalg.norm(next_f1 - f1))
            f = next_f
            f1 = next_f1
   # return f1,f2
    return np.array(f/f1) 

def build_network(obj, net, biogrid_flag = False, human_flag = False, random_noise=False):
    #print(len(list(pd.concat([pa.Source,pa.Target]).drop_duplicates())))
    if not biogrid_flag:
        net.columns = ["Source","Target","Conn"]
        net = net.loc[net.Conn >= 0.5]
    
    else:
         net.columns = ["Source","Target"]
    
    if not human_flag:
        net["Source"] = net["Source"].apply(lambda x: x[0] + x[1:].lower()).astype(str)
        net["Target"] = net["Target"].apply(lambda x: x[0] + x[1:].lower()).astype(str)

         
    genes = list(pd.concat([net.Source, net.Target]).drop_duplicates())
    genes =  obj.var[obj.var.index.isin(genes)].index
    node_feature = sc.get.obs_df(obj.raw.to_adata(),list(genes)).T
    node_feature["non_zero"] = node_feature.apply(lambda x: x.astype(bool).sum(), axis=1)
    node_feature = node_feature.loc[node_feature.non_zero > node_feature.shape[1] *  0.1]
    heg = (node_feature["non_zero"] /  node_feature.shape[1]) > 0.5
    print(heg.sum())
    node_feature.drop("non_zero",axis=1,inplace=True)
    #pa["Target"] = pa.Target.sample(frac=1).values
   #print( pa.Target.sample(frac=1))
    net = net.loc[net.Source.isin(node_feature.index)]
    net = net.loc[net.Target.isin(node_feature.index)]
    #if random_noise:
   #   net.Source = net.Source.sample(frac=1).values
    gp = nx.from_pandas_edgelist(net, "Source", "Target")
    node_feature = node_feature.loc[list(gp.nodes)]
    #if random_noise:
  #    node_feature = node_feature.sample(frac=1)
    #  node_feature += np.random.normal(loc=0,scale=3,size=node_feature.shape)
    return net, gp, node_feature, heg

def upload_model(model_name, obj, knn_edge_index, num_layers=2, human_flag=True,cd=False):
    import MultyGraphModel as mgm
    import imp
    imp.reload(mgm)
    net = pd.read_csv(r"./Data/format_h_sapiens.csv")[["g1_symbol","g2_symbol","conn"]]
    net, ppi, node_feature,_ = build_network(obj, net,human_flag=human_flag)
    ppi_geo = convert.from_networkx(ppi)
    ppi_edge_index = ppi_geo.edge_index
    model = mgm.BiGraphAutoEncoder(node_feature.shape[0], node_feature.shape[1], 250, 75, 250, 75,lambda_cols=50,vae_flag=False, add_linear_row=False, add_linear_col=False, num_layers=num_layers,cd=cd).to(device)
    state = torch.load(model_name)
    model.load_state_dict(state)
    model.eval()
    obj.X = obj.raw.X


    scaler = StandardScaler()
    x = node_feature.values
    x = scaler.fit_transform(x.T).T
    x = torch.tensor(x, dtype=torch.float32, device=device)
    knn_edge_index = knn_edge_index.to(device)
    ppi_edge_index = ppi_edge_index.to(device)
    embbed = model.encoder(x.to(device), knn_edge_index.to(device), ppi_edge_index.to(device))
    embbed
    embbed_rows = model.rows_encoder(embbed, ppi_edge_index.to(device))
    embbed_cols = model.cols_encoder(embbed.T, knn_edge_index)
    return model, x, node_feature, embbed_cols, embbed_rows ,ppi_edge_index



def discret_tensor(tensor, num_classes=20):

    min_vals, _ = torch.min(tensor, dim=1, keepdim=True)
    max_vals, _ = torch.max(tensor, dim=1, keepdim=True)

    # Generate equally spaced intervals for each row
    linspace = torch.linspace(0, 1, num_classes + 1).view(1, -1)
    intervals = min_vals + (max_vals - min_vals) * linspace

    # Use searchsorted to assign each element to a class
    discretized_tensor = torch.searchsorted(intervals, tensor) - 1
    discretized_tensor = torch.clamp(discretized_tensor, 0, num_classes - 1)
    return discretized_tensor




def convert_from_symbol(symbols):
  mg = mygene.MyGeneInfo()
  ginfo = mg.querymany(symbols, scopes='symbol')
  symb = {}
  for g in ginfo:
    if "entrezgene" in g:
      symb[g["query"]]=g["entrezgene"]
        
  return list(map(lambda x: symb[x], symbols))


def covert_to_symbol(exp):
  mg = mygene.MyGeneInfo()
  ens = exp.index
  ginfo = mg.querymany(ens, scopes='ensembl.gene')
  symb = {}
  for g in ginfo:
    if "symbol" in g:
      symb[g["query"]]=g["symbol"]
        
  exp = exp.loc[symb.keys()]
  exp.index = list(map(lambda x: symb[x], exp.index))
  return exp


apply_egad = r(""" function(net){
  genelist <- make_genelist(net)
  gene_network <- make_gene_network(net ,genelist)
  goterms <- unique(GO.human[,3])
  annotations <- make_annotations(GO.human[,c(2,3)],genelist,goterms)
  #return(annotations)
  GO_groups_voted <- run_GBA(gene_network, annotations, min=10)
  print(GO_groups_voted[[1]])
  return(GO_groups_voted[[1]])
}""")


make_go_annotations = r(""" function(net){
  genelist <- make_genelist(net)
  gene_network <- make_gene_network(net ,genelist)
  goterms <- unique(GO.human[,3])
  annotations <- make_annotations(GO.human[,c(2,3)],genelist,goterms)
  return(as.data.frame(annotations))
}
""")

make_annotations = r(""" function(net, annot, groups){
  genelist <- make_genelist(net)
  gene_network <- make_gene_network(net ,genelist)
  annotations <- make_annotations(annot,genelist,groups)
  return(as.data.frame(annotations))
  }
""")


def crate_anndata(path, pcs = 15,neighbors = 30):
    exp = pd.read_csv(path,index_col=0)
    #exp = pd.read_table(path, sep='\t')
    adata = sc.AnnData(exp.T)
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata.var['mt'] = adata.var_names.str.startswith('MT-')  
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    adata = adata[adata.obs.n_genes_by_counts < 6000, :]
    adata = adata[adata.obs.pct_counts_mt < 10, :]
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata.raw = adata
    sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=neighbors, n_pcs=pcs)
    sc.tl.leiden(adata)
    sc.tl.tsne(adata)
    return adata

def crate_perm(array):
    df = pd.DataFrame(array)
    df = df.apply(lambda x: pd.Series(x.values[random.sample(range(x.shape[0]),x.shape[0])]), axis=1 )
    return df.values

def permutation_test_coexpression(observed,embbed_row_co, embbed_row_cond, num_of_perm = 100):
    results = []
    for _ in range(num_of_perm):
        cor_row_embbed_cond_perm = np.abs(np.corrcoef(crate_perm(embbed_row_cond.detach().cpu().numpy())))
        cor_row_embbed_co_perm = np.abs(np.corrcoef(crate_perm(embbed_row_co.detach().cpu().numpy())))
        perm = cor_row_embbed_cond_perm - cor_row_embbed_co_perm
        results.append(perm > observed)
    
    results = sum(np.array(results))
    return (results < 0.01 * num_of_perm).astype(int)

def save_model(path, model):
    torch.save(model.state_dict(), path)

def load_model(path,node_feature,device):
    model = BiGraphAutoEncoder(node_feature.shape[0], node_feature.shape[1], 250, 75, 250, 75,lambda_cols=100, add_linear_row=False, add_linear_col=False, num_layers=2) .to(device)
    state = torch.load(path)
    model.load_state_dict(state)
    return model 



def premutate_graph(edgelist, max_ineration = 100):

    edgelist["target"] = np.random.permutation(edgelist["target"])
    dup = edgelist.drop_duplicates().index
    dup = edgelist.loc[~edgelist.index.isin(dup)]
    dup2 = edgelist.drop_duplicates().merge(edgelist.drop_duplicates(), left_on=["source","target"], right_on=["target","source"], how="left")
    dup2.index = edgelist.drop_duplicates().index
    dup2.dropna(inplace=True)
    dup = list(dup.index) + list(dup2.index)
    inter = 0
    while len(dup)> 100 and inter < max_ineration:
        edgelist.loc[dup,"target"] = np.random.permutation(edgelist.loc[dup,"target"])
        dup = edgelist.drop_duplicates().index
        dup = edgelist.loc[~edgelist.index.isin(dup)]
        dup2 = edgelist.drop_duplicates().merge(edgelist.drop_duplicates(), left_on=["source","target"], right_on=["target","source"], how="left")
        dup2.index = edgelist.drop_duplicates().index
        dup2.dropna(inplace=True)
        dup = list(dup.index) + list(dup2.index)
        inter+=1

    return edgelist


def run_graph_perm_test(go, perm = 100, alphs=0.05):
    cent = nx.eigenvector_centrality(go)
    obs =  np.fromiter(dict(sorted(cent.items(), key=lambda item: item[0], reverse=True)).values(),dtype=float)
    names = np.array(list(dict(sorted(cent.items(), key=lambda item: item[0], reverse=True)).keys()))
    results = []
    for i in range(perm):
        perm_graph = premutate_graph(nx.to_pandas_edgelist(go))
        self_edge = pd.DataFrame({"source":list(go.nodes), "target":list(go.nodes),"weight":[1 for _ in range(len(go.nodes))]})
        perm_graph = pd.concat([perm_graph, self_edge])
        go_perm = nx.from_pandas_edgelist(perm_graph)
        cent = nx.eigenvector_centrality(go_perm)
        cent = np.fromiter(dict(sorted(cent.items(), key=lambda item: item[0], reverse=True)).values(), dtype=float)
        results.append(cent)
    
    results = np.array([value >= obs for value in results]).sum(axis=0) < alphs*perm
    return names[results]


def crate_perm(array):
    df = pd.DataFrame(array)
    df = df.apply(lambda x: pd.Series(x.values[random.sample(range(x.shape[0]),x.shape[0])]), axis=1 )
    return df.values

def calculate_dc_matrix(embbed_row1, embbed_row2):
    cor_rows_embbed1 = np.abs(np.corrcoef(embbed_row1))
    mask = (cor_rows_embbed1 > np.percentile(cor_rows_embbed1, 99)).astype(np.int64)
    cor_rows_embbed1*= mask

    cor_rows_embbed2 = np.abs(np.corrcoef(embbed_row2))
    mask = (cor_rows_embbed2 > np.percentile(cor_rows_embbed2, 99)).astype(np.int64)
    cor_rows_embbed2*= mask

    return  cor_rows_embbed1 - cor_rows_embbed2

def permutation_test_coexpression(embbed_row_co, embbed_row_cond, num_of_perm = 100, threshold=0.2, down_flag=True):
    results = []
    if down_flag:
        observed = calculate_dc_matrix(embbed_row_co.detach().cpu().numpy(), embbed_row_cond.detach().cpu().numpy())
    else:
        observed = calculate_dc_matrix(embbed_row_cond.detach().cpu().numpy(), embbed_row_co.detach().cpu().numpy())
    
    for _ in range(num_of_perm):
        cor_row_embbed_cond_perm = np.abs(np.corrcoef(crate_perm(embbed_row_cond.detach().cpu().numpy())))
        cor_row_embbed_co_perm = np.abs(np.corrcoef(crate_perm(embbed_row_co.detach().cpu().numpy())))

        if down_flag:
            perm = calculate_dc_matrix(cor_row_embbed_co_perm, cor_row_embbed_cond_perm)
        else:
            perm = calculate_dc_matrix(cor_row_embbed_cond_perm, cor_row_embbed_co_perm)

        results.append(perm >= np.abs(observed))
    
    results = sum(np.array(results))
    return (observed > threshold) * (results < 0.05 * num_of_perm).astype(int)




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
        pred = propagation(train_vec.values, w.values)
        results.append([calc_roc(pred, term_vec, test_vec)])
    return results


def crate_random_graph(go,perm=100):
    resuls = []
    self_edge = pd.DataFrame({"source":list(go.nodes), "target":list(go.nodes),"weight":[1 for _ in range(len(go.nodes))]})
    for i in range(perm):
        perm_graph = premutate_graph(nx.to_pandas_edgelist(go))
        perm_graph = pd.concat([perm_graph, self_edge])
        resuls.append(nx.from_pandas_edgelist(perm_graph))
    return resuls 


def calc_term_z_score(grap_vec,term):
    pred = make_term_predication(grap_vec, term)
    pred = np.array(pred)
    obs, background = pred[0], pred[1:]
    return (obs - background.mean()) / background.std() 

def calcalte_z_anaylsis(go, annot):
    annot_threshold = annot.sum()>=30
    annot_threshold = annot_threshold[annot_threshold == True]
    perm = crate_random_graph(go,perm=30)
    grap_vec = [go] + perm
    return annot[annot_threshold.index].apply(lambda x: calc_term_z_score(grap_vec,x), axis=0)
    

    
def helper_z(args):
    print(args)
    return calcalte_z_anaylsis(args[0], annot=args[1])