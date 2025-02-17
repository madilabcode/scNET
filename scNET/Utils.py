import torch
import numpy as np
import pandas as pd 
import scanpy as sc
from scipy.stats import ranksums
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import average_precision_score
import pickle 
import pkg_resources

alpha  = 0.9
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

def run_signature(obj, up_sig, down_sig=None, umap_flag = True, alpha = 0.9,prop_exp = None):
    """
    Calculate and visualize a propagated signature score for cells in the given object.
    Parameters
    ----------
    obj : AnnData
        The annotated data object containing gene expression matrix and graph data.
    up_sig : list or set
        A collection of genes used to calculate the up-regulated signature score.
    down_sig : list or set, optional
        A collection of genes used to calculate the down-regulated signature score.
        If None, only the up-regulated signature is used. Default is None.
    umap_flag : bool, optional
        If True, generates a UMAP plot colored by the calculated signature score.
        If False, generates a t-SNE plot. Default is True.
    alpha : float, optional
        A parameter controlling the smoothing or propagation factor during signature
        score calculation. Default is 0.9.
    prop_exp : None or other, optional
        An unused parameter placeholder, reserved for future use or extended
        signature propagation functionality.
    Returns
    -------
    np.ndarray
        An array of propagated signature scores, with one score per cell. The
        scores are also stored in obj.obs["SigScore"].
    """

    exp = obj.to_df().T
    graph = obj.obsp["connectivities"].toarray()
    sigs_scores = signature_values(exp, up_sig, down_sig)
    sigs_scores = propagation(sigs_scores, graph)
    obj.obs["SigScore"] = sigs_scores
    # color_map = "jet"
    if umap_flag:
        sc.pl.umap(obj, color=["SigScore"],color_map="magma")
    else:
        sc.pl.tsne(obj, color=["SigScore"],color_map="magma")
    return sigs_scores

def calculate_roc_auc(idents, predict):
    fpr, tpr, _ = roc_curve(idents, predict, pos_label=1)
    return auc(fpr, tpr)

def calculate_aupr(idents, predict):
    return average_precision_score(idents, predict)

def calculate_roc_auc(idents, predict):
    fpr, tpr, _ = roc_curve(idents, predict, pos_label=1)
    return auc(fpr, tpr)

def calculate_aupr(idents, predict):
    return average_precision_score(idents, predict)

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

def save_model(path, model):
    torch.save(model.state_dict(), path)


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
    embeded_genes = load_obj(pkg_resources.resource_filename(__name__,r"./Embedding/row_embedding_" + proj_name))
    embeded_cells = load_obj(pkg_resources.resource_filename(__name__,r"./Embedding/col_embedding_" + proj_name))
    node_features = pd.read_csv(pkg_resources.resource_filename(__name__,r"./Embedding/node_features_" + proj_name),index_col=0)
    out_features = load_obj(pkg_resources.resource_filename(__name__,r"./Embedding/out_features_" + proj_name))
    return embeded_genes, embeded_cells, node_features, out_features