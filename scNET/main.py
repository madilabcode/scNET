import os
import pandas as pd
import numpy as np
import scanpy as sc
import torch
import networkx as nx
from scNET.MultyGraphModel import scNET
from scNET.Utils import save_model, save_obj
import torch
from torch_geometric.utils import  convert
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges
from scNET.KNNDataset import KNNDataset, CellDataset
from torch.utils.data import DataLoader
import warnings
import gc 
import scNET.Utils as ut
import pkg_resources
from tqdm import tqdm
import warnings

INTER_DIM = 250
EMBEDDING_DIM = 75
NETWORK_CUTOFF = 0.5
MAX_CELLS_BATCH_SIZE = 2000
MAX_CELLS_FOR_SPLITING = 10000

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings('ignore')
torch.manual_seed(101)


def build_network(obj, net, biogrid_flag = False, human_flag = False):
    """
    Build a gene-gene network from the provided interaction information.
    Args:
      obj (anndata.AnnData): Single-cell data object (AnnData) containing gene expression data.
      net (pandas.DataFrame): DataFrame containing gene interactions (Source, Target, and Conn columns).
      biogrid_flag (bool, optional): If True, columns for net are set to ["Source", "Target"] only.
      human_flag (bool, optional): If True, keeps gene names unchanged; otherwise adjusts gene name casing.
    Returns:
      tuple:
        pandas.DataFrame: Filtered interaction DataFrame for valid genes.
        networkx.Graph: Graph representation of the gene network.
        pandas.DataFrame: Node-level gene expression features.
    """
    if not biogrid_flag:
        net.columns = ["Source","Target","Conn"]
        net = net.loc[net.Conn >= NETWORK_CUTOFF]
    
    else:
         net.columns = ["Source","Target"]
    
    if not human_flag:
        net["Source"] = net["Source"].apply(lambda x: x[0] + x[1:].lower()).astype(str)
        net["Target"] = net["Target"].apply(lambda x: x[0] + x[1:].lower()).astype(str)

         
    genes = list(pd.concat([net.Source, net.Target]).drop_duplicates())
    genes =  obj.var[obj.var.index.isin(genes)].index
    node_feature = sc.get.obs_df(obj.raw.to_adata(),list(genes)).T
    node_feature["non_zero"] = node_feature.apply(lambda x: x.astype(bool).sum(), axis=1)
    node_feature = node_feature.loc[node_feature.non_zero > node_feature.shape[1] *  0.0]
    node_feature.drop("non_zero",axis=1,inplace=True)

    net = net.loc[net.Source != net.Target]
    net = net.loc[net.Source.isin(node_feature.index)]
    net = net.loc[net.Target.isin(node_feature.index)]

    gp = nx.from_pandas_edgelist(net, "Source", "Target")

    node_feature = node_feature.loc[list(gp.nodes)]


    return net, gp, node_feature

def test_recon(model,x, data, knn_edge_index):
    """
    Evaluate model reconstruction performance on test edges.
    Args:
      model (torch.nn.Module): Trained scNET model.
      x (torch.Tensor): Input features for the nodes.
      data (torch_geometric.data.Data): Graph data object containing positive and negative edges.
      knn_edge_index (torch.Tensor): k-NN graph edges for the rows.
    Returns:
      float: AUC score of edge reconstruction.
    """
    model.eval()
    with torch.no_grad():
        embbed_rows, _, _ = model(x, knn_edge_index, data.train_pos_edge_index)
    return model.test(embbed_rows, data.test_pos_edge_index, data.test_neg_edge_index)

def pre_processing(adata,n_neighbors): 
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
   
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata.copy()
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=15)

    return adata

def crate_knn_batch(knn,idxs,k=15):
  """
  Create a mini-batch of the k-NN graph for the given subset of indices.
  Args:
    knn (scipy.sparse.csr_matrix): Sparse adjacency matrix representing the k-NN graph.
    idxs (list[int]): List of indices used to subset the k-NN graph.
    k (int, optional): Number of nearest neighbors (used for reference if needed).
  Returns:
    torch.Tensor: Edge index for the sub-batch of the k-NN graph.
  """
  adjacency_matrix = torch.tensor(knn[idxs][:,idxs].toarray())
  row_indices, col_indices = torch.nonzero(adjacency_matrix, as_tuple=True)
  knn_edge_index = torch.stack((row_indices, col_indices))
  knn_edge_index = torch.unique(knn_edge_index, dim=1)
  return knn_edge_index.to(device)

def train(data, loader, highly_variable_index,number_of_batches=5 ,
          max_epoch = 500, rduce_interavel = 30,model_name="", cell_flag=False):
    """
      Train the scNET model using mini-batches of the k-NN graph or cells.
      Args:
        data (torch_geometric.data.Data): Graph data including edge information.
        loader (torch.utils.data.DataLoader): DataLoader for batches of edges or cells.
        highly_variable_index (pandas.Series or np.ndarray): Boolean mask for highly variable genes.
        number_of_batches (int, optional): Number of mini-batches.
        max_epoch (int, optional): Maximum number of training epochs.
        rduce_interavel (int, optional): Interval at which the model attempts graph reduction.
        model_name (str, optional): Custom string identifier for saving the model and outputs.
        cell_flag (bool, optional): If True, performs mini-batch training by cells rather than by edges.
      Returns:
        scNET: Trained scNET model instance.
      Build a k-NN graph from precomputed distances in the AnnData object.
      Args:
        obj (anndata.AnnData): Single-cell data object with 'distances' stored in obsp.
      Returns:
        tuple:
          torch.Tensor: Edge index of the k-NN graph.
          pandas.Series: Boolean mask for highly variable genes.
      Create a mini-batch DataLoader for k-NN edges.
      Args:
        edge_index (torch.Tensor): All edges of the k-NN graph.
        batch_size (int): Number of edges per mini-batch.
      Returns:
        torch.utils.data.DataLoader: DataLoader object for batching edges.
    """
    x_full = data.x.clone()
    if cell_flag:
      model = scNET(x_full.shape[0], x_full.shape[1]//number_of_batches,
                                INTER_DIM, EMBEDDING_DIM, INTER_DIM, EMBEDDING_DIM, lambda_rows = 1, lambda_cols=1,num_layers=3).to(device)
    else:
      model = scNET(x_full.shape[0], x_full.shape[1], INTER_DIM, EMBEDDING_DIM, INTER_DIM, EMBEDDING_DIM, 
                                lambda_rows = 1, lambda_cols=1).to(device)
      x = x_full.clone()
      x = ((x.T - (x.mean(axis=1)))/ (x.std(axis=1)+ 0.00001)).T

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

    best_auc = 0.5 
    concat_flag = False

    for epoch in tqdm(range(max_epoch), desc="Training", total=max_epoch):

        total_row_loss = 0
        total_col_loss = 0
        col_emb_lst = []
        row_emb_lst = []
        imput_lst = []
        out_features_lst = []
        concat_flag = False 

        for _,batch in enumerate(loader):
            model.train()
           
            if cell_flag:
              x = batch[0].T
              x = ((x.T - (x.mean(axis=1)))/ (x.std(axis=1)+ 0.00001)).T
              knn_edge_index = crate_knn_batch(loader.dataset.knn, batch[1])
           
            else:
              knn_edge_index = batch.T.to(device)

            if cell_flag or knn_edge_index.shape[1] == loader.dataset.edge_index.shape[0] // number_of_batches :
                
                loss, row_loss, col_loss = model.calculate_loss(x.clone().to(device), knn_edge_index.to(device),
                                                                data.train_pos_edge_index,highly_variable_index)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_row_loss += row_loss
                total_col_loss += col_loss

                with torch.no_grad():
                  if cell_flag:
                    row_embed, col_embed, out_features = model(x.clone().to(device), knn_edge_index, data.train_pos_edge_index)
                    imput = model.encoder(x.to(device), knn_edge_index, data.train_pos_edge_index)
                    col_emb_lst.append(col_embed.cpu())
                    row_emb_lst.append(row_embed.cpu())
                    imput_lst.append(imput.T.cpu())
                    out_features_lst.append(out_features.cpu())
                  else:
                    row_embed, col_embed, out_features = model(x.to(device),knn_edge_index.to(device), data.train_pos_edge_index)

            else:
              concat_flag = True
            
            gc.collect()
            torch.cuda.empty_cache()

        if not cell_flag:
          new_knn_edge_index, _ = model.cols_encoder.reduce_network()   

          if concat_flag:
              new_knn_edge_index = torch.concat([new_knn_edge_index,knn_edge_index], axis=-1)
              knn_edge_index = new_knn_edge_index

          if (epoch+1) % rduce_interavel == 0:
              print(new_knn_edge_index.shape[1] / loader.dataset.edge_index.shape[0])
              loader = mini_batch_knn(new_knn_edge_index, new_knn_edge_index.shape[1] // number_of_batches)
 


        if epoch%10 == 0:
          if not cell_flag:
            knn_edge_index = list(loader)[0].T.to(device)

          auc, ap = test_recon(model, x.to(device), data, knn_edge_index)
          
          if auc > best_auc:
            best_auc = auc

          if cell_flag:
            st = torch.stack(row_emb_lst)
            row_embed = st.mean(dim=0)
            save_obj(torch.concat(col_emb_lst).cpu().detach().numpy(), pkg_resources.resource_filename(__name__,r"Embedding/col_embedding_" + model_name))
            save_obj(row_embed.cpu().detach().numpy(), pkg_resources.resource_filename(__name__,r"Embedding/row_embedding_" + model_name))           
            save_obj(torch.concat(out_features_lst).cpu().detach().numpy(),  pkg_resources.resource_filename(__name__,r"Embedding/out_features_" + model_name))
          else:
            save_obj(new_knn_edge_index.cpu(),pkg_resources.resource_filename(__name__, r"KNNs/best_new_knn_graph_" + model_name))
            save_obj(col_embed.cpu().detach().numpy(), pkg_resources.resource_filename(__name__,r"Embedding/col_embedding_" + model_name))
            save_obj(row_embed.cpu().detach().numpy(), pkg_resources.resource_filename(__name__,r"Embedding/row_embedding_" + model_name))
            save_obj(out_features.cpu().detach().numpy(),  pkg_resources.resource_filename(__name__,r"Embedding/out_features_" + model_name))

    print(f"Best Network AUC: {best_auc}")
   # if cell_flag:
   #   save_obj(loader, "knn_loader"+model_name)
   # else:
   #   save_obj(new_knn_edge_index.cpu(), "new_knn_graph_"+model_name)

    return model

def build_knn_graph(obj):
    graph = obj.obsp["distances"].toarray()
    graph = (graph > 0).astype(int)
    graph = nx.from_numpy_array(np.matrix(graph))
    ppi_geo = convert.from_networkx(graph)
    edge_index = ppi_geo.edge_index
    sc.pp.highly_variable_genes(obj)
    return edge_index, obj.var.highly_variable

def mini_batch_knn(edge_index, batch_size):
    """
    Create a mini-batch DataLoader for cells and their corresponding edges.
    Args:
      x (torch.Tensor): Matrix of gene expression features.
      edge_index (scipy.sparse.spmatrix): Distance or similarity matrix for cells.
      batch_size (int): Number of cells per mini-batch.
    Returns:
      torch.utils.data.DataLoader: DataLoader object for batching cells.
    Convert a NetworkX graph to a PyTorch Geometric edge index.
    Args:
      G (networkx.Graph): Input NetworkX graph.
      mapping (dict, optional): Dictionary mapping original node IDs to new indices.
    Returns:
      tuple:
        torch.Tensor: PyTorch Geometric edge index.
        dict: Mapping of graph nodes to tensor indices.
    """
    knn_dataset = KNNDataset(edge_index)
    knn_loader = DataLoader(knn_dataset,batch_size=batch_size, shuffle=True, drop_last=False)
    return knn_loader

def mini_batch_cells(x,edge_index, batch_size):
    cell_dataset = CellDataset(x, edge_index)
    cell_loader = DataLoader(cell_dataset,batch_size=batch_size, shuffle=False, drop_last=True)
    return cell_loader

def nx_to_pyg_edge_index(G, mapping=None):
    G = G.to_directed() if not nx.is_directed(G) else G
    if mapping is None:  
       mapping = dict(zip(G.nodes(), range(G.number_of_nodes())))
    edge_index = torch.empty((2, G.number_of_edges()), dtype=torch.long).to(device)
    for i, (src, dst) in enumerate(G.edges()):
        edge_index[0, i] = mapping[src]
        edge_index[1, i] = mapping[dst]
    return edge_index, mapping

def run_scNET(obj,pre_processing_flag = True ,biogrid_flag = False,
          human_flag=False,number_of_batches=5,split_cells = False, n_neighbors=25,
          max_epoch=150, model_name="", save_model_flag = False):
    """
    Main function to load data, build networks, and run the scNET training pipeline.
    Args:
      obj (AnnData, optional): AnnData obj.
      pre_processing_flag (bool, optional): If True, perform pre-processing steps.
      biogrid_flag (bool, optional): If True, use BioGRID-formatted data for network building.
      human_flag (bool, optional): Controls gene name casing in the network.
      number_of_batches (int, optional): Number of mini-batches for the training.
      split_cells (bool, optional): If True, split by cells instead of edges during training.
      n_neighbors (int, optional): Number of neighbors for building the adjacency graph.
      max_epoch (int, optional): Max number of epochs for model training.
      model_name (str, optional): Identifier for saving the model outputs.
      save_model_flag (bool, optional): If True, save the trained model.
    Returns:
      scNET: A trained scNET model.
    """
    torch.manual_seed(42)

    if pre_processing_flag:
       obj = pre_processing(obj,n_neighbors)

    else:
      if not obj.raw is None:
        obj.X = obj.raw.X
      sc.pp.log1p(obj)
      sc.pp.neighbors(obj, n_neighbors=n_neighbors, n_pcs=15)
    
    if obj.obs.shape[0] > MAX_CELLS_FOR_SPLITING:
       split_cells = True
    
    if split_cells:
       batch_size = obj.obs.shape[0] // number_of_batches
       if batch_size > MAX_CELLS_BATCH_SIZE:
          number_of_batches = obj.obs.shape[0] // 2000


    if not biogrid_flag:
      print(pkg_resources.resource_filename(__name__,r"Data/format_h_sapiens.csv"))

      net = pd.read_csv(pkg_resources.resource_filename(__name__,r"Data/format_h_sapiens.csv"))[["g1_symbol","g2_symbol","conn"]].drop_duplicates()
      net, ppi, node_feature = build_network(obj, net,human_flag=human_flag)
      print(f"N genes: {node_feature.shape}")

    else:
      print(pkg_resources.resource_filename(__name__,r"Data/BIOGRID.tab.txt"))
      net = pd.read_table(pkg_resources.resource_filename(__name__,r"Data/BIOGRID.tab.txt"))[["OFFICIAL_SYMBOL_A","OFFICIAL_SYMBOL_B"]].drop_duplicates()
      net, ppi, node_feature  = build_network(obj, net, biogrid_flag,human_flag)
      print(f"N genes: {node_feature.shape}")

    ppi_edge_index, _ = nx_to_pyg_edge_index(ppi)
    ppi_edge_index = ppi_edge_index.to(device)

    if split_cells:
      obj = obj[:,node_feature.index]
      sc.pp.highly_variable_genes(obj)
      highly_variable_index =  obj.var.highly_variable 
  
    else:
      obj = obj[:,node_feature.index]
      knn_edge_index, highly_variable_index = build_knn_graph(obj)    
      loader = mini_batch_knn(knn_edge_index, knn_edge_index.shape[1] // number_of_batches)
  
    highly_variable_index = highly_variable_index[node_feature.index]
    node_feature.to_csv(pkg_resources.resource_filename(__name__,r"Embedding/node_features_" + model_name))
    x = node_feature.values

    x = torch.tensor(x, dtype=torch.float32).cpu()
    if split_cells: 
      loader = mini_batch_cells(x, obj.obsp["distances"], x.shape[1] // number_of_batches)

    data = Data(x,ppi_edge_index)
    data = train_test_split_edges(data,test_ratio=0.2, val_ratio=0)
    model = train(data, loader, highly_variable_index, number_of_batches=number_of_batches, max_epoch=max_epoch, 
                    rduce_interavel=30,model_name=model_name, cell_flag=split_cells)
    
    if save_model_flag:
      save_model(pkg_resources.resource_filename(__name__, r"Models/scNET_" + model_name + ".pt"), model)

