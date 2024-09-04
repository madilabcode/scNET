import os
import pandas as pd
import numpy as np
import scanpy as sc
import torch
import networkx as nx
from MultyGraphModel import scNET
from Utils import save_model, save_obj
from random import seed
import torch
from torch_geometric.utils import  convert
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges
from torch.utils.tensorboard import SummaryWriter
from KNNDataset import KNNDataset, CellDataset
from torch.utils.data import DataLoader
import warnings
import gc 
import Utils as ut



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings('ignore')
torch.manual_seed(101)

def build_network(obj, net, biogrid_flag = False, human_flag = False, random_noise=False):
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
    node_feature = node_feature.loc[node_feature.non_zero > node_feature.shape[1] *  0.0]
    node_feature.drop("non_zero",axis=1,inplace=True)

    net = net.loc[net.Source != net.Target]
    net = net.loc[net.Source.isin(node_feature.index)]
    net = net.loc[net.Target.isin(node_feature.index)]

    gp = nx.from_pandas_edgelist(net, "Source", "Target")

    node_feature = node_feature.loc[list(gp.nodes)]


    return net, gp, node_feature

def test_recon(model,x, data, knn_edge_index):
    model.eval()
    with torch.no_grad():
        embbed_rows, embbed_cols, out_features = model(x, knn_edge_index, data.train_pos_edge_index)
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
  adjacency_matrix = torch.tensor(knn[idxs][:,idxs].toarray())
  row_indices, col_indices = torch.nonzero(adjacency_matrix, as_tuple=True)
  knn_edge_index = torch.stack((row_indices, col_indices))
  knn_edge_index = torch.unique(knn_edge_index, dim=1)
  return knn_edge_index.to(device)

def train(data, loader, highly_variable_index,number_of_batches=5 ,
          max_epoch = 500, rduce_interavel = 50,writer = None,model_name="", cell_flag=False):
    x_full = data.x.clone()
    if cell_flag:
      model = scNET(x_full.shape[0], x_full.shape[1]//number_of_batches,
                                250, 75, 250, 75, lambda_rows = 1, lambda_cols=1,num_layers=3).to(device)
    else:
      model = scNET(x_full.shape[0], x_full.shape[1], 250, 75, 250, 75, 
                                lambda_rows = 1, lambda_cols=1).to(device)
      x = x_full.clone()
      x = ((x.T - (x.mean(axis=1)))/ (x.std(axis=1)+ 0.00001)).T

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

    best_auc = 0.5 
    concat_flag = False

    for epoch in range(max_epoch):
        
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

          print(f"row loss:{total_row_loss}, col loss:{total_col_loss}")

          auc, ap = test_recon(model, x.to(device), data, knn_edge_index)

          print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))
          
          if writer is not None:
              writer.add_scalar("Loss/row_loss", total_row_loss / number_of_batches, epoch) 
              writer.add_scalar("Loss/col_loss", total_col_loss / number_of_batches, epoch) 


          if auc > best_auc:
            best_auc = auc

          save_model(r"./Models/BiEncdoer_best_" + model_name + ".pt", model)
          if cell_flag:
            st = torch.stack(row_emb_lst)
            row_embed = st.mean(dim=0)
            save_obj(torch.concat(col_emb_lst).cpu().detach().numpy(), r"./Embedding/col_embedding_" + model_name)
            save_obj(row_embed.cpu().detach().numpy(), r"./Embedding/row_embedding_" + model_name)              
            save_obj(torch.concat(out_features_lst).cpu().detach().numpy(),  r"./Embedding/out_features_" + model_name)
          else:
            save_obj(new_knn_edge_index.cpu(), r"./KNNs/best_new_knn_graph_" + model_name)
            save_obj(col_embed.cpu().detach().numpy(), r"./Embedding/col_embedding_" + model_name)
            save_obj(row_embed.cpu().detach().numpy(), r"./Embedding/row_embedding_" + model_name)
            save_obj(out_features.cpu().detach().numpy(),  r"./Embedding/out_features_" + model_name)

  
    if cell_flag:
      save_obj(loader, "knn_loader"+model_name)
    else:
      save_obj(new_knn_edge_index.cpu(), "new_knn_graph_"+model_name)

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

def main(path = "./Data/cell_line.h5ad",pre_processing_flag = True ,biogrid_flag = False,
          human_flag=False,number_of_batches=5,split_cells = False, n_neighbors=25,
          max_epoch=150, model_name=""):
    
    if pre_processing_flag:
       obj = sc.read(path)
       obj = pre_processing(obj,n_neighbors)
    else:
      obj = sc.read(path)
      obj.X = obj.raw.X
      sc.pp.log1p(obj)
      sc.pp.neighbors(obj, n_neighbors=n_neighbors, n_pcs=15)
    
    if not biogrid_flag:
      net = pd.read_csv(r"./Data/format_h_sapiens.csv")[["g1_symbol","g2_symbol","conn"]].drop_duplicates()
      net, ppi, node_feature = build_network(obj, net,human_flag=human_flag)
      print(f"N genes: {node_feature.shape}")

    else:
      net = pd.read_table(r"./Data/BIOGRID.tab.txt")[["OFFICIAL_SYMBOL_A","OFFICIAL_SYMBOL_B"]].drop_duplicates()
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
    node_feature.to_csv(r"./Embedding/node_features_" + model_name)
    x = node_feature.values

    x = torch.tensor(x, dtype=torch.float32).cpu()
    if split_cells: 
      loader = mini_batch_cells(x, obj.obsp["distances"], x.shape[1] // number_of_batches)


    writer = SummaryWriter(log_dir=f"./runs/{model_name}")          
    data = Data(x,ppi_edge_index)
    data = train_test_split_edges(data,test_ratio=0.2, val_ratio=0)
    model = train(data, loader, highly_variable_index, number_of_batches=number_of_batches, max_epoch=max_epoch, 
                    rduce_interavel=30,model_name=model_name, cell_flag=split_cells, writer=writer)
    writer.flush()
    writer.close()
    save_model(r"./Models/scNET_" + model_name + ".pt", model)

if __name__ == "__main__":
    main(path = r"./Data/cell_line.h5ad", pre_processing_flag = False,human_flag=True,split_cells=True,number_of_batches=3)