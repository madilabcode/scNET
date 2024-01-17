import os
#os.environ["R_HOME"] = r"C:\Program Files\R\R-4.2.1"
#os.environ['path'] += r";C:\Program Files\R\R-4.2.1"
#from rpy2.robjects import r
#from rpy2.robjects.conversion import localconverter
import pandas as pd
import numpy as np
import scanpy as sc
import seaborn as sns 
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import networkx as nx
from MultyGraphModel import BiGraphAutoEncoder
from Utils import crate_anndata, save_model, save_obj
from random import seed
from sklearn.metrics import roc_curve, auc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import  convert
from torch_geometric.data import Data
from torch_geometric.nn import sequential, GATConv, GraphNorm, VGAE, GCNConv, InnerProductDecoder, GAE
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import train_test_split_edges
from torch.utils.tensorboard import SummaryWriter
from KNNDataset import KNNDataset, CellDataset
from torch.utils.data import DataLoader, random_split
import warnings
import matplotlib.pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings('ignore')
#device = torch.device("cpu")
torch.manual_seed(101)

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

    net = net.loc[net.Source.isin(node_feature.index)]
    net = net.loc[net.Target.isin(node_feature.index)]
    #if random_noise:
     # net.Source = net.Source.sample(frac=1).values
      # node_feature += np.random.normal(loc=0,scale=2,size=node_feature.shape)
    gp = nx.from_pandas_edgelist(net, "Source", "Target")
    node_feature = node_feature.loc[list(gp.nodes)]
   # if random_noise:
  #    node_feature = node_feature.sample(frac=1)
      #node_feature += np.random.normal(loc=0,scale=3,size=node_feature.shape)
    return net, gp, node_feature, heg


def test_recon(model,x, data, knn_edge_index, vae_flag=False):
    model.eval()
    with torch.no_grad():
        embbed = model.encoder(x, knn_edge_index, data.train_pos_edge_index)
        if vae_flag:
            mu, logstd = model.rows_encoder(embbed, data.train_pos_edge_index)
            embbed_rows = model.rows_encoder.encode(mu,logstd)
        else:
            embbed_rows = model.rows_encoder(embbed, data.train_pos_edge_index)
    return model.test(embbed_rows, data.test_pos_edge_index, data.test_neg_edge_index)


def pre_processing(adata): 
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

    adata = adata[adata.obs.n_genes_by_counts < 2500, :]
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata.raw = adata
    sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
    #adata = adata[:, adata.var.highly_variable]
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=15)
    sc.tl.leiden(adata,resolution =0.5)
    sc.tl.umap(adata)
    sc.pl.leiden(adata,color = "leiden")
    return adata

def crate_knn(tensor,k=10):
    num_points = tensor.shape[0]
    pairwise_distances = torch.cdist(tensor, tensor, p=2)  # Euclidean distance

    # Step 2: Find k-Nearest Neighbors
    _, indices = torch.topk(pairwise_distances, k + 1, largest=False, sorted=True)
    # Exclude the self-loop
    indices = indices[:, 1:]

    # Step 3: Create Graph (Adjacency Matrix)
    row_indices = torch.arange(num_points).view(-1, 1).expand(-1, k).contiguous().view(-1).to(device)
    col_indices = indices.reshape(-1).to(device)
    adjacency_matrix = torch.sparse_coo_tensor(torch.stack([row_indices, col_indices]), torch.ones(k * num_points).to(device), (num_points, num_points)).to(device)
    adjacency_matrix = (adjacency_matrix.to_dense() + adjacency_matrix.to_dense().t()).clamp(0, 1)
    row_indices, col_indices = torch.nonzero(adjacency_matrix, as_tuple=True)
    # Stack them together to form the edge_index tensor
    edge_index = torch.stack((row_indices, col_indices)).to(device)

    # If your graph is undirected, the adjacency matrix is symmetric
    # In this case, you might want to keep only one of the (i, j) and (j, i) pairs
    edge_index = torch.unique(edge_index, dim=1).to(device)
    return edge_index

def train(data, loader ,highly_variable_index,number_of_batches=5 , max_epoch = 500, rduce_interavel = 50,writer = None,model_name="", train_all = False, cell_flag=False):
    x = data.x.clone()
    if cell_flag:
      model = BiGraphAutoEncoder(x.shape[0], x.shape[1]//number_of_batches, 250, 75, 500, 150, lambda_rows = 1, lambda_cols=1, vae_flag=False, add_linear_row=False, add_linear_col=False, num_layers=3, cd=False) .to(device)
    else:
      model = BiGraphAutoEncoder(x.shape[0], x.shape[1], 250, 75, 250, 75, lambda_rows = 1, lambda_cols=1, vae_flag=False, add_linear_row=False, add_linear_col=False, num_layers=3, cd=False) .to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_auc = 0.5 
    col_flag = False  
    for epoch in range(max_epoch):
        total_row_loss = 0
        total_col_loss = 0
        concat_flag = False
        for _,batch in enumerate(loader):
            model.train()
            if cell_flag:
              x = batch[0].T
              knn_edge_index = crate_knn(x.T)
            else:
              knn_edge_index = batch.T

            if cell_flag or knn_edge_index.shape[1] == loader.dataset.edge_index.shape[0] // number_of_batches :
                #if train_all:
                #    loss, row_loss, col_loss = model.calculate_loss(x, knn_edge_index, data.edge_index, highly_variable_index)
                #else:
                loss, row_loss, col_loss = model.calculate_loss(x, knn_edge_index, data.train_pos_edge_index, highly_variable_index)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_row_loss += row_loss
                total_col_loss += col_loss
            else:
              concat_flag = True
              
        if not cell_flag:
          new_knn_edge_index, df = model.cols_encoder.reduce_network()   
          if concat_flag:
              new_knn_edge_index = torch.concat([new_knn_edge_index,knn_edge_index], axis=-1)
              knn_edge_index = new_knn_edge_index
          if (epoch+1) % rduce_interavel == 0:
              print(new_knn_edge_index.shape[1] / loader.dataset.edge_index.shape[0])
              loader = mini_batch_knn(new_knn_edge_index, new_knn_edge_index.shape[1] // number_of_batches)
            # optimizer.param_groups[0]["lr"] *= 0.9

            

        if epoch%10 == 0:
          print(loss)
          print(f"row loss:{total_row_loss}, col loss:{total_col_loss}")
            #if not train_all:
          auc, ap = test_recon(model, x, data, knn_edge_index, vae_flag = False)
          print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))
          if auc > best_auc - 0.01 and train_all:
            print(f"best result of model with auc: {auc}")
            save_model(r"./Models/BiEncdoer_best_" + model_name + ".pt", model)
            if cell_flag:
              save_obj(loader, "Best_knn_loader"+model_name)
            else:
              save_obj(new_knn_edge_index, "Best_new_knn_graph" + model_name)

            if auc > best_auc:
              best_auc = auc


           # plt.show()
            #atten = model.ae.encoder.atten_map[1].detach().cpu().numpy()
            #sns.displot(atten[atten > 0.05])
            #plt.show()

          if writer is not None:
                  writer.add_scalar("total_loss/train", loss, epoch)
                  writer.add_scalar("row_loss/train", row_loss, epoch) 
                  writer.add_scalar("col_loss/train", col_loss, epoch)       

   

    if train_all:
      if cell_flag:
        save_obj(loader, "knn_loader"+model_name)
      else:
        save_obj(new_knn_edge_index, "new_knn_graph_"+model_name)

      return model

    else:
      return model, x, knn_edge_index




def plot_k_fold_roc(fpr_list, tpr_list, auc_list):

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)

    mean_tpr = np.mean(tpr_list, axis = 0)
    mean_fpr = np.linspace(0, 1, 1000)
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(auc_list)

    plt.plot(mean_fpr, mean_tpr, color='b',
            label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tpr_list, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate',fontsize=18)
    plt.ylabel('True Positive Rate',fontsize=18)
    plt.title('Cross-Validation ROC',fontsize=18)
    plt.legend(loc="lower right", prop={'size': 15})
    plt.show()

def train_on_k_fold(folds, loader ,highly_variable_index,number_of_batches=5, max_epoch = 600, rduce_interavel = 50, cell_flag=False, data=None,writer = None):
    fpr_list = []
    tpr_list = []
    auc_list = []
    mean_fpr = np.linspace(0, 1, 1000)
    for data in folds:
        model,x, knn_edge_index = train(data, loader ,highly_variable_index, number_of_batches=number_of_batches, max_epoch=max_epoch, rduce_interavel=30, writer=writer, train_all= False, cell_flag=cell_flag)
        model.eval()   
        with torch.no_grad():
            embbed = model.encoder(x, knn_edge_index, data.train_pos_edge_index)
            z = model.rows_encoder(embbed, data.train_pos_edge_index)  
            pos_y = z.new_ones(data.test_pos_edge_index.size(1))
            neg_y = z.new_zeros(data.test_neg_edge_index.size(1))
            y = torch.cat([pos_y, neg_y], dim=0)

            pos_pred = model.ipd(z, data.test_pos_edge_index, sigmoid=True)
            neg_pred = model.ipd(z, data.test_neg_edge_index, sigmoid=True)
            pred = torch.cat([pos_pred, neg_pred], dim=0)

            y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
            fpr, tpr, _ = roc_curve(y, pred)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            fpr_list.append(fpr)
            tpr_list.append(interp_tpr)
            auc_list.append(auc(fpr, tpr))

    plot_k_fold_roc(fpr_list, tpr_list, auc_list)


def build_knn_graph(obj):
    graph = obj.obsp["distances"].toarray()
    graph = (graph > 0).astype(int)
    graph = nx.from_numpy_array(np.matrix(graph))
    ppi_geo = convert.from_networkx(graph)
    edge_index = ppi_geo.edge_index.to(device)
    obj.X = obj.raw.X
  #  sc.pp.highly_variable_genes(obj)
    return edge_index#, obj.var.highly_variable


def crate_k_fold_cross_validation(x, ppi_edge_index, k=5):
    n = ppi_edge_index.shape[1] // k 
    ppi_edge_index = ppi_edge_index[:,torch.randperm(ppi_edge_index.size()[1])]
    folds = []
    for i in range(k):
        test_set = ppi_edge_index[:,i*n:(i+1)*n]
        train_set = torch.concat([ppi_edge_index[:,:i*n], ppi_edge_index[:,(i+1)*n:]], dim=-1)
        data = Data(x,ppi_edge_index).to(device)
        data = train_test_split_edges(data,test_ratio=2*(1/k), val_ratio=0)
        data.train_pos_edge_index = train_set
        data.test_pos_edge_index = test_set
        folds.append(data)
    return folds


def mini_batch_knn(edge_index, batch_size):
    knn_dataset = KNNDataset(edge_index)
    knn_loader = DataLoader(knn_dataset,batch_size=batch_size, shuffle=True, drop_last=False)
    return knn_loader


def mini_batch_cells(x,edge_index, batch_size):
    cell_dataset = CellDataset(x, edge_index)
    cell_loader = DataLoader(cell_dataset,batch_size=batch_size, shuffle=False, drop_last=True)
    return cell_loader

def main(path = "./Data/cell_line.h5ad",pre_processing_flag = True ,biogrid_flag = False, train_all = False, human_flag=False,number_of_batches=5, random_noise=False, cell_flag = False, model_name=""):
    if pre_processing_flag:
       obj = pre_processing(path)
    else:
      obj = sc.read(path)
      sc.pp.log1p(obj)
      sc.pp.neighbors(obj, n_neighbors=40, n_pcs=15)
    if not biogrid_flag:
      net = pd.read_csv(r"./Data/format_h_sapiens.csv")[["g1_symbol","g2_symbol","conn"]].drop_duplicates()
      #pa, ppi, node_feature = build_ppi(pd.read_csv(r"./files/humanProtinAction.csv"), obj)
      net, ppi, node_feature, highly_variable_index = build_network(obj, net,human_flag=human_flag,random_noise=random_noise)
    else:
      net = pd.read_table(r"./Data/BIOGRID.tab.txt")[["OFFICIAL_SYMBOL_A","OFFICIAL_SYMBOL_B"]].drop_duplicates()
      #pa, ppi, node_feature = build_ppi(pd.read_csv(r"./files/humanProtinAction.csv"), obj)
      net, ppi, node_feature, highly_variable_index = build_network(obj, net, biogrid_flag,human_flag,random_noise=random_noise)
      print(node_feature.shape)
    ppi_geo = convert.from_networkx(ppi)
    ppi_edge_index = ppi_geo.edge_index
    ppi_edge_index = ppi_edge_index.to(device)
    knn_edge_index = build_knn_graph(obj)    
    loader = mini_batch_knn(knn_edge_index, knn_edge_index.shape[1] // number_of_batches)
    highly_variable_index = highly_variable_index[node_feature.index]
    
    scaler = StandardScaler()
    x = node_feature.values
    zero_mask = torch.tensor((x[highly_variable_index.values].T != 0 ) , dtype=torch.float32, device=device)
    x = scaler.fit_transform(x.T).T
    x = torch.tensor(x, dtype=torch.float32, device=device)
    if cell_flag: 
      loader = mini_batch_cells(x, knn_edge_index, x.shape[1] // number_of_batches)

    if random_noise:
  #    node_feature = node_feature.sample(frac=1)
      x += torch.tensor(np.random.normal(loc=0,scale=1,size=x.shape), device=device)
      
    if not train_all:
        folds =  crate_k_fold_cross_validation(x, ppi_edge_index)
        train_on_k_fold(folds, loader, highly_variable_index,max_epoch=50, rduce_interavel=30, number_of_batches=number_of_batches, cell_flag=cell_flag)
    else:
        writer = SummaryWriter(log_dir = f"./runs")  
        data = Data(x,ppi_edge_index)
        data = train_test_split_edges(data,test_ratio=0.2, val_ratio=0)
        model = train(data, loader ,highly_variable_index, number_of_batches=number_of_batches, max_epoch=500, rduce_interavel=30, writer=writer,model_name=model_name, train_all= True, cell_flag=cell_flag)
        writer.flush()
        save_model(r"./Models/BiEncdoer_" + model_name + ".pt", model)

    #model = BiGraphAutoEncoder(node_feature.shape[0], node_feature.shape[1], 150, 100, 250, 50,lambda_cols=10, add_linear_row=False, add_linear_col=True) .to(device)
    #writer = SummaryWriter(log_dir = f"./runs")  

    #model = train(model,node_feature, ppi_edge_index, knn_edge_index, highly_variable_index,writer=writer)
    #writer.flush()

if __name__ == "__main__":
    main(path = r"./Data/viti.h5ad", pre_processing_flag = False,human_flag=True, train_all = True,random_noise=True)