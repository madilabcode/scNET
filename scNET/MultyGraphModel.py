
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import sequential, GATConv, GraphNorm, VGAE, GCNConv, InnerProductDecoder, TransformerConv, GAE,LayerNorm, SAGEConv
from torch_geometric.nn.conv import transformer_conv
from torch_geometric.utils import negative_sampling
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
import math
import numpy as np
import pandas as pd
EPS = 1e-15
MAX_LOGSTD = 10

class FeatureDecoder(torch.nn.Module):
    def __init__(self, feature_dim, embd_dim,inter_dim , drop_p = 0.0):
        super(FeatureDecoder, self).__init__()
        self.feature_dim = feature_dim
        self.embd_dim = embd_dim
        self.inter_dim = inter_dim
        self.decoder = nn.Sequential(nn.Linear(embd_dim, inter_dim),
                                        nn.Dropout(drop_p),
                                        nn.ReLU(),
                                        nn.Linear(inter_dim, inter_dim),
                                        nn.Dropout(drop_p),
                                        nn.ReLU(),
                                        nn.Linear(inter_dim, feature_dim),
                                        nn.Dropout(drop_p))
              
    def forward(self, z):
        out = self.decoder(z)
        return out 
        
class MutualEncoder(torch.nn.Module):
  def __init__(self,col_dim, row_dim,num_layers=4, drop_p = 0.25):
    super(MutualEncoder, self).__init__()
    self.col_dim = col_dim
    self.row_dim = row_dim
    self.num_layers = num_layers

    self.rows_layers = nn.ModuleList([
      sequential.Sequential('x,edge_index', [
                                  (SAGEConv(self.row_dim, self.row_dim), 'x, edge_index -> x1'),
                                  (nn.Dropout(drop_p,inplace=False), 'x1-> x2'),
                                  nn.LeakyReLU(inplace=True),
                                  ]) for _ in range(num_layers)])
    
    self.cols_layers = nn.ModuleList([
      sequential.Sequential('x,edge_index', [
                                  (SAGEConv(self.col_dim, self.col_dim), 'x, edge_index -> x1'),
                                  nn.LeakyReLU(inplace=True),
                                  (nn.Dropout(drop_p,inplace=False), 'x1-> x2'),
                                ]) for _ in range(num_layers)])
                      

  def forward(self, x, knn_edge_index, ppi_edge_index):
    
      embbded = x.clone()
      for i in range(self.num_layers):
        embbded = self.cols_layers[i](embbded.T,knn_edge_index).T
        embbded = self.rows_layers[i](embbded, ppi_edge_index)
      
      return embbded

class TransformerConvReducrLayer(TransformerConv):
  def __init__(self, in_channels, out_channels, heads=1, dropout=0 , add_self_loops=True,scale_param = 2, **kwargs):
     super().__init__(in_channels, out_channels, heads, dropout, add_self_loops, **kwargs)
     self.treshold_alpha = None
     self.scale_param = scale_param
    
  def message(self, query_i, key_j, value_j,
                edge_attr, index, ptr,
                size_i):

        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads,
                                                      self.out_channels)
            key_j += edge_attr

        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        if not self.scale_param is None:
          alpha = alpha - alpha.mean()
          alpha = alpha / ((1/self.scale_param) * alpha.std())
          alpha = F.sigmoid(alpha)
        else:
          alpha = softmax(alpha, index, ptr, size_i)
        self.treshold_alpha = alpha 

        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value_j
        if edge_attr is not None:
            out += edge_attr

        out *= alpha.view(-1, self.heads, 1)
        return out

class DimEncoder(torch.nn.Module):
      def __init__(self,feature_dim, inter_dim, embd_dim,reducer=False,drop_p = 0.2, scale_param=3):
        super(DimEncoder, self).__init__()
        self.feature_dim = feature_dim
        self.embd_dim = embd_dim
        self.inter_dim = inter_dim
        self.reducer = reducer

        self.encoder = sequential.Sequential('x, edge_index', [
                                    (GCNConv(self.feature_dim, self.inter_dim), 'x, edge_index -> x1'),
                                    nn.LeakyReLU(inplace=True),
                                    (nn.Dropout(drop_p,inplace=False), 'x1-> x2')
                                  ])
        print(scale_param) 
        if self.reducer:                
          self.atten_layer = TransformerConvReducrLayer(self.inter_dim, self.embd_dim,dropout= drop_p,add_self_loops = False,heads=1, scale_param=scale_param)
        else:
           self.atten_layer = TransformerConv(self.inter_dim, self.embd_dim,dropout=drop_p)
           
        self.atten_map = None
        self.atten_weights = None
        self.plot_count = 0
      

      def reduce_network(self, threshold = 0.2, min_connect=10):
        self.plot_count += 1
        graph = self.atten_weights.cpu().detach().numpy()
        threshold_bound = np.percentile(graph, 10)
        threshold = min(threshold,threshold_bound) 
        df = pd.DataFrame({"v1": self.atten_map[0].cpu().detach().numpy(), "v2": self.atten_map[1].cpu().detach().numpy(), "atten": graph.squeeze()})
        saved_edges = df.groupby('v1')['atten'].nlargest(min_connect).index.values
        saved_edges = [v2 for _, v2 in saved_edges]
        df.iloc[saved_edges,2]  = threshold + EPS
        indexs = list(df.loc[df.atten >= threshold].index)
        atten_map = self.atten_map[:,indexs]
        self.atten_map = None
        self.atten_weights = None
        return atten_map, df 

      def forward(self, x, edge_index, infrance=False):
        embbded = x.clone()
        embbded = self.encoder(embbded,edge_index)
        embbded, atten_map = self.atten_layer(embbded, edge_index, return_attention_weights=True)
        if self.reducer and not infrance :
          if self.atten_map is None:
            self.atten_map = atten_map[0].detach()
            self.atten_weights = atten_map[1].detach()
          else:
            self.atten_map = torch.concat([self.atten_map.T, atten_map[0].detach().T]).T
            self.atten_weights = torch.concat([self.atten_weights, atten_map[1].detach()])

        return  embbded   


class scNET(torch.nn.Module):
  def __init__(self,col_dim, row_dim,inter_row_dim, embd_row_dim, inter_col_dim,embd_col_dim,
                lambda_rows = 1, lambda_cols = 1, num_layers=2, drop_p = 0.25):

    super(scNET, self).__init__()
    self.col_dim = col_dim
    self.row_dim = row_dim
    self.inter_row_dim = inter_row_dim
    self.embd_row_dim = embd_row_dim
    self.inter_col_dim = inter_col_dim
    self.embd_col_dim = embd_col_dim
    self.lambda_rows = lambda_rows
    self.lambda_cols = lambda_cols


    self.encoder = MutualEncoder(col_dim, row_dim,num_layers, drop_p)
    self.rows_encoder =  DimEncoder(row_dim, inter_row_dim, embd_row_dim,drop_p = drop_p, scale_param=None, reducer=False)

    self.cols_encoder =  DimEncoder(col_dim, inter_col_dim, embd_col_dim,drop_p=drop_p, reducer=True)
    self.feature_decodr = FeatureDecoder(col_dim, embd_col_dim, inter_col_dim, drop_p = 0 )
    self.ipd = InnerProductDecoder()
    self.feature_critarion = nn.MSELoss(reduction ='mean')

  def recon_loss(self, z, pos_edge_index, neg_edge_index = None, sig=False) :
      if neg_edge_index is None:
          neg_edge_index = negative_sampling(pos_edge_index, z.size(0))

      if not sig:
        embd = torch.corrcoef(z)
        pos = torch.sigmoid(embd[pos_edge_index[0],pos_edge_index[1]])
        neg = torch.sigmoid(embd[neg_edge_index[0],neg_edge_index[1]])
        pos_loss = -torch.log(pos +EPS).mean()
        neg_loss = -torch.log(1 - neg + EPS).mean()
      else:
        pos_loss = -torch.log(
            self.ipd(z, pos_edge_index, sigmoid=sig) + EPS).mean()

      
        neg_loss = -torch.log(1 -
                              self.ipd(z, neg_edge_index, sigmoid=sig) +
                              EPS).mean()

      return pos_loss + neg_loss


  def kl_loss(self, mu = None , logstd = None):

        mu = self.rows_encoder.__mu__ if mu is None else mu
        logstd = self.rows_encoder.__logstd__ if logstd is None else logstd
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))
  
  def test(self, z, pos_edge_index, neg_edge_index ):

        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.ipd(z, pos_edge_index, sigmoid=True)
        neg_pred = self.ipd(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)
    

  def calculate_loss(self, x ,knn_edge_index, ppi_edge_index, highly_variable_index):
    embbed = self.encoder(x, knn_edge_index, ppi_edge_index)
    embbed_rows = self.rows_encoder(embbed, ppi_edge_index)
    row_loss = self.recon_loss(embbed_rows, ppi_edge_index,sig=True)
 
    embbed_cols = self.cols_encoder(embbed.T, knn_edge_index)
    out_features = self.feature_decodr(embbed_cols)
    out_features = (out_features - (out_features.mean(axis=0)))/ (out_features.std(axis=0)+ EPS)
    reg = self.recon_loss(out_features.T, ppi_edge_index, sig=False)

    out_features =  out_features.T[highly_variable_index.values].T
    col_loss = self.feature_critarion(x[highly_variable_index.values].T, out_features)


    return self.lambda_rows * row_loss + self.lambda_cols * (col_loss + reg), row_loss, col_loss
  
  
  def forward(self, x, knn_edge_index, ppi_edge_index):
    embbed = self.encoder(x, knn_edge_index, ppi_edge_index)
    embbed_rows = self.rows_encoder(embbed, ppi_edge_index)
    embbed_cols = self.cols_encoder(embbed.T, knn_edge_index, infrance=True)
    out_features = self.feature_decodr(embbed_cols)

    return embbed_rows, embbed_cols, out_features
  
