
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import sequential, GATConv, GraphNorm, VGAE, GCNConv, InnerProductDecoder, TransformerConv, GAE,LayerNorm, SAGEConv
from torch_geometric.nn.conv import transformer_conv
from torch_geometric.utils import negative_sampling
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
import pandas as pd
EPS = 1e-15
MAX_LOGSTD = 10

class FatureDecoder(torch.nn.Module):
    def __init__(self, feature_dim, embd_dim,inter_dim , transpose_feature_dim, transpose_inter_dim, ge_dim, drop_p = 0.0, cd=True):
        super(FatureDecoder, self).__init__()
        self.feature_dim = feature_dim
        self.embd_dim = embd_dim
        self.inter_dim = inter_dim
        self.ge_dim  = ge_dim
        self.transpose_feature_dim = transpose_feature_dim
        self.transpose_inter_dim = transpose_inter_dim
        self.cd = cd
        if cd:
            self.decoder = nn.Sequential(nn.Linear(embd_dim, feature_dim),
                                          nn.BatchNorm1d(feature_dim),
                                          nn.Dropout(drop_p),
                                          nn.ReLU())
            
            self.transposer =  nn.Sequential(nn.Linear(transpose_feature_dim, transpose_inter_dim),
                                         # nn.BatchNorm1d(transpose_inter_dim),
                                          nn.Dropout(drop_p),
                                          nn.ReLU())
                                            

            self.combine_decoder = nn.Sequential(
                                          nn.Linear(transpose_inter_dim + ge_dim, transpose_feature_dim),
                                          #nn.BatchNorm1d(transpose_feature_dim),
                                          nn.Dropout(drop_p),
                                          nn.ReLU())
        else :
            self.decoder = nn.Sequential(nn.Linear(embd_dim, inter_dim),
                                           # nn.BatchNorm1d(inter_dim),
                                            nn.Dropout(drop_p),
                                            nn.ReLU(),
                                           # nn.Linear(inter_dim, inter_dim),
                                           # nn.BatchNorm1d(inter_dim),
                                            #nn.Dropout(drop_p),
                                            #nn.ReLU(),
                                            nn.Linear(inter_dim, inter_dim),
                                           # nn.BatchNorm1d(inter_dim),
                                            nn.Dropout(drop_p),
                                            nn.ReLU(),
                                            nn.Linear(inter_dim, feature_dim),
                                           # nn.BatchNorm1d(inter_dim),
                                            nn.Dropout(drop_p))
              
    def forward(self, z, ge, sigmoid =True):
        out = self.decoder(z)
        if not self.cd:
          return out 
        out = self.transposer(out.T)
        return self.combine_decoder(torch.concat([out, ge],axis=1)).T
  
class MutaelEncoder(torch.nn.Module):
  def __init__(self,col_dim, row_dim,num_layers=4, drop_p = 0.25, add_linear_rows = False, add_linear_cols = False):
    super(MutaelEncoder, self).__init__()
    self.col_dim = col_dim
    self.row_dim = row_dim
    self.add_linear_rows = add_linear_rows
    self.add_linear_cols  = add_linear_cols
    self.num_layers = num_layers

    if self.add_linear_rows:
      self.row_linear = nn.Linear(row_dim, row_dim)
    
    if self.add_linear_cols:
       self.col_linear = nn.Linear(col_dim, col_dim)

    self.rows_layers = nn.ModuleList([
      sequential.Sequential('x,edge_index', [
                                  (SAGEConv(self.row_dim, self.row_dim), 'x, edge_index -> x1'),
                                  (nn.Dropout(drop_p,inplace=False), 'x1-> x2'),
                                  nn.LeakyReLU(inplace=True),
                                 # (LayerNorm(self.row_dim),'x2-> x3')
                                ]) for _ in range(num_layers)])
    
    self.cols_layers = nn.ModuleList([
      sequential.Sequential('x,edge_index', [
                                  (SAGEConv(self.col_dim, self.col_dim), 'x, edge_index -> x1'),
                                  nn.LeakyReLU(inplace=True),
                                  (nn.Dropout(drop_p,inplace=False), 'x1-> x2'),
                                ]) for _ in range(num_layers)])
                      

  def forward(self, x, knn_edge_index, ppi_edge_index):
    
      embbded = x.clone()
      if self.add_linear_rows:
        embbded = self.row_linear(embbded)
      if self.add_linear_cols:
        embbded = self.col_linear(embbded.T).T
      
      for i in range(self.num_layers):
        embbded = self.cols_layers[i](embbded.T,knn_edge_index).T
        embbded = self.rows_layers[i](embbded, ppi_edge_index)
      
      return embbded

class GATReducerLayer(GATConv):
  def __init__(self, in_channels, out_channels, heads=1, dropout=0 , add_self_loops=True,scale_param =2,  **kwargs):
     super().__init__(in_channels, out_channels, heads, dropout, add_self_loops, **kwargs)
     self.treshold_alpha = None
     self.scale_param = scale_param
  
  def edge_update(self, alpha_j, alpha_i,
                    edge_attr, index, ptr,
                    size_i):
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i

        if edge_attr is not None and self.lin_edge is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge

        #alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.sigmoid(alpha)
        if not self.scale_param is None:
        #  alpha = alpha / ((1/self.scale_param) * alpha.std())
          alpha = F.sigmoid(alpha)
        else:
          alpha = softmax(alpha, index, ptr, size_i)
        self.treshold_alpha = alpha 
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

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
        #print(alpha)
        if not self.scale_param is None:
          alpha = alpha - alpha.mean()
          alpha = alpha / ((1/self.scale_param) * alpha.std())
          alpha = F.sigmoid(alpha)
         # alpha = F.leaky_relu(alpha)
        else:
          alpha = softmax(alpha, index, ptr, size_i)
        #self.treshold_alpha = alpha.clone().detach()
        #self.treshold_alpha -= self.treshold_alpha.mean() 
        #self.treshold_alpha /= ((1/self.scale_param) * self.treshold_alpha.std())
        #self.treshold_alpha = F.sigmoid(self.treshold_alpha) 
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
                                  # (GraphNorm(self.feature_dim),'x->x'),
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
      

      def reduce_network(self, threshold = 0.2, min_connect=5):
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

class DimVariationalEmcoder(torch.nn.Module):
      def __init__(self,feature_dim, inter_dim, embd_dim,drop_p = 0.0):
        super(DimVariationalEmcoder, self).__init__()
        self.feature_dim = feature_dim
        self.embd_dim = embd_dim
        self.inter_dim = inter_dim
        self.__logstd__ = None
        self.__mu__ = None

        self.encoder = sequential.Sequential('x, edge_index', [
                                   # (GraphNorm(self.feature_dim),'x->x'),
                                    (GCNConv(self.feature_dim, self.inter_dim), 'x, edge_index -> x1'),
                                    nn.LeakyReLU(inplace=True),
                                    (nn.Dropout(drop_p,inplace=False), 'x1-> x2')
                                  ])
        self.mu = TransformerConv(self.inter_dim, self.embd_dim,dropout= drop_p,add_self_loops = True)
        self.logstd = TransformerConv(self.inter_dim, self.embd_dim,dropout= drop_p, add_self_loops = True)

  
      def forward(self, x, edge_index):
        embbded = x.clone()
        embbded = self.encoder(embbded,edge_index)
        self.__mu__, self.__logstd__ = self.mu(embbded, edge_index), self.logstd(embbded, edge_index)
        self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD)
        return  self.__mu__,  self.__logstd__
      
      
      def encode(self, mu=None, logstd=None):
        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd
        return self.__mu__ + torch.randn_like(self.__logstd__) * torch.exp(self.__logstd__)

class scNET(torch.nn.Module):
  def __init__(self,col_dim, row_dim,inter_row_dim, embd_row_dim, inter_col_dim,embd_col_dim,
                lambda_rows = 1, lambda_cols = 1, num_layers=2, drop_p = 0.25, add_linear_row = True,vae_flag=False, add_linear_col = False,cd=False):

    super(scNET, self).__init__()
    self.col_dim = col_dim
    self.row_dim = row_dim
    self.inter_row_dim = inter_row_dim
    self.embd_row_dim = embd_row_dim
    self.inter_col_dim = inter_col_dim
    self.embd_col_dim = embd_col_dim
    self.lambda_rows = lambda_rows
    self.lambda_cols = lambda_cols
    self.vae_flag = vae_flag
    self.cd = cd


    self.encoder = MutaelEncoder(col_dim, row_dim,num_layers, drop_p, add_linear_row, add_linear_col)
    self.rows_encoder =  DimEncoder(row_dim, inter_row_dim, embd_row_dim,drop_p = drop_p, scale_param=None, reducer=False)

    if vae_flag:
        self.cols_encoder =  DimVariationalEmcoder(col_dim, inter_col_dim, embd_col_dim,drop_p)
    else:
        self.cols_encoder =  DimEncoder(col_dim, inter_col_dim, embd_col_dim,drop_p=drop_p, reducer=True)


    #self.cols_encoder =  DimEncoder(col_dim, inter_col_dim, embd_col_dim,drop_p=drop_p, reducer=True)
    self.feature_decodr = FatureDecoder(col_dim, embd_col_dim, inter_col_dim, row_dim, inter_row_dim, embd_row_dim, drop_p = 0.2,cd=cd)
    self.ipd = InnerProductDecoder()
    self.feature_critarion = nn.MSELoss(reduction ='mean')

  def recon_loss(self, z, pos_edge_index, neg_edge_index = None) :
      
      pos_loss = -torch.log(
          self.ipd(z, pos_edge_index, sigmoid=True) + EPS).mean()

      if neg_edge_index is None:
          neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
      neg_loss = -torch.log(1 -
                            self.ipd(z, neg_edge_index, sigmoid=True) +
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
    row_loss = self.recon_loss(embbed_rows, ppi_edge_index)
 
    if self.vae_flag:
      mu, logstd = self.cols_encoder(embbed.T, knn_edge_index)
      embbed_cols = self.cols_encoder.encode(mu,logstd)
      out_features = self.feature_decodr(embbed_cols,embbed_rows)
      out_features =  out_features.T[highly_variable_index.values].T
      out_features = (out_features - (out_features.mean(axis=0)))/ (out_features.std(axis=0)+ EPS)
      col_loss = self.feature_critarion(x[highly_variable_index.values].T, out_features)
      col_loss +=  (5 / x.shape[1]) *  self.kl_loss(mu, logstd)
    else:
      #embbed_cols = self.cols_encoder(embbed.T, knn_edge_index)
      #out_features = self.feature_decodr(embbed_cols,embbed_rows)
      #out_features =  out_features.T[highly_variable_index.values].T
      #out_features = (out_features - (out_features.mean(axis=0)))/ (out_features.std(axis=0)+ EPS)
      #col_loss = self.feature_critarion(x[highly_variable_index.values].T, out_features)
      # if self.cd:
      embbed_cols = self.cols_encoder(embbed.T, knn_edge_index)
      out_features = self.feature_decodr(embbed_cols,embbed_rows)
      out_features = (out_features - (out_features.mean(axis=0)))/ (out_features.std(axis=0)+ EPS)
      #col_loss = 0.1*self.recon_loss(out_features.T, target_edge_index)
    #  out_features = out_features * (~zero_mask).T
    #  x = x* (~zero_mask)
      out_features =  out_features.T[highly_variable_index.values].T
      #out_features = (out_features - (out_features.mean(axis=0)))/ (out_features.std(axis=0)+ EPS)
      col_loss = self.feature_critarion(x[highly_variable_index.values].T, out_features)
   

    return self.lambda_rows * row_loss + self.lambda_cols * col_loss, row_loss, col_loss
  
  
  def forward(self, x, knn_edge_index, ppi_edge_index):
    embbed = self.encoder(x, knn_edge_index, ppi_edge_index)
    embbed_rows = self.rows_encoder(embbed, ppi_edge_index)

    if self.vae_flag:
      mu, logstd = self.cols_encoder(embbed.T, knn_edge_index)
      embbed_cols = self.cols_encoder.encode(mu,logstd)
    else:
       embbed_cols = self.cols_encoder(embbed.T, knn_edge_index, infrance=True)

    #out_netowrk = self.ipd(embbed_rows, ppi_edge_index)
    #out_features = self.feature_decodr(embbed_cols,embbed_rows)
    return embbed_rows, embbed_cols
  
