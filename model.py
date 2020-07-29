import torch
import torch.nn as nn
import torch.nn.functional as F
from pygsp import graphs, filters, reduction
import sklearn
import sklearn.metrics
import numpy as np
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g.to(device)
        self.fc = nn.Linear(in_dim, out_dim, bias=False).to(device)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False).to(device)

    def edge_attention(self, edges):
        inter=torch.diag(torch.mm(edges.src['z'].to(device), torch.transpose(edges.dst['z'].to(device),0,1) ))
        norm_z1 = torch.diag(torch.sqrt(torch.mm(edges.src['z'].to(device),torch.transpose(edges.src['z'].to(device),0,1))))
        norm_z2 = torch.diag(torch.sqrt(torch.mm(edges.dst['z'].to(device),torch.transpose(edges.dst['z'].to(device),0,1))))

        a = inter

        return {'e':a.to(device)}

    def message_func(self, edges):
        return {'z': edges.src['z'].to(device), 'e': edges.data['e'].to(device)}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1).to(device)
        h = torch.mm(alpha, nodes.mailbox['z'][0,:,:]).to(device)
        return {'h': h}

    def forward(self, h):
        z = self.fc(h.to(device)).to(device)
        self.g.ndata['z'] = z.to(device)
        self.g.apply_edges(self.edge_attention)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h').to(device)


def edge_attention(self, edges):
    z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
    a = self.attn_fc(z2)
    return {'e' : F.leaky_relu(a)}
  


def reduce_func(self, nodes):
    alpha = F.softmax(nodes.mailbox['e'], dim=1)
    h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
    return {'h' : h}

class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge

    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))

class GAT(nn.Module):
    def __init__(self, g, in_dim, hidden_dim1, hidden_dim2, gru_dim, num_locs, num_heads, pred_horizon, input_attn_dim):
        super(GAT, self).__init__()
        
        self.Wq = torch.eye(in_dim)
        self.ig = g.to(device)
        
        
        self.layer1 = MultiHeadGATLayer(g, in_dim, hidden_dim1, num_heads)
        self.layer2 = MultiHeadGATLayer(g, hidden_dim1 * num_heads, hidden_dim2, 1)
        
        self.num_locs = num_locs

        self.pred_horizon = pred_horizon
        
        self.gru = nn.GRUCell(hidden_dim2, gru_dim)
        
        self.nn_res1 = nn.Linear(gru_dim+2,2*pred_horizon)
        self.nn_res2 = nn.Linear(gru_dim+2,2)
        self.hidden_dim2 = hidden_dim2
        self.gru_dim = gru_dim
  
    def input_attention(self, edges):
      
        dot_p=torch.diag(torch.mm(edges.src['iz'].to(device), torch.transpose(edges.dst['iz'].to(device),0,1) ))
        norm_h1 = torch.diag(torch.sqrt(torch.mm(edges.src['iz'].to(device),torch.transpose(edges.src['iz'].to(device),0,1))))
        norm_h2 = torch.diag(torch.sqrt(torch.mm(edges.dst['iz'].to(device),torch.transpose(edges.dst['iz'].to(device),0,1))))
        
        
        iz2=(dot_p/norm_h1/norm_h2)**4
      
        return {'e': iz2.to(device)}

    def input_message_func(self, edges):
        return {'iz': edges.src['iz'].to(device), 'e': edges.data['e'].to(device)}

    def input_reduce_func(self, nodes):
        alpha = nodes.mailbox['e'].to(device)
        ih = torch.mm(alpha, nodes.mailbox['iz'][0,:,:]).to(device)
        return {'ih': ih}  
  

    def forward(self, h, N, I, R, S, It, Rt):
        T = h.size(0)
        N = N.squeeze()
        
        hx = torch.randn(1, self.gru_dim).to(device)
        
        new_I = []
        new_R = []
        phy_I = []
        phy_R = []
        self.alpha_list = []
        self.beta_list = []
        self.alpha_scaled = []
        self.beta_scaled = [] 
        
        for each_step in range(T):        
          iz=h[each_step,:]
          self.ig.ndata['iz'] = iz.to(device)
          self.ig.apply_edges(self.input_attention)
          self.ig.update_all(self.input_message_func, self.input_reduce_func)
          ih=self.ig.ndata.pop('ih').to(device)
        
          cur_h = self.layer1(ih)
          cur_h = F.relu(cur_h)
          cur_h = self.layer2(cur_h)
          cur_h = torch.max(F.relu(cur_h), 0)[0].reshape(1, self.hidden_dim2)
          hx = self.gru(cur_h,hx)
          
          new_hx = torch.cat((hx, It[each_step].reshape(1,1), Rt[each_step].reshape(1,1)),dim=1)
          
          pred_res = self.nn_res1(new_hx).squeeze()
          alpha,beta = self.nn_res2(new_hx).squeeze()
          self.alpha_list.append(alpha)
          self.beta_list.append(beta)
          alpha = torch.sigmoid(alpha)
          beta = torch.sigmoid(beta)
          self.alpha_scaled.append(alpha)
          self.beta_scaled.append(beta)
          
          I_idx = [2*i for i in range(self.pred_horizon)]
          R_idx = [2*i+1 for i in range(self.pred_horizon)]
          new_I.append(pred_res[I_idx])
          new_R.append(pred_res[R_idx])
          
          for i in range(self.pred_horizon):
            last_I = I[each_step] if i == 0 else last_I + dI.detach()
            last_R = R[each_step] if i == 0 else last_R + dR.detach()
            last_S = S[each_step] if i == 0 else N - last_I - last_R
            dI = alpha * last_I * (last_S/N) - beta * last_I
            dR = beta * last_I #1
            phy_I.append(dI)
            phy_R.append(dR)
          
        new_I = torch.stack(new_I).to(device).squeeze()
        new_R = torch.stack(new_R).to(device).squeeze()
        phy_I = torch.stack(phy_I).to(device).squeeze()
        phy_R = torch.stack(phy_R).to(device).squeeze()
        
        self.alpha_list = torch.stack(self.alpha_list).to(device).squeeze()
        self.beta_list = torch.stack(self.beta_list).to(device).squeeze()
        self.alpha_scaled = torch.stack(self.alpha_scaled).to(device).squeeze()
        self.beta_scaled = torch.stack(self.beta_scaled).to(device).squeeze()
        return new_I, new_R, phy_I, phy_R

def ccc(y_true, y_pred):
    cor = np.corrcoef(y_true, y_pred)[0][1]
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    return numerator / denominator