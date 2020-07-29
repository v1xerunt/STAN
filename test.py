import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from utils import Data2Graph
import scipy.stats
from dgl import DGLGraph
import sklearn
import model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_filename='FILEPATH'
auxiliary_filename='uszips.csv'

slot_len = 6
pred_horizon = 60
step_size = 5
mid_time = (pred_horizon//2)+1

nhid1 = 400
nhid2 = 200
gru_dim = 100
num_heads = 3
lr = 1e-3
r = 1

num_epochs = 2000

feat_tensor, Adj, active_cases, confirmed_cases, popn, selected_counties,g = Data2Graph.load_data(data_filename, auxiliary_filename,active_thresh=1000,r=r)
recovered_cases = confirmed_cases - active_cases
state_names=np.array(selected_counties['state'])

d_active = []
d_recovered = []
for i in range(active_cases.size(0)):
  i_active = [0]
  i_recovered = [0]
  for j in range(1,active_cases.size(1)):
    i_active.append(active_cases[i,j]-active_cases[i,j-1])
    i_recovered.append(recovered_cases[i,j]-recovered_cases[i,j-1])
  d_active.append(i_active)
  d_recovered.append(i_recovered)
    
d_active = torch.tensor(d_active).to(device)
d_recovered = torch.tensor(d_recovered).to(device)
    

N,T,D = feat_tensor.shape
nfeat = slot_len*D
nclass = pred_horizon*3
attn_dim=int(nfeat*1.5)


num_pred_time = len(range(slot_len,T))

num_val_pred_time = 60
num_test_pred_time = 5
num_train_pred_time = num_pred_time-num_val_pred_time - num_test_pred_time

test_pred_time=np.arange(T-num_test_pred_time,T)
val_pred_time=np.arange(test_pred_time[0]-num_val_pred_time,test_pred_time[0])
train_pred_time = np.arange(slot_len,val_pred_time[0]+pred_horizon-1)

train_input_time = np.arange(train_pred_time[-1]+1-pred_horizon)
val_input_time = np.arange(val_pred_time[0]-slot_len,val_pred_time[-1]+1-pred_horizon)
test_input_time = np.array([78,79,80,81,82,83])

num_train = int(1*N)
num_val = int(0*N)
num_test=N-num_train-num_val

#shuffle_order = np.random.permutation(N)
shuffle_order=np.arange(N)

train_locs = np.arange(N)[shuffle_order[:num_train]]
val_locs = np.arange(N)[shuffle_order[num_train:num_train+num_val]]
test_locs = np.arange(N)[shuffle_order[num_train+num_val:]]


# data splitting

W_train=Adj[train_locs,:][:,train_locs].to(device)                        
train_features=feat_tensor[train_locs,:,:][:,train_input_time,:][:].to(device)
train_labels_active = active_cases[train_locs,:][:,train_pred_time].to(device)
train_d_active = d_active[train_locs,:][:,train_pred_time].to(device)
train_labels_confirmed = confirmed_cases[train_locs,:][:,train_pred_time].to(device)
train_labels_recovered = recovered_cases[train_locs,:][:,train_pred_time].to(device)
train_d_recovered = d_recovered[train_locs,:][:,train_pred_time].to(device)
train_popn=popn[train_locs,:][:].to(device)


# For validation and testing, we use the full training data points as well

val_set = np.concatenate((train_locs, val_locs),axis=0)

W_val=Adj[val_set,:][:,val_set].to(device)
val_features = feat_tensor[val_set,:,:][:, val_input_time,:][:].to(device)
val_labels_active = active_cases[val_set,:][:,val_pred_time].to(device)
val_d_active = d_active[val_set,:][:,val_pred_time].to(device)
val_labels_confirmed = confirmed_cases[val_set,:][:,val_pred_time].to(device)
val_popn=popn[val_set,:][:].to(device)
val_d_recovered = d_recovered[val_set,:][:,val_pred_time].to(device)

# For testing, we use the full training set as well for graph based inference 

test_set = np.concatenate((train_locs, test_locs),axis=0)

W_test=Adj[test_set,:][:,test_set].to(device)
test_features = feat_tensor[test_set, :,:][:,test_input_time].to(device)
test_labels_active = d_active[test_set,:][:,test_pred_time].to(device)
test_d_active =  d_active[test_set,:][:,test_pred_time].to(device)
test_labels_confirmed = confirmed_cases[test_set,:][:,test_pred_time].to(device)
test_d_recovered = d_recovered[test_set,:][:,test_pred_time].to(device)

headers = ['Name','MSE','MAE','r2','ccc','beta','gamma']
data = []

active_pred = []
mse_list = []
mae_list  =[]
ccc_list = []

for target_loc in range(45):
  cur_data = {}
  
  model_name = str(state_names[target_loc])
  net_opt=phy_model.GAT(g, nfeat, nhid1, nhid2, gru_dim, N, num_heads, pred_horizon,attn_dim).to(device)
  net_opt.load_state_dict(torch.load(model_name))   

  test_feat = []
  test_active = []
  test_recover = []
  test_I = []
  test_R = []
  test_S = []
  test_It = []
  test_Rt = []

  num_test_steps = test_features.shape[1]-slot_len+1
  for t in range(0, num_test_steps, 20):
    t_feat = test_features[:,t:t+slot_len,:].view(int(num_train),slot_len*D).float()
    t_active=test_d_active[:][target_loc,t:t+pred_horizon].float()
    t_recovered = test_d_recovered[:][target_loc,t:t+pred_horizon].float()

    last_I = active_cases[target_loc,test_pred_time[t]-1].unsqueeze(-1).float().to(device)
    last_R = recovered_cases[target_loc,test_pred_time[t]-1].unsqueeze(-1).float().to(device)
    last_S = popn[target_loc,0].to(device).float().unsqueeze(-1) - last_I - last_R

    test_It.append(d_active[target_loc, test_pred_time[t]-1].float())
    test_Rt.append(d_recovered[target_loc, test_pred_time[t]-1].float())

    test_feat.append(t_feat)
    test_active.append(t_active)
    test_recover.append(t_recovered)
    test_I.append(last_I)
    test_R.append(last_R)
    test_S.append(last_S)

  test_It = torch.stack(test_It).to(device).flatten()
  test_Rt = torch.stack(test_Rt).to(device).flatten()
  test_feat = torch.stack(test_feat).to(device)
  test_active = torch.stack(test_active).to(device)
  test_I = torch.stack(test_I).to(device).flatten()
  test_R = torch.stack(test_R).to(device).flatten()
  test_S = torch.stack(test_S).to(device).flatten()

  net_opt.eval()
  with torch.no_grad():
      active_test, recovered_test, tphy_active, tphy_recover = net_opt(test_feat, popn[target_loc,0].to(device).float(), test_I, test_R,test_S,test_It,test_Rt) # forward pass
      active_pred.append(active_test.cpu().detach().numpy())
      sum_I = [last_I[0].cpu().detach().numpy()]
      for i in range(60):
        sum_I.append(sum_I[-1]+active_test[i].cpu().detach().numpy())
  
      sum_I = np.array(sum_I)[1:]
      mse = sklearn.metrics.mean_squared_error(sum_I[:5],active_cases[target_loc, -5:].numpy())
      mae = sklearn.metrics.mean_absolute_error(sum_I[:5],active_cases[target_loc, -5:].numpy())
      
      ccc = phy_model.ccc(sum_I[:5],active_cases[target_loc, -5:].numpy())
      r2 = sklearn.metrics.r2_score(sum_I[:5],active_cases[target_loc, -5:].numpy())
      
      alpha = float(net_opt.alpha_scaled.cpu().detach().data)
      beta = float(net_opt.beta_scaled.cpu().detach().data)
      
  cur_data['Name'] = state_names[target_loc]
  cur_data['pred'] = sum_I
  cur_data['MSE'] = mse
  cur_data['MAE'] = mae
  cur_data['ccc'] = ccc
  cur_data['r2'] = r2
  data.append(cur_data)
  mse_list.append(mse)
  mae_list.append(mae)
  ccc_list.append(ccc)

print(np.mean(mse_list))
print(np.mean(mae_list))
print(np.mean(ccc_list))
  