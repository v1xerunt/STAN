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


# train, val, test split along the spatial dimension

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
test_d_active =  active_cases[test_set,:][:,test_pred_time].to(device)
test_labels_confirmed = confirmed_cases[test_set,:][:,test_pred_time].to(device)
test_d_recovered = d_recovered[test_set,:][:,test_pred_time].to(device)


for target_loc in range(45):
  print('-----------Start training for loc %d-----------'%target_loc)
  num_train_steps = len(train_input_time)-slot_len+1
  batch_feat = []
  batch_active = []
  batch_recover = []
  batch_I = []
  batch_R = []
  batch_S = []
  batch_It = []
  batch_Rt = []

  for t in range(0, num_train_steps, step_size):
    t_feat = train_features[:,t:t+slot_len,:].view(int(num_train),slot_len*D).float()
    t_active=train_d_active[:][target_loc,t:t+pred_horizon].float()
    t_recovered = train_d_recovered[:][target_loc,t:t+pred_horizon].float()
    last_I = active_cases[target_loc,train_pred_time[t]-1].unsqueeze(-1).float().to(device)
    last_R = recovered_cases[target_loc,train_pred_time[t]-1].unsqueeze(-1).float().to(device)
    last_S = popn[target_loc,0].to(device).float().unsqueeze(-1) - last_I - last_R

    batch_It.append(d_active[target_loc, train_pred_time[t]-1].float())
    batch_Rt.append(d_recovered[target_loc, train_pred_time[t]-1].float())

    batch_feat.append(t_feat)
    batch_active.append(t_active)
    batch_recover.append(t_recovered)
    batch_I.append(last_I)
    batch_R.append(last_R)
    batch_S.append(last_S)


  batch_It = torch.stack(batch_It).to(device).squeeze()
  batch_Rt = torch.stack(batch_Rt).to(device).squeeze()
  batch_feat = torch.stack(batch_feat).to(device)
  batch_active = torch.stack(batch_active).to(device)
  batch_recover = torch.stack(batch_recover).to(device)
  batch_I = torch.stack(batch_I).to(device).squeeze()
  batch_R = torch.stack(batch_R).to(device).squeeze()
  batch_S = torch.stack(batch_S).to(device).squeeze()

  valid_feat = []
  valid_active = []
  valid_recover = []
  valid_I = []
  valid_R = []
  valid_S = []
  valid_It = []
  valid_Rt = []

  num_val_steps = val_features.shape[1]-slot_len+1
  for t in range(0, num_val_steps, step_size):
    t_feat = val_features[:,t:t+slot_len,:].view(int(num_train),slot_len*D).float()
    t_active=val_d_active[:][target_loc,t:t+pred_horizon].float()
    t_recovered = val_d_recovered[:][target_loc,t:t+pred_horizon].float()

    last_I = active_cases[target_loc,val_pred_time[t]-1].unsqueeze(-1).float().to(device)
    last_R = recovered_cases[target_loc,val_pred_time[t]-1].unsqueeze(-1).float().to(device)
    last_S = popn[target_loc,0].to(device).float().unsqueeze(-1) - last_I - last_R

    valid_It.append(d_active[target_loc, val_pred_time[t]-1].float())
    valid_Rt.append(d_recovered[target_loc, val_pred_time[t]-1].float())

    valid_feat.append(t_feat)
    valid_active.append(t_active)
    valid_recover.append(t_recovered)
    valid_I.append(last_I)
    valid_R.append(last_R)
    valid_S.append(last_S)

  valid_It = torch.stack(valid_It).to(device).flatten()
  valid_Rt = torch.stack(valid_Rt).to(device).flatten()
  valid_feat = torch.stack(valid_feat).to(device)
  valid_active = torch.stack(valid_active).to(device)
  valid_I = torch.stack(valid_I).to(device).flatten()
  valid_R = torch.stack(valid_R).to(device).flatten()
  valid_S = torch.stack(valid_S).to(device).flatten()



  # training the model
  model_name = state_names[target_loc]
  model=model.GAT(g, nfeat, nhid1, nhid2, gru_dim, N, num_heads, pred_horizon,attn_dim).to(device)
  model = model.float()
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)

  criterion = nn.MSELoss(reduction='sum')

  for epoch in range(num_epochs):

    model.train()
    optimizer.zero_grad()

    active_pred, recovered_pred, phy_active, phy_recover = model(batch_feat, popn[target_loc,0].to(device).float(), batch_I, batch_R,batch_S,batch_It,batch_Rt) # forward pass

    loss = F.mse_loss(active_pred,batch_active)+F.mse_loss(recovered_pred,batch_recover)#+F.mse_loss(phy_active, batch_active.flatten())+F.mse_loss(phy_recover, batch_recover.flatten())

    loss.backward()
    optimizer.step()

    #Evaluate
    model.eval()
    with torch.no_grad():
      active_val, recovered_val, vphy_active, vphy_recover = model(valid_feat, popn[target_loc,0].to(device).float(), valid_I, valid_R,valid_S,valid_It,valid_Rt) # forward pass
      mse = sklearn.metrics.mean_squared_error(valid_active.cpu().detach().flatten().numpy(),active_val.cpu().detach().numpy())
      ccc = model.ccc(valid_active.squeeze().cpu().detach().flatten().numpy(),active_val.squeeze().cpu().detach().flatten().numpy())
      r2 = sklearn.metrics.r2_score(valid_active.cpu().detach().flatten().numpy(), active_val.cpu().detach().flatten().numpy())

    if epoch==0:
      min_loss = mse
      min_r2 = r2
      torch.save(model.state_dict(),model_name)
    if mse < min_loss:
      min_loss = mse
      min_r2 = r2
      torch.save(model.state_dict(),model_name)
    if epoch % 100 == 0:
      print("Epoch %d, Train loss %f, Val loss %f, Val R2 %f, Val ccc %f, %.2f, %.2f"%(epoch, loss, mse, np.mean(r2), np.mean(ccc), model.alpha_list, model.beta_list))
