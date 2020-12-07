# STAN

The source code for *STAN: Spatio-Temporal Attention Network for Pandemic Prediction Using Real World Evidence*

*Our paper is a pre-print version and is also under reviewing. There are still some issues in our code and paper and we are currently fixing them.*

## Requirements

* Install python, pytorch. We use Python 3.7.3, Pytorch 1.1.
* Install dgl and pygsp.
* If you plan to use GPU computation, install CUDA.

## 60-day prediction results

We provide the predicted number of infected cases in 45 US states for future 60 days since 2020-06-14 using STAN. We use 5 days data from 2020-06-14 to 2020-06-18 as the testset. You can obtain the prediction results and test performance by simply load ```predictions``` using following codes:

```python
import pickle
pred = pickle.load(open('./predictions','rb'))
```

```predictions``` is a list, and each item in the list is a dictionary. The keys are: 'Name', 'MSE', 'MAE', 'ccc' and 'pred'.

## Train STAN for other locations

You can train STAN for other locations using our code. 

1. We use ```utils/Data2Graph.py``` to process time series data and generate location graphs. The edges are calculated using ```utils/calc_distance.py```.

2. You can modify hyper-parameters in the ```train.py``` file. ```slot_len``` indicates the input sliding window ```L_1```. ```nhid1``` and ```nhid2``` indicate the hidden dimension of the GCN. ```gru_dim``` indicates the GRU hidden dimension. ```num_heads``` indicates the number of graph attention heads. ```pred_horizon``` indicates the prediction window.

   We recommend you to prepare your data using your own codes. Here we provide an example to prepare the data for the model:

   ```susceptible/infected/recovered```: the sequence for susceptible/infected/recovered sequences

   ```dS/dI/dR```: the increment sequence of susceptible/infected/recovered. dS[i] = S[i]-S[i-1], dS[0] = 0

   ```h/g```: feature tensor and defined graph

   ```N```: population matrix of all locations

   The model receives feature tensor with shape `timestep, n_locs, (feature_dim*slot_len)` for `h`. `slot_len` defines the input window for each timestep. For example,  when slot_len=4, at day 5, the input is from day 1 - day 4. The model will also receives following parameters:

   `I/R/S`: the increment number of infected/recovered/susceptible at the last day of the input window from all locations. In our example, it's `dI[4],dR[4],dS[4]`

   `It/Rt`: the number of infected/recovered at the last day of the input window from all locations. In our example, it's `infected[4], recovered[4]`.

3. When training is complete, you can use ```test.py``` to test the model performance.
