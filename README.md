# STAN

The source code for *STAN: Spatio-Temporal Attention Network for Pandemic Prediction Using Real World Evidence*

*Our paper is a pre-print version and is also under reviewing. There are still some issues in our code and paper and we are currently fixing them*

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
3. Modify ```test/train/val_input_time``` to split training, validation and testing date.
4. When training is complete, you can use ```test.py``` to test the model performance.
