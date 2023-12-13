# STAN

The source code for *STAN: Spatio-Temporal Attention Network for Pandemic Prediction Using Real World Evidence*

**We also recommend you check the HOIST model (https://github.com/v1xerunt/HOIST), which could achieve higher prediction performance in terms of spatio-temporal predictions.**

## Requirements

* Install python, pytorch. We use Python 3.7.3, Pytorch 1.1.
* Install dgl (if you use CUDA, please install cuda version of dgl), epiweeks, haversine.
* If you plan to use GPU computation, install CUDA.

## Train STAN with public JHU data

We provide a sample code in ```train_stan.ipynb``` jupyter notebook file to train STAN with publicly available COVID statistics. The data will be downloaded from the COVID-19 Data Repository by the Center for Systems Science and Engineering (CSSE) at Johns Hopkins University (https://github.com/CSSEGISandData/COVID-19).

In the notebook file, you can specify the date range in 

```python
GenerateTrainingData().download_jhu_data('2020-05-01', '2020-12-01')
```

You can modify the detailed data settings. ```valid_window``` and ```test_window``` indicates the date range used from validation and testing. ```history_window```, ```pred_window``` and ```slide_step``` indicate sliding window settings for data inputs and prediction outputs. 

For hyperparameters of the STAN model, ```gru_dim``` indicates the GRU hidden dimension. ```num_heads``` indicates the number of graph attention heads. ```hidden_dim``` indicates the hidden dimension of the GAT layer.

We use another setting ```normalize``` to decide whether to do the data normalization. With data normalization, model can be fitted in fewer epochs and the estimated $\beta$ and $\gamma$ of the SIR equation is more stable. However, we also find that without normalization, the model may get higher accuracy.

Once finished training, you can get the estimated $\beta$ and $\gamma$ of the SIR equation used in our model by using ```model.alpha_scaled``` and ```model.beta_scaled```.

We have noticed some data quality issues in the JHU data, for example zero cases in some days. This may significantly affect model's performance, especially when the input features and training objectives are purely from the JHU data. We're still working on fixing these issues.
