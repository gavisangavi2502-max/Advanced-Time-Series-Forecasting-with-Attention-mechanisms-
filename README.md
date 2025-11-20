# Improved Advanced Time Series Forecasting Project
Includes Transformer, LSTM, ARIMA baselines, attention extraction, evaluation (RMSE/MAE/MAPE), and simple hyperparameter search.
Directory structure:
└── gavisangavi2502-max-advanced-time-series-forecasting-with-attention-mechanisms-/
    ├── README.md
    ├── attention_extract.py
    ├── baselines.py
    ├── model.py
    ├── train.py
    └── utils.py


Files Content:

================================================
FILE: README.md
================================================
# Improved Advanced Time Series Forecasting Project
Includes Transformer, LSTM, ARIMA baselines, attention extraction, evaluation (RMSE/MAE/MAPE), and simple hyperparameter search.



================================================
FILE: attention_extract.py
================================================
# placeholder: save attention weights



================================================
FILE: baselines.py
================================================
from statsmodels.tsa.arima.model import ARIMA
import torch, torch.nn as nn
class LSTMForecaster(nn.Module):
    def __init__(self,input_dim=1,hidden=64,layers=2,pred_len=24):
        super().__init__();self.lstm=nn.LSTM(input_dim,hidden,layers,batch_first=True);self.fc=nn.Linear(hidden,pred_len)
    def forward(self,x):o,_=self.lstm(x);o=o[:,-1,:];return self.fc(o).unsqueeze(-1)



================================================
FILE: model.py
================================================
# transformer same as earlier placeholder



================================================
FILE: train.py
================================================
# placeholder training integrating all models



================================================
FILE: utils.py
================================================
# windowing utilities


