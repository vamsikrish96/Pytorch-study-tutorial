#LSTM implementation using pytorch 
# Study tutorial of pytorch while implementing LSTM model for time-series data.

import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sample data for study purpose
data=np.arange(1,101)

from sklearn.preprocessing import MinMaxScaler


xtrain=data[:80]
test_data=data[80:100]

scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_normalized = scaler.fit_transform(xtrain .reshape(-1, 1))
train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)
train_window = 10

def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq
train_inout_seq = create_inout_sequences(train_data_normalized, train_window)


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        lstm_out2, self.hidden_cell = self.lstm(predictions.view(len(predictions) ,1, -1), self.hidden_cell)
        predictions2 = self.linear(lstm_out.view(len(predictions), -1))
        predictions_softmax2=F.log_softmax(predictions2)
        return predictions_softmax2

model = LSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def running_problem(train_inout):
    asdf=[]
    for seq, labels in train_inout:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))

        y_pred = model(seq)
        print("y_pred",y_pred)
        asdf.append((y_pred,y_pred[-1]))
        with torch.autograd.set_detect_anomaly(True):
             y_pred=torch.rand(1,requires_grad=True)
        single_loss = loss_function(y_pred, labels)
        single_loss.backward(retain_graph=True)
        optimizer.step()
        return asdf
        
for i in range(2):
    out_val=running_problem(train_inout_seq)
    train_inout_seq=out_val
    print("train_inout_seq",train_inout_seq)
