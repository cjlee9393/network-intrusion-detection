import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
import argparse


class model(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, binary, multi_to_binary, dropout):
		super(model, self).__init__()
		self.input_size=input_size
		self.hidden_size=hidden_size
		self.num_layers=num_layers
		self.dropout=dropout
		
		# define variables
		embSize_proto = 5
		embSize_service = 3
		embSize_state = 2
		
		numCat_proto = 134
		numCat_service = 14
		numCat_state = 10

		# add emb layers
		self.emb_proto = nn.Embedding(numCat_proto, embSize_proto)
		self.emb_service = nn.Embedding(numCat_service, embSize_service)
		self.emb_state = nn.Embedding(numCat_state, embSize_state)
		self.inputSize_lstm = input_size + embSize_proto + embSize_service + embSize_state - 3
		
		# make lstm instance
		self.rnn = nn.LSTM(self.inputSize_lstm, self.hidden_size, self.num_layers, batch_first=True)
		self.fc1 = nn.Linear(self.hidden_size,50)
		
		if(binary == 1 and multi_to_binary == 0):
			self.fc2 = nn.Linear(50,2)
		elif(binary == 0 or (binary ==1 and multi_to_binary == 1) ):
			self.fc2 = nn.Linear(50,10)	
		#self.fc3 = nn.Linear(10,2)

		self.dropout = nn.Dropout(p=dropout)
	def forward(self, x_cont, x_cat, seq_len):
		
		# go through emb layers
		out_proto = self.emb_proto(x_cat[:, 0])
		out_service = self.emb_service(x_cat[:, 1])
		out_state = self.emb_state(x_cat[:, 2])

		x = torch.cat([out_proto, out_service, out_state], 1)
		x = torch.cat([x_cont, x], 1)

		# modify shape for lstm
		x = x.view(-1, seq_len, self.inputSize_lstm)

		lstm_output, hidden = self.rnn(x, None)

		out = F.leaky_relu(self.fc1(lstm_output))
		out = self.dropout(out)
		out = self.fc2(out)
		
		#out = self.fc3(out)

		return out
