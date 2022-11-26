import torch
from torchvision import transforms
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
		
class custom_dataset(Dataset):
	def __init__(self, which_data, csv_path, binary,multi_to_binary,batch_size,seq_len,valid_ratio,train,scaler_=None):
		super(custom_dataset, self).__init__()
		
		if which_data == 'nsl_kdd' or which_data =='kdd': #kdd
			self.data_info =  pd.read_csv(csv_path).drop(["id"], axis=1)
		
		elif which_data =='unsw': #usnw	
			# self.data_info =  pd.read_csv(csv_path).drop(["trash", "id", "attack_cat"], axis=1)
			self.data_info =  pd.read_csv(csv_path).drop(["attack_cat"], axis=1)
		cols_cat = ["proto", "service", "state"] # categorical data
		cols_target = ["label"]

		self.data_cat = self.data_info[cols_cat]
		self.data_target = self.data_info[cols_target].squeeze()
		self.data_cont = self.data_info.drop(cols_cat + cols_target, axis=1)
		
		col_len = len(self.data_info.columns)
		row_len = len(self.data_info.index)
	
		# convert to numpy
		self.data_cont = np.asarray(self.data_cont, dtype=np.float32)
		self.data_cat = np.asarray(self.data_cat, dtype=np.long)
		self.data_target = np.asarray(self.data_target, dtype=np.long)
		
		##
		self.scaler = 0
		if scaler_ != None:
			self.scaler = scaler_
		else:
			self.scaler = StandardScaler().fit(self.data_cont)
		
		self.data_cont = self.scaler.transform(self.data_cont)
		
		self.data_len = len(self.data_info)
		self.binary = binary
		self.multi_to_binary = multi_to_binary

		print("data len",self.data_len)

	def __getitem__(self, index):
		data_cont = self.data_cont[index]
		data_cat = self.data_cat[index]
		target = self.data_target[index]

		if self.binary == 1 and self.multi_to_binary ==0:
			if target != 6:
				target = 1
			elif target == 6:
				target = 0

		return data_cont, data_cat, target

	def __len__(self):
		return self.data_len

	def __getscaler__(self):
		return self.scaler

class gen_dataset_iter(Dataset):
	def __init__(self, which_data, binary,multi_to_binary, valid_ratio, batch_size,seq_len):
		super(gen_dataset_iter, self).__init__()
		
		self.dataset = []
		
		if which_data == 'nsl_kdd' or which_data =='kdd': # kdd
			train_data = custom_dataset(which_data,'~/repos/processed_dataset/convertedKDDTrain+_c.csv', binary,multi_to_binary,batch_size,seq_len,valid_ratio,True,None)
			test_data = custom_dataset(which_data,'~/repos/processed_dataset/convertedKDDTest+_c.csv', binary,multi_to_binary, batch_size,seq_len,valid_ratio,False,scaler_= train_data.__getscaler__())

		elif which_data =='unsw': #usnw	
			train_data = custom_dataset(which_data,'./data/convertedUNSW_NB15_training-set_c.csv',binary,multi_to_binary, batch_size,seq_len,valid_ratio,True,None)
			test_data = custom_dataset(which_data,'./data/convertedUNSW_NB15_testing-set_c.csv', binary,multi_to_binary, batch_size,seq_len,valid_ratio,False,scaler_= train_data.__getscaler__())
		
		else:
			pass

		block_size = seq_len

		train_row_len = len(train_data)
		train_indices = list(range(len(train_data))) # original train data
		train_block_indices = list(range(0,train_row_len - block_size + 1,1)) # overlap data
		##
		np.random.seed(1) # shuffle seed
		##
		np.random.shuffle(train_block_indices)
			
		train_block_list = []
		for i in train_block_indices:
			train_block_list.append(torch.utils.data.Subset(train_data, train_indices[i:i+block_size]))
		train_shuffled_block = torch.utils.data.ConcatDataset(train_block_list)

		splitted_number = len(train_shuffled_block)/block_size
		train_size = int(splitted_number*(1-valid_ratio))
		valid_size = splitted_number - train_size
		split_indices = list(range(len(train_shuffled_block)))
		train_data = torch.utils.data.Subset(train_shuffled_block, split_indices[:train_size*block_size])
		valid_data = torch.utils.data.Subset(train_shuffled_block, split_indices[train_size*block_size:])


		test_row_len = len(test_data)
		test_indices = list(range(len(test_data)))
		test_block_indices = list(range(0,test_row_len - block_size +1,1))
		test_block_list = []
		for i in test_block_indices:
			test_block_list.append(torch.utils.data.Subset(test_data,test_indices[i:i+block_size]))
		test_shuffled_block = torch.utils.data.ConcatDataset(test_block_list)
		test_data = test_shuffled_block

		
		train_iter= DataLoader(train_data, batch_size=batch_size*block_size,shuffle=False)
		valid_iter= DataLoader(valid_data, batch_size=batch_size*block_size,shuffle=False)
		test_iter= DataLoader(test_data, batch_size=batch_size*block_size,shuffle=False)
	
		self.dataset.append(train_iter)
		self.dataset.append(valid_iter)
		self.dataset.append(test_iter)
		

			
	def get(self):
		return self.dataset
