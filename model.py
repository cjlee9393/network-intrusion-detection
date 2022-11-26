import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import pandas as pd

#import kdd_mlp
#import unsw_mlp
#import kdd_rnn
import unsw_rnn
import load_dataset

import argparse
import csv
import time

from sklearn.metrics import f1_score

parser = argparse.ArgumentParser(description='collect some hyperparameters.')
parser.add_argument("-name", "--file_name", required=True, type=str)
parser.add_argument("-batch", "--batch_size", required=True, type=int,  help='set batch_size')
parser.add_argument("-seq_len", "--seq_len", required=True, type=int,  help='set how many directions. if     2, it\'s bidirection')
parser.add_argument("-binary", "--binary", required=True, type=int, help="select binary or multi classification. 0 == multi, 1 == binary")
parser.add_argument("-multi_to_binary", "--multi_to_binary", required=True, type=int, help="multi_to_binary = 1, binary_to_binary = 0")
parser.add_argument("-many_train", "--many_train", required=True, type=float, help="many to many train =1 , many to one train =0") 

parser.add_argument("-valid_ratio", "--valid_ratio", required=False, type=float, default=0.1, help="set how much ta    ke data from training_set as validation_set")
parser.add_argument("-hidden_size", "--hidden_size", required=False, type=int, default=100, help='set hidden_size o    f LSTM')
parser.add_argument("-num_layers", "--num_layers", required=False, type=int, default=1, help='set # of layers of LS    TM')
parser.add_argument("-lr", "--lr", required=False, type=float, default=0.001)
parser.add_argument("-max_patience", "--max_patience", required=False, type=int, default=5)
parser.add_argument("-max_epoch", "--max_epoch", required=False, type=int, default= 10000)
parser.add_argument("-try_entries", "--try_entries", required=False, type=int, default = 5)
parser.add_argument("-dataset", "--dataset", required=False, type=str, help="select dataset(nsl_kdd, unsw). kdd accepted as nsl_kdd", default='unsw')
parser.add_argument("-model", "--model", required=False, type=str, help="select model(mlp, lstm). rnn accepted as lstm", default='rnn')
parser.add_argument("-path", "--path", required=False, default='', type=str, help="if entered the result file will be recorded on input path. Caution!!! - wrong directory")
parser.add_argument("-optim", "--optimizer_", required=False, default=0, type=int, help="0	torch.optim.RMSprop	1: torch.optim.Adadelta	2: torch.optim.Adagran	3: torch.optim.Adam	4: torch.optim.SGD");
parser.add_argument("-dropout", "--dropout", required=False, default=0.5, type=float, help="input dropout, defalut value is 0.5. The input value should be ranged from 0.0 to 1.0") 


args = parser.parse_args()

args_optimizer = args.optimizer_
valid_ratio = args.valid_ratio
batch_size= args.batch_size
seq_len= args.seq_len
hidden_size= args.hidden_size
num_layers= args.num_layers

lr= args.lr
max_patience= args.max_patience
max_epoch = args.max_epoch
try_entries= args.try_entries
file_name= args.file_name
dataset_type = args.dataset
model_type = args.model
binary = args.binary
multi_to_binary = args.multi_to_binary
path = args.path
dropout = args.dropout
many_train = args.many_train

if(binary == 0 and multi_to_binary == 1):
	print("If multi classification, you can't choose multi_to_binary = 1")
	exit()

input_size = 0
output_size= 0

if dataset_type == 'unsw':
	input_size = 42
else:
	input_size = 41

if binary == 1 and multi_to_binary ==0:
	output_size = 2
elif binary == 0 or (binary ==1 and multi_to_binary == 1):
	output_size = 10

print("import processing is completed")
data_preprocessing_start=time.time()

####### init model
seed_model = 0

print('model, dataset: %s, %s\n'%(model_type, dataset_type))
if model_type == 'mlp':
	if dataset_type == 'kdd' or dataset_type =='nsl_kdd':
		seed_model = kdd_mlp
		print('mlp, kdd selected')
	elif dataset_type == 'unsw':
		seed_model = unsw_mlp
		print('mlp, unsw selected')
	else:
		print("dataset initializing failure (wrong input)")
elif model_type == 'lstm' or model_type == 'rnn':
	if dataset_type == 'kdd' or dataset_type == 'nsl_kdd':
		seed_model = kdd_rnn
		print('rnn, kdd selected')
	elif dataset_type == 'unsw':
		seed_model = unsw_rnn
		print('rnn, unsw selected')
	else:
		print("dataset initializing failure (wrong input)")
else:
	print("model initializing failure. (wrong input)")

dataset = load_dataset.gen_dataset_iter(dataset_type, binary, multi_to_binary, valid_ratio, batch_size,seq_len)

dataset = dataset.get()

train_iter = dataset[0]
valid_iter = dataset[1]
test_iter = dataset[2]

print("dataset fetched")

block_size = seq_len


#### finish datset init


def list_multi_to_binary(array):
	for i, item in enumerate(array):
		if item == 6:
			array[i] = 0
		else:
			array[i] = 1

def single_multi_to_binary(element):
	if(element == 6):
		return 0
	else:
		return 1

def train_mlp():
	model.train()
	for b_idx, (data, target) in enumerate(train_iter):
		data, target = data.cuda(), target.cuda()
		data, target = Variable(data), Variable(target)
		optimizer.zero_grad()
		output = model(data)
		loss = F.nll_loss(output, target)
		loss.backward()
		optimizer.step()



def train_rnn(many_train):
	model.train()
	print("training..")
	for b_idx, (data_cont, data_cat, target) in enumerate(train_iter):

		data_cont, data_cat, target = data_cont.cuda(), data_cat.cuda(), target.cuda()
		data_cont, data_cat, target = Variable(data_cont), Variable(data_cat), Variable(target)
		
		#data = data.view(-1, seq_len , input_size) # batch x seq x input

		output = model(data_cont, data_cat ,seq_len) # batch x seq x 10
		
		if(many_train == 1):
			output = output.view(-1,output_size) # batch x output

		elif(many_train == 0):
			output = output[:,-1,:] # batch x 1(last seq) x output
			target = target.view(-1,block_size,1) # batch x seq x 1
			target = target[:,-1,:] # batch x 1(last seq) x 1
			output = output.view(-1,output_size) # batch x output
			target = target.view(-1) # batch x 1			

		optimizer.zero_grad()
		loss_func = nn.CrossEntropyLoss()
		loss = loss_func(output, target)
		loss.backward()
		optimizer.step()

def valid_test_mlp(is_valid, binary): #### 0 == test, 1 == valid
	model.eval()
	correct = 0
	input_dataset = 0
	size = 0
	class_pred = [0.0 for i in range(2)]
	class_correct = [0.0 for i in range(2)]
	class_total = [0.0 for i in range(2)]


	multi_class_pred = [0.0 for i in range(10)]
	multi_class_correct = [0.0 for i in range(10)]
	multi_class_total = [0.0 for i in range(10)]	

	if is_valid:
		input_dataset = valid_iter
		size = len(valid_iter.dataset)
	else:
		input_dataset = test_iter
		size = len(test_iter.dataset)
	
	for b_idx, (data, target) in enumerate(input_dataset):
		data, target = data.cuda(), target.cuda()
		data, target = Variable(data), Variable(target)
		optimizer.zero_grad()
		output = model(data)
		_, pred = torch.max(output, 1)
		correct += (pred == target).sum()
		######## calc scores
		c = (pred == target).squeeze()
		for j in range(pred.size(0)):
			class_pred[pred[j]] += 1
			label = target[j]
			class_correct[label] += c[j].item()
			class_total[label] += 1

	pr = (class_correct[1]/class_pred[1])
	rc = (class_correct[1]/class_total[1])
	f1 = 2*(pr * rc)/(pr + rc)

	return float(correct)/size, f1

def valid_test_rnn(is_valid, binary,multi_to_binary): #### 0 == test, 1 == valid
	model.eval()
	
	correct = 0
	size = 0

	class_pred = [0.0 for i in range(2)]
	class_correct = [0.0 for i in range(2)]
	class_total = [0.0 for i in range(2)]

	multi_class_pred = [0.0 for i in range(10)]
	multi_class_correct = [0.0 for i in range(10)]
	multi_class_total = [0.0 for i in range(10)]	
	
	if is_valid:
		print("validation..")
		input_dataset = valid_iter
		size = len(valid_iter.dataset)/block_size
		print("validation numbers",size)

		for b_idx, (data_cont, data_cat, target) in enumerate(input_dataset):
			data_cont, data_cat, target = data_cont.cuda(), data_cat.cuda(), target.cuda()
			data_cont, data_cat, target = Variable(data_cont), Variable(data_cat), Variable(target)
		

			#data =data.view(-1, block_size , input_size) # batch x seq x input
			output = model(data_cont, data_cat ,block_size) # batch x seq x output_size
		
			if( len(data_cont) > 1 ):
				target = target.view(-1,block_size,1) # batch x seq x 1

				output = output[:,-1,:] # batch x 1(last seq) x output
				target = target[:,-1,:] # batch x 1(last seq) x 1

				output = output.view(-1,output_size) # batch x output
				target = target.view(-1) # batch x 1

				_, pred = torch.max(output, 1)
				

				if(binary==1 and multi_to_binary == 1):
					list_multi_to_binary(target)
					list_multi_to_binary(pred)

				correct += (pred == target).sum()
				if(binary==1):
					c = (pred == target).squeeze()
					for j in range(pred.size(0)):
						class_pred[pred[j]] += 1
						label = target[j]
						class_correct[label] += c[j].item()
						class_total[label] += 1
			
			elif( len(data_cont) ==1 ):
				output = output.view(-1,output_size) # batch x output
				output = output[-1]
				target = target[-1]
				
				_, pred = torch.max(output,0)
				
				if(binary==1 and multi_to_binary == 1):
					target  = single_multi_to_binary(target)
					pred = single_multi_to_binary(pred)

				correct += (pred == target).sum()
				if(binary==1):
					class_pred[pred] += 1
					label = target
					class_correct[label] += (pred==target).sum()
					class_total[label] +=1

		if(binary == 1):
			pr = (float(class_correct[1])/class_pred[1])
			rc = (float(class_correct[1])/class_total[1])
			f1 = 2*(pr * rc)/(pr + rc)		
		
			return float(correct)/size , f1
		elif(binary == 0):
			return float(correct)/size , 0
	else:
		print("testing..")
		input_dataset = test_iter
		size = (len(test_iter.dataset)/block_size) + block_size -1
		print("test data numbers",size)
		
		count = 0
		for b_idx, (data_cont, data_cat, target) in enumerate(input_dataset):
			data_cont, data_cat, target = data_cont.cuda(), data_cat.cuda(), target.cuda()
			data_cont, data_cat, target = Variable(data_cont), Variable(data_cat), Variable(target)
			


			#data =data.view(-1, block_size , input_size) # batch x seq x input
			target = target.view(-1,block_size,1) # batch x seq x 1
			output = model(data_cont, data_cat, block_size) # batch x seq x output_size
			
			data_length = len(target)

			if count ==0:
				first_output = output[0]
				first_target = target[0]
				first_output = first_output.view(-1,output_size)
				first_target = first_target.view(-1)
					
				##
				total_len = len(first_target)
				_, pred = torch.max(first_output, 1)
					
				if(binary==1 and multi_to_binary == 1):
					list_multi_to_binary(first_target)
					list_multi_to_binary(pred)
					
				correct += (pred == first_target).sum()
					
				if(binary == 1):
					c = (pred == first_target).squeeze()
					for j in range(pred.size(0)):
						class_pred[pred[j]] += 1
						label = first_target[j]
						class_correct[label] += c[j].item()
						class_total[label] += 1
				elif(binary == 0):
					c = (pred == first_target).squeeze()
					for j in range(pred.size(0)):
						multi_class_pred[pred[j]] += 1
						label = first_target[j]
						multi_class_correct[label] += c[j].item()
						multi_class_total[label] += 1
				
				data_length -= 1

				if(data_length>0):
					output = torch.cat([output[0:0], output[0+1:]])
					target = torch.cat([target[0:0], target[0+1:]])

			if( data_length > 1 ):
				
				output = output[:,-1,:] # batch x 1(last seq) x output
				target = target[:,-1,:] # batch x 1(last seq) x 1

				output = output.view(-1,output_size) # batch x output
				target = target.view(-1) # batch x 1
				
				##
				total_len += len(target)

				_, pred = torch.max(output, 1)
				
				if(binary==1 and multi_to_binary == 1):
					list_multi_to_binary(target)
					list_multi_to_binary(pred)

				correct += (pred == target).sum()
				if(binary==1):
					c = (pred == target).squeeze()
					for j in range(pred.size(0)):
						class_pred[pred[j]] += 1
						label = target[j]
						class_correct[label] += c[j].item()
						class_total[label] += 1
				elif(binary == 0):
					c = (pred == target).squeeze()
					for j in range(pred.size(0)):
						multi_class_pred[pred[j]] += 1
						label = target[j]
						multi_class_correct[label] += c[j].item()
						multi_class_total[label] += 1
			
			elif( data_length ==1 ):
				output = output.view(-1,output_size) # 1 x seq x output
				target = target.view(-1)
				output = output[-1]
				target = target[-1]
				
				##
				total_len += 1
				_, pred = torch.max(output,0)

				if(binary==1 and multi_to_binary ==1):
					target  = single_multi_to_binary(target)
					pred = single_multi_to_binary(pred)

		
				correct += (pred == target)
				if(binary==1):
					class_pred[pred] += 1
					label = target
					class_correct[label] += (pred==target)
					class_total[label] +=1
				elif(binary==0):
					multi_class_pred[pred] += 1
					label = target
					multi_class_correct[label] += (pred==target).sum()
					multi_class_total[label] +=1
		

			count +=1
			
		print(total_len)
		if(binary==1):
			pr = (float(class_correct[1])/class_pred[1])
			rc = (float(class_correct[1])/class_total[1])
			f1 = 2*(pr * rc)/(pr + rc)

			return float(correct)/size , f1

		elif(binary ==0):
			for i in range(10):
				print(multi_class_correct[i]/multi_class_total[i])
			return float(correct)/size, 0
		




		

def optimizer_align():
	if args_optimizer == 0:
		return torch.optim.RMSprop(model.parameters(), lr=lr, alpha=0.99, eps=1e-08, weight_decay=1e-05, momentum=0.9, centered=False)
	elif args_optimizer == 1:
		return torch.optim.Adadelta(model.parameters(), lr=lr, rho=0.9, eps=1e-06, weight_decay=1e-05)
	elif args_optimizer == 2:
		return torch.optim.Adagrad(model.parameters(), lr=lr, lr_decay=0, weight_decay=1e-05, initial_accumulator_value=0)
	elif args_optimizer == 3:
		return torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-05, amsgrad=False)
	elif args_optimizer == 4:
		return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, dampening=0, weight_decay=1e-05, nesterov=False)
	else:
		print("wrong optimizer input.")
		print("RMSprop optimizer is selected")
		return torch.optim.RMSprop(model.parameters(), lr=lr, alpha=0.99, eps=1e-08, weight_decay=1e-05, momentum=0.9, centered=False)


### align functions
train = 0
valid = 0
test = 0

if model_type == 'mlp':
	train = train_mlp
	valid = valid_test_mlp
	test = valid_test_mlp
else:
	train = train_rnn
	valid = valid_test_rnn
	test = valid_test_rnn

###


start_time = time.time()
avr_valid =0
avr_test =0
avr_i = 0
avr_f1 = 0

result_file = open(path+file_name+'.txt', 'w')
if model_type == 'mlp':
	result_file.write('model, dataset, lr, try_entries, max_epoch: %s, %s, %03f, %d, %d\n'%(model_type, dataset_type, lr, try_entries, max_epoch))
	result_file.write('batch_size, binary, max_patience: %d, %d, %d\n'%(batch_size, binary, max_patience))
	
else:
	result_file.write('model, dataset, lr, try_entries, max_epoch: %s, %s, %03f, %d, %d\n'%(model_type, dataset_type, lr, try_entries, max_epoch))
	result_file.write('batch_size, binary, multi_to_binary, many_train ,max_patience: %d, %d, %d, %d, %d\n'%(batch_size, binary, multi_to_binary, many_train, max_patience))
	result_file.write('input_size, hidden_size, num_layers, seq_len: %d, %d, %d, %d\n'%(input_size, hidden_size, num_layers ,seq_len))




data_preprocessing_finish = time.time()
optimizer =0
print('data preprocessing time: ' + str(int(data_preprocessing_finish - data_preprocessing_start)))
for i in range(try_entries):
	best_valid = 0
	best_valid_f1 = 0
	best_test = 0
	best_test_f1 = 0

	current_valid = 0
	current_valid_f1 =0
	current_test = 0
	current_test_f1 =0
	current_time = 0 
	
	current_patience = 0
	

	j=0
	
	model = 0
	if model_type == 'rnn':
		model = seed_model.model(input_size, hidden_size, num_layers, binary, multi_to_binary, dropout)
	else:
		model = seed_model.model(binary, dropout)
	model.cuda()
	optimizer = optimizer_align()
	
	

	for j in range(max_epoch):
		epoch_time = time.time()
		train(many_train)
		current_valid , current_valid_f1 = valid(1, binary,multi_to_binary)
				
		
		print('%d in %d try, training epoch: %d' %(i ,try_entries,j))
		print('current, best valid acc : %03f, %03f , current, best valid f1 : %03f, %03f'%(current_valid, best_valid, current_valid_f1, best_valid_f1))

		if current_valid_f1 > best_valid_f1:
			best_valid_f1 = current_valid_f1

		if current_valid > best_valid:
			best_valid = current_valid
			current_test, current_test_f1 = test(0, binary,multi_to_binary)
	
			current_patience = 0
			torch.save(model, file_name + ".model.pt")
			
			print('current, best test acc : %03f, %03f , current, best test f1 : %03f, %03f'%(current_test, best_test, current_test_f1, best_test_f1))
			
			if current_test_f1 > best_test_f1:
				best_test_f1 = current_test_f1
			if current_test >best_test:
				best_test = current_test
			
		else:
			current_patience += 1
			print('current patience: %d'%(current_patience))
			if current_patience >= max_patience:
				break

		current_time = time.time()
		print('elapsed time: ' + str(int(current_time - epoch_time)))
		print("---------------------------------------------------------------------------")
	result_file.write('best_valid, result_test_acc , result_test_f1 , break_i: %03f, %03f, %03f, %d\n'%(best_valid, current_test, current_test_f1, j))
	
	avr_valid += best_valid
	avr_test += current_test
	avr_i += j
	avr_f1 += current_test_f1

avr_valid /= float(try_entries)
avr_test /= float(try_entries)
avr_i = float(avr_i)/try_entries
avr_f1 /= float(try_entries)

result_file.write('average valid, test, f1, i: %03f, %03f, %03f,  %d\n'%(avr_valid, avr_test,avr_f1, avr_i))


stop_time = time.time()

result_file.write('used optimizer: ' + str(optimizer) + '\n')
result_file.write('data preprocessing time: ' + str(int(data_preprocessing_finish - data_preprocessing_start))+'\n')
result_file.write('elapsed time: ' + str(int(stop_time - start_time)) + '\n')
result_file.write('-----------end line------------\n')
result_file.write("\n")
result_file.flush()
result_file.close()

