import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch
import numpy as np
import multiprocessing
import torch.optim as optim
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
from auto_LiRPA.wrapper import *
from auto_LiRPA.utils import MultiAverageMeter
from auto_LiRPA.eps_scheduler import LinearScheduler, AdaptiveScheduler, SmoothedScheduler, FixedScheduler
import time
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.nn import CrossEntropyLoss
from tensorboardX import SummaryWriter
## Step 1: Define computational graph by implementing forward()
class Flatten(nn.Module):
	def forward(self, x):
		return x.view(x.size(0), -1)

# This simple model comes from https://github.com/locuslab/convex_adversarial

class cnn_4layer(nn.Module):
	def __init__(self, in_ch, in_dim, width=2, linear_size=256):
		super(cnn_4layer, self).__init__()
		self.conv1 = nn.Conv2d(in_ch, 4 * width, 4, stride=2, padding=1)
		self.conv2 = nn.Conv2d(4 * width, 8 * width, 4, stride=2, padding=1)
		self.fc1 = nn.Linear(8 * width * (in_dim // 4) * (in_dim // 4), linear_size)
		self.fc2 = nn.Linear(linear_size, 10)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = x.view(x.size(0), -1)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x

class cnn_4layer_g(nn.Module):
	def __init__(self, in_ch, in_dim, width=2, linear_size=256):
		super(cnn_4layer_g, self).__init__()
		self.conv1 = nn.Conv2d(in_ch, 4 * width, 4, stride=2, padding=1)
		self.conv2 = nn.Conv2d(4 * width, 8 * width, 4, stride=2, padding=1)
		self.fc1 = nn.Linear(8 * width * (in_dim // 4) * (in_dim // 4), linear_size)
		self.fc2 = nn.Linear(linear_size, 3)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = x.view(x.size(0), -1)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x
# Load the pretrained weights
## Step 2: Prepare dataset as usual


## Step 3: wrap model with auto_LiRPA
# The second parameter is for constructing the trace of the computational graph, and its content is not important.

# For larger convolutional models, setting bound_opts={"conv_mode": "patches"} is more efficient.
# model = BoundedModule(model, torch.empty_like(image), bound_opts={"conv_mode": "patches"})

## Step 4: Compute bounds using LiRPA given a perturbation
#label = torch.argmax(pred, dim=1).cpu().numpy()

## Step 5: Compute bounds for final output

import pickle
def Train(models, core, gate, loader, norm, train, bound_type, method='robust'):
	num_class = 10
	m = 3
	correct = 0
	v2 = 0
	hist = []
	dummy_input = torch.randn(1, 1, 28, 28)
	dummy_input = dummy_input.cuda()


	flabel = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2] 
	#flabel = [0, 0, 1, 1, 2, 1, 0, 2, 0, 2]
	cnt = torch.zeros(3)
	allt = torch.zeros(3)
	allc = 0
	verified = 0
	gg, hh = 0, 0
	aa, bb, cc, dd, ee, ff = 0, 0, 0, 0, 0, 0
	aa2, bb2, cc2, dd2, ee2, ff2 = 0, 0, 0, 0, 0, 0
	qc = 0
	for i, (data, labels) in enumerate(loader):
		start = time.time()
		target = torch.zeros((data.size(0), 3))

		
		batch_method = method

		# generate specifications
		c = torch.eye(num_class).type_as(data)[labels].unsqueeze(1) - torch.eye(num_class).type_as(data).unsqueeze(0)
		# remove specifications to self
		I = (~(labels.data.unsqueeze(1) == torch.arange(num_class).type_as(labels.data).unsqueeze(0)))
		c = (c[I].view(data.size(0), num_class - 1, num_class))
		# bound input for Linf norm used only



		eps = 0.3
		if norm == np.inf:
			data_max = torch.reshape((1. - loader.mean) / loader.std, (1, -1, 1, 1))
			data_min = torch.reshape((0. - loader.mean) / loader.std, (1, -1, 1, 1))
			data_ub = torch.min(data + (eps / loader.std).view(1,-1,1,1), data_max)
			data_lb = torch.max(data - (eps / loader.std).view(1,-1,1,1), data_min)
		else:
			data_ub = data_lb = data

		#if list(model.parameters())[0].is_cuda:
		data, labels, c = data.cuda(), labels.cuda(), c.cuda()
		data_lb, data_ub = data_lb.cuda(), data_ub.cuda()

		# Specify Lp norm perturbation.
		# When using Linf perturbation, we manually set element-wise bound x_L and x_U. eps is not used for Linf norm.
		if norm > 0:
			ptb = PerturbationLpNorm(norm=norm, eps=eps, x_L=data_lb, x_U=data_ub)
		elif norm == 0:
			ptb = PerturbationL0Norm(eps = eps_scheduler.get_max_eps(), ratio = eps_scheduler.get_eps()/eps_scheduler.get_max_eps())
		x = BoundedTensor(data, ptb)

		pred = torch.zeros(data.size(0)).type(torch.LongTensor)

		output = gate(x)
		
		
		output = nn.Softmax()(output).detach().cpu()
		for k in range(data.size(0)):
			we = BoundedModule(WEnsWrapper3(core[0], core[1], core[2], output[0]), torch.empty_like(dummy_input), device="cuda")
			we(x)
			ilb, iub = we.compute_bounds(IBP=True, C=c, method=None)
			qc += ((ilb >= 0).all()).item()
			#certified = ((ilb >= 0).all(dim=1, keepdim=True).cpu().detach().squeeze())
		print(qc / (i+1))
		
		continue	
		gate_lbs, ind_lbs = [], []
		for j in range(3):
			nlabel = torch.ones((data.size(0),)).type(torch.LongTensor) * j
			c_g = torch.eye(3).type_as(data)[nlabel].unsqueeze(1) - torch.eye(3).type_as(data).unsqueeze(0)
			I_g = (~(nlabel.data.unsqueeze(1) == torch.arange(3).type_as(nlabel).unsqueeze(0)))
			c_g = (c_g[I_g].view(data.size(0), 3 - 1, 3))
			gate(x)
			glb, gub = gate.compute_bounds(IBP=True, C=c_g, method=None, bound_lower=False)
			gate_lbs.append((gub >= 0).all(dim=1, keepdim=True).cpu().detach().squeeze())

		for j in range(m): models[j](x)
		
		for j in range(m):
			ilb, iub = models[j].compute_bounds(IBP=True, C=c, method=None)
			lb, cub = models[j].compute_bounds(IBP=False, C=c, method="backward", bound_upper=False)
			certified = ((ilb >= 0).all(dim=1, keepdim=True).cpu().detach().squeeze())
			ind_lbs.append(certified)


		for k in range(data.size(0)):
			chosen = 0
			for j in range(3):
				chosen += (gate_lbs[j])[k] # model j can be chosen by gate in instance k or not.
				if (gate_lbs[j])[k] == 1: idx = j
			#if (chosen != 1): continue
			#if (idx == 2): continue
			#print(chosen)
			ok = True
			for j in range(3):
				if ((gate_lbs[j])[k] == False): continue
				if ((ind_lbs[j])[k] == False): ok = False
	
			ok2 = False
			for j in range(3):
				if ((ind_lbs[j])[k] == True): ok2 = True

			correct += ok
			allc += 1
			if (flabel[labels[k]] == idx): 
				verified += 1
				v2 += ok
			
			if (chosen == 1): 
				if (flabel[labels[k]] == idx): # correctly certified from gate
					if (ok == True): aa += 1
					else: bb += 1

					if (ok2 == True): aa2 += 1
					else: bb2 += 1
					
				else:

					if ((ind_lbs[fl])[k]): gg += 1
					else: hh += 1
		
					if (ok == True): cc += 1
					else: dd += 1

					if (ok2 == True): cc2 += 1
					else: dd2 += 1
			else:
				fl = flabel[labels[k]]
				
				if ((ind_lbs[fl])[k]): ee += 1
				else: ff += 1

				if (ok2 == True): ee2 += 1
				else: ff2 += 1

				
				
		#for j in range(1, 2**m):
		#	selects = [j % 2, (j // 2) % 2, (j // 4)]
			#if (sum(selects) != 3): continue
			
			#idx = -1
		#	if (sum(selects) == 1):
		#		for k in range(m): 
		#			if (selects[k] == 1): 
		#				ens = models[k]
		#				idx = k
		#	else: continue
			#elif (sum(selects) == 2):
			#	if (selects[0] == 0): ens = BoundedModule(EnsWrapper(core[1], core[2]), torch.empty_like(dummy_input), device="cuda")
			#	elif (selects[1] == 0): ens = BoundedModule(EnsWrapper(core[0], core[2]), torch.empty_like(dummy_input), device="cuda")
			#	else: tmp = BoundedModule(EnsWrapper(core[0], core[1]), torch.empty_like(dummy_input), device="cuda")
			#else: ens = BoundedModule(EnsWrapper3(core[0], core[1], core[2]), torch.empty_like(dummy_input), device="cuda")

		#	ens(x)
		#	ilb, iub = ens.compute_bounds(IBP=True, C=c, method=None)
		#	lb, cub = ens.compute_bounds(IBP=False, C=c, method="backward", bound_upper=False)

		#	certified = ((lb >= 0).all(dim=1, keepdim=True).cpu().detach().squeeze())
		#	pred = pred | certified 
			

				

		#	for k in range(data.size(0)):
		#		fake_label = 0
		#		if (labels[k] < 3): fake_label = 0
		#		elif (labels[k] >= 6): fake_label = 2
		#		else: fake_label = 1
			
				#fake_label = (fake_label + 1) % 3
		#		if (certified[k] == True and idx == fake_label): cnt[idx] += 1
		#		if (idx == fake_label): allt[idx] += 1

		#correct += torch.sum(pred)
		#hist.extend(list(torch.min(lb, dim=1)[0].cpu().detach().squeeze().numpy()))
	print(correct)
	print(correct / allc)
	print(allc)
	print(verified)
	print(v2 / verified)
	print(aa, bb, cc, dd, ee, ff)
	print(aa2, bb2, cc2, dd2, ee2, ff2)
	print(gg, hh)
	#print(len(hist))
	#with open("hist_margin2.pkl", "wb+") as tf:
	#	pickle.dump(hist, tf)
	#print(correct.item())
	#print(cnt)
	#print(cnt[0] + cnt[1] + cnt[2])
	#print(allt)
	#print(cnt / allt)

def __main__():
	m = 3

	dummy_input = torch.randn(1, 1, 28, 28)
	dummy_input = dummy_input.cuda()
	train_data = datasets.MNIST("../toolbox/data", train=True, download=True, transform=transforms.ToTensor())
	test_data = datasets.MNIST("../toolbox/data", train=False, download=True, transform=transforms.ToTensor())
	train_data = torch.utils.data.DataLoader(train_data, batch_size=256, shuffle=True, pin_memory=True,
											 num_workers=min(multiprocessing.cpu_count(), 4))
	test_data = torch.utils.data.DataLoader(test_data, batch_size=1, pin_memory=True,
											num_workers=min(multiprocessing.cpu_count(), 4))
	train_data.mean = test_data.mean = torch.tensor([0.0])
	train_data.std = test_data.std = torch.tensor([1.0])

	models, core = [], []
	for i in range(m):
		model = cnn_4layer(in_ch=1, in_dim=28)
		checkpoint = torch.load("./reweight_cibp_0.3_15/model%d.pt" % (i))['state_dict']
		model.load_state_dict(checkpoint)
		model_pred = BoundedModule(model, torch.empty_like(dummy_input), device="cuda")
		model_pred.eval()	
		models.append(model_pred)
		core.append(model)

	gate = cnn_4layer_g(in_ch=1, in_dim=28)
	checkpoint = torch.load("./gate_0.3/model0.pt")['state_dict']
	gate.load_state_dict(checkpoint)
	gate_pred = BoundedModule(gate, torch.empty_like(dummy_input), device="cuda")
	gate_pred.eval()	

	
	norm = np.inf
	with torch.no_grad():
		Train(models, core, gate_pred, test_data, norm, False, None, "CROWN-IBP")

__main__()
