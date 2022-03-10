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
def Train(model, loader, norm, train, bound_type, method='robust'):
	num_class = 10
	m = 3
	correct = 0
	hist = []
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
		model(x)

		ilb, iub = model.compute_bounds(IBP=True, C=c, method=None)
		lb, cub = model.compute_bounds(IBP=False, C=c, method="backward", bound_upper=False)
		correct += torch.sum((lb >= 0).all(dim=1, keepdim=True).cpu().detach().squeeze())
		#hist.extend(list(torch.min(lb, dim=1)[0].cpu().detach().squeeze().numpy()))
		
	#print(len(hist))
	#with open("hist_margin2.pkl", "wb+") as tf:
	#	pickle.dump(hist, tf)
	print(correct.item())

def __main__():
	m = 3

	dummy_input = torch.randn(1, 1, 28, 28)
	dummy_input = dummy_input.cuda()
	train_data = datasets.MNIST("../toolbox/data", train=True, download=True, transform=transforms.ToTensor())
	test_data = datasets.MNIST("../toolbox/data", train=False, download=True, transform=transforms.ToTensor())
	train_data = torch.utils.data.DataLoader(train_data, batch_size=256, shuffle=True, pin_memory=True,
											 num_workers=min(multiprocessing.cpu_count(), 4))
	test_data = torch.utils.data.DataLoader(test_data, batch_size=256, pin_memory=True,
											num_workers=min(multiprocessing.cpu_count(), 4))
	train_data.mean = test_data.mean = torch.tensor([0.0])
	train_data.std = test_data.std = torch.tensor([1.0])



	core = []
	for i in range(m):
		model = cnn_4layer(in_ch=1, in_dim=28)
		checkpoint = torch.load("./reweight_cibp_0.3_15/model%d.pt" % (i))['state_dict']
		model.load_state_dict(checkpoint)
		core.append(model)
	
	ens = BoundedModule(EnsWrapper3(core[0], core[1], core[2]), torch.empty_like(dummy_input), device="cuda")
	#model_pred = BoundedModule(model, torch.empty_like(dummy_input), device="cuda")
	ens.eval()
	#model_pred.eval()	
	norm = np.inf
	with torch.no_grad():
		Train(ens, test_data, norm, False, None, "CROWN-IBP")

__main__()
