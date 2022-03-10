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

class dbg_net(nn.Module):
	def __init__(self, in_ch=1, in_dim=28, width=2, linear_size=256):
		super(dbg_net, self).__init__()
		self.conv1_1 = nn.Conv2d(1, 4 * 2, 4, stride=2, padding=1)	
		self.conv1_2 = nn.Conv2d(1, 4 * 2, 4, stride=2, padding=1)
		self.conv1_3 = nn.Conv2d(1, 4 * 2, 4, stride=2, padding=1)

		self.conv2_1 = nn.Conv2d(4 * 2, 4 * 4, 4, stride=2, padding=1)	
		self.conv2_2 = nn.Conv2d(4 * 2, 4 * 4, 4, stride=2, padding=1)
		self.conv2_3 = nn.Conv2d(4 * 2, 4 * 4, 4, stride=2, padding=1)

		self.fc_1 = nn.Linear(8 * 2 * 7 * 7, 256)	
		self.fc_2 = nn.Linear(8 * 2 * 7 * 7, 256)
		self.fc_3 = nn.Linear(8 * 2 * 7 * 7, 256)

		
		self.fc = nn.Linear(3 * 256, 3) #8 * width * (in_dim // 4) * (in_dim // 4), 3)
	#	self.fc2 = nn.Linear(128, 3)
		#self.fc = nn.Linear(3 * 256, 64)
		#self.fc1 = nn.Linear(64, 3)
	def forward(self, x):
		#x = F.relu(self.fc(x))
		a, b, c = F.relu(self.conv1_1(x)), F.relu(self.conv1_2(x)), F.relu(self.conv1_3(x))
		a, b, c = F.relu(self.conv2_1(a)), F.relu(self.conv2_2(b)), F.relu(self.conv2_3(c))
		a, b, c = F.relu(self.fc_1(a.view(a.size(0), -1))), F.relu(self.fc_2(b.view(b.size(0), -1))), F.relu(self.fc_3(c.view(c.size(0), -1)))
		x = torch.cat([a, b, c], dim=1)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		return x


import pickle

def Train_gate(models, gate, t, loader, norm, train, opt, method='robust'):
	num_class = 10
	m = 3
	correct = 0

	meter = MultiAverageMeter()
	gate.train()
	for i in range(m): 
		models[i].eval()
	
	for i, (data, labels) in enumerate(loader):
		start = time.time()
		target = torch.zeros((data.size(0), 3))
		batch_method = method
		opt.zero_grad()
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

		for j in range(m): models[j](x)
		
		for j in range(m):
			lb, ub = models[j].compute_bounds(IBP=True, C=c, method=None)
			target[:, j] = torch.min(lb, dim=1)[0].cpu().detach().squeeze()

		flabel = torch.max(target, dim=1)[1]

		gate(x)

		c_g = torch.eye(3).type_as(data)[flabel].unsqueeze(1) - torch.eye(3).type_as(data).unsqueeze(0)
		I_g = (~(flabel.data.unsqueeze(1) == torch.arange(3).type_as(flabel).unsqueeze(0)))
		c_g = (c_g[I_g].view(data.size(0), 3 - 1, 3))
		
		glb, gub = gate.compute_bounds(IBP=True, C=c_g, method=None, bound_upper=False)
		
		target_g = (glb >= 0).all(dim=1, keepdim=True).cpu().detach().squeeze()
		lb_padded = torch.cat((torch.zeros(size=(glb.size(0),1), dtype=glb.dtype, device=glb.device), glb), dim=1)
		fake_labels = torch.zeros(size=(glb.size(0),), dtype=torch.int64, device=glb.device)
		loss = CrossEntropyLoss()(-lb_padded, fake_labels)
		loss.backward()
		opt.step()

		meter.update('Loss', loss.item(), data.size(0))

		meter.update('Certified_acc', torch.sum(target_g).item() / data.size(0) * 100., data.size(0))

		if i % 50 == 0 and train:
			print('[{:2d}:{:4d}]: eps={:.8f} {}'.format(t, i, eps, meter))


def Train_sub(models, gate, t, loader, norm, train, opt, method='robust'):
	num_class = 10
	m = 3
	correct = 0

	meter = MultiAverageMeter()
	gate.eval()
	for i in range(m): 
		models[i].train()
	
	for i, (data, labels) in enumerate(loader):
		start = time.time()
		target = torch.zeros((data.size(0), 3))

		gate_lbs = torch.zeros((data.size(0), 3))
		
		batch_method = method
		opt.zero_grad()
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

		for j in range(3):
			nlabel = torch.ones((data.size(0),)).type(torch.LongTensor) * j
			c_g = torch.eye(3).type_as(data)[nlabel].unsqueeze(1) - torch.eye(3).type_as(data).unsqueeze(0)
			I_g = (~(nlabel.data.unsqueeze(1) == torch.arange(3).type_as(nlabel).unsqueeze(0)))
			c_g = (c_g[I_g].view(data.size(0), 3 - 1, 3))
			gate(x)
			glb, gub = gate.compute_bounds(IBP=True, C=c_g, method=None, bound_upper=False)
			gate_lbs[:, j] = torch.min(glb, dim=1)[0].cpu().detach().squeeze()

		flabels = torch.max(gate_lbs, dim=1)[1].cuda()
	
		weights = [1 + 15 * (flabels == 0), 1 + 15 * (flabels == 1), 1 + 15 * (flabels == 2)]
		for j in range(m): models[j](x)
		
		loss = 0
		for j in range(m):
			lb, ub = models[j].compute_bounds(IBP=True, C=c, method=None)
			target[:, j] = (lb >= 0).all(dim=1, keepdim=True).cpu().detach().squeeze()
			lb_padded = torch.cat((torch.zeros(size=(lb.size(0),1), dtype=lb.dtype, device=lb.device), lb), dim=1)
			fake_labels = torch.zeros(size=(lb.size(0),), dtype=torch.int64, device=lb.device)
			loss += torch.mean(CrossEntropyLoss(reduce=False)(-lb_padded, fake_labels) * weights[j])
	
		loss.backward()
		opt.step()

		meter.update('Loss', loss.item(), data.size(0))
		for j in range(m):
			meter.update('Certified_acc_%d' % (j), torch.sum(target[:,j]).item() / data.size(0) * 100., data.size(0))
		if i % 50 == 0 and train:
			print('[{:2d}:{:4d}]: eps={:.8f} {}'.format(t, i, eps, meter))

	print(correct)

def Train(model, writer, t, loader, eps_scheduler, norm, train, opt, bound_type, method='robust'):
	num_class = 3
	meter = MultiAverageMeter()
	allc = 0
	if (train == False):
		model.train()
		eps_scheduler.eval()
	else:
		model.eval()
		eps_scheduler.train()
		eps_scheduler.step_epoch()
		eps_scheduler.set_epoch_length(int((len(loader.dataset) + loader.batch_size - 1) / loader.batch_size))

	dict = torch.zeros((8))
	flabel = [0, 0, 1, 1, 2, 1, 0, 2, 0, 2]
	for i, (data, labels) in enumerate(loader):
		
		nlabel = torch.zeros((data.size(0),)).type(torch.LongTensor)

		for j in range(data.size(0)):
			nlabel[j] = flabel[labels[j]]
	
		start = time.time()
		eps_scheduler.step_batch()
		eps = eps_scheduler.get_eps()
		# For small eps just use natural training, no need to compute LiRPA bounds
		batch_method = method
		if eps < 1e-20:
			batch_method = "natural"
		if train:
			opt.zero_grad()
		# generate specifications
		c = torch.eye(num_class).type_as(data)[nlabel].unsqueeze(1) - torch.eye(num_class).type_as(data).unsqueeze(0)
		# remove specifications to self
		I = (~(nlabel.data.unsqueeze(1) == torch.arange(num_class).type_as(nlabel.data).unsqueeze(0)))
		c = (c[I].view(data.size(0), num_class - 1, num_class))
		# bound input for Linf norm used only
		if norm == np.inf:
			data_max = torch.reshape((1. - loader.mean) / loader.std, (1, -1, 1, 1))
			data_min = torch.reshape((0. - loader.mean) / loader.std, (1, -1, 1, 1))
			data_ub = torch.min(data + (eps / loader.std).view(1,-1,1,1), data_max)
			data_lb = torch.max(data - (eps / loader.std).view(1,-1,1,1), data_min)
		else:
			data_ub = data_lb = data

		#if list(model.parameters())[0].is_cuda:
		data, nlabel, c = data.cuda(), nlabel.cuda(), c.cuda()
		data_lb, data_ub = data_lb.cuda(), data_ub.cuda()

		# Specify Lp norm perturbation.
		# When using Linf perturbation, we manually set element-wise bound x_L and x_U. eps is not used for Linf norm.
		if norm > 0:
			ptb = PerturbationLpNorm(norm=norm, eps=eps, x_L=data_lb, x_U=data_ub)
		elif norm == 0:
			ptb = PerturbationL0Norm(eps = eps_scheduler.get_max_eps(), ratio = eps_scheduler.get_eps()/eps_scheduler.get_max_eps())
		x = BoundedTensor(data, ptb)

		output = model(x)

		regular_ce = CrossEntropyLoss()(output, nlabel)  # regular CrossEntropyLoss used for warming up
		
		meter.update('CE', regular_ce.item(), x.size(0))
		meter.update('Err', torch.sum(torch.argmax(output, dim=1) != nlabel).cpu().detach().numpy() / x.size(0), x.size(0))

		loss = 0

		if batch_method == "robust":
			robust_ce = 0
			factor = (eps_scheduler.get_max_eps() - eps) / eps_scheduler.get_max_eps()
			ilb, iub = model.compute_bounds(IBP=True, C=c, method=None)
			clb, cub = model.compute_bounds(IBP=False, C=c, method="backward", bound_upper=False)
			if factor < 1e-5:
				lb = ilb
			else:
				lb = clb * factor + ilb * (1 - factor)

			lb_padded = torch.cat((torch.zeros(size=(lb.size(0),1), dtype=lb.dtype, device=lb.device), lb), dim=1)
			fake_labels = torch.zeros(size=(lb.size(0),), dtype=torch.int64, device=lb.device)
			robust_ce += nn.CrossEntropyLoss()(-lb_padded, fake_labels)

			loss = robust_ce# + grad_loss
		elif batch_method == "natural":
			loss = regular_ce
		if train:
			loss.backward()
			eps_scheduler.update_loss(loss.item() - regular_ce.item())
			opt.step()
		meter.update('Loss', loss.item(), data.size(0))
		if batch_method != "natural":
			meter.update('Certified_acc', torch.sum((lb >= 0).all(dim=1)).item() / data.size(0) * 100., data.size(0))
		meter.update('Time', time.time() - start)
		if i % 50 == 0 and train:
			print('[{:2d}:{:4d}]: eps={:.8f} {}'.format(t, i, eps, meter))


	print('[{:2d}:{:4d}]: eps={:.8f} {}'.format(t, i, eps, meter))
	#prefix += "%d/" % (m)

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

	models, core = [], []

	gate_core = dbg_net()
	st = gate_core.state_dict()


	for i in range(m):
		model = cnn_4layer(in_ch=1, in_dim=28)
		checkpoint = torch.load("./nreweight_cibp_0.3_15/model%d.pt" % (i))['state_dict']
		for j in range(1, 3):
			st["conv%d_%d.weight" % (j, i + 1)].copy_(checkpoint["conv%d.weight" % (j)])
			st["conv%d_%d.bias" % (j, i + 1)].copy_(checkpoint["conv%d.bias" % (j)])

		#st["fc_%d.weight" % (i + 1)].copy_(checkpoint["11.weight"])
		#st["fc_%d.bias" % (i + 1)].copy_(checkpoint["11.bias"])
		model.load_state_dict(checkpoint)
		model_pred = BoundedModule(model, torch.empty_like(dummy_input), device="cuda")
		model_pred.eval()	
		models.append(model_pred)
		core.append(model)

	param = []
	for name, para in gate_core.named_parameters():
		if (name.startswith("conv1") or name.startswith("conv2")):
			continue
		param.append(para)


	gate_pred = BoundedModule(gate_core, torch.empty_like(dummy_input), device="cuda")

	norm = np.inf

	opt = optim.Adam(param, lr=5e-4)
	lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
	eps_scheduler = eval("SmoothedScheduler")(0.3, "start=3,length=60")

	test_eps_scheduler = FixedScheduler(0.3)
	timer = 0.0
	epochs = 100

	writer = SummaryWriter("Ablation/joint")
	for t in range(1, epochs + 1):
		if eps_scheduler.reached_max_eps():
			lr_scheduler.step()
		print("Epoch {}, learning rate {}".format(t, lr_scheduler.get_lr()))
		start_time = time.time()
		Train(gate_pred, writer, t, train_data, eps_scheduler, norm, True, opt, "CROWN-IBP")
		epoch_time = time.time() - start_time
		timer += epoch_time
		print('Epoch time: {:.4f}, Total time: {:.4f}'.format(epoch_time, timer))
		print("Evaluating...")
		with torch.no_grad():
			Train(gate_pred, writer, t, test_data, eps_scheduler, norm, False, None, "CROWN-IBP")
	torch.save({'state_dict': gate_core.state_dict(), 'epoch': 0},
		"altsave/gatemodel.pt")


	for T in range(5):
		opt_g = optim.Adam(param, lr = 5e-4 / 8)
		gate_epochs = 5	
		opt_s = optim.Adam(param, lr = 5e-4 / 8)
		sub_epochs = 5

		print("=============== ROUND %d: Sub =====================" % (T + 1))
		for i in range(sub_epochs):
			Train_sub(models, gate_pred, i, train_data, norm, True, opt_s, "CROWN-IBP")

		st = gate_core.state_dict()
		for i in range(3):
			checkpoint = core[i].state_dict()
			for j in range(1, 3):
				st["conv%d_%d.weight" % (j, i + 1)].copy_(checkpoint["conv%d.weight" % (j)])
				st["conv%d_%d.bias" % (j, i + 1)].copy_(checkpoint["conv%d.bias" % (j)])

		print("=============== ROUND %d: Gate =====================" % (T + 1))
		for i in range(gate_epochs):
			Train_gate(models, gate_pred, i, train_data, norm, True, opt_g, "CROWN-IBP")

		torch.save({'state_dict': gate_core.state_dict(), 'epoch': T},
				"alt/gatemodel.pt")
		for i in range(3):
			torch.save({'state_dict': core[i].state_dict(), 'epoch': T},
					"alt/submodel%d.pt" % (i))

__main__()
