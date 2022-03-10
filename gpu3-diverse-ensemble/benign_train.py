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

def Train(model, ens, writer, t, loader, eps_scheduler, norm, train, opt, bound_type, method='robust'):
	num_class = 10
	meter = MultiAverageMeter()
	m = 3
	allc = 0
	if (train == False):
		for i in range(m): model[i].eval()
		eps_scheduler.eval()
		ens.eval()
	else:
		for i in range(m): model[i].train()
		ens.train()
		eps_scheduler.train()
		eps_scheduler.step_epoch()
		eps_scheduler.set_epoch_length(int((len(loader.dataset) + loader.batch_size - 1) / loader.batch_size))

	dict = torch.zeros((8))
	for i, (data, labels) in enumerate(loader):
		start = time.time()
		target = torch.zeros((data.size(0), 3))
		eps_scheduler.step_batch()
		eps = eps_scheduler.get_eps()
		# For small eps just use natural training, no need to compute LiRPA bounds
		batch_method = method
		if eps < 1e-20:
			batch_method = "natural"
		if train:
			opt.zero_grad()
		# generate specifications
		c = torch.eye(num_class).type_as(data)[labels].unsqueeze(1) - torch.eye(num_class).type_as(data).unsqueeze(0)
		# remove specifications to self
		I = (~(labels.data.unsqueeze(1) == torch.arange(num_class).type_as(labels.data).unsqueeze(0)))
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
		data, labels, c = data.cuda(), labels.cuda(), c.cuda()
		data_lb, data_ub = data_lb.cuda(), data_ub.cuda()

		# Specify Lp norm perturbation.
		# When using Linf perturbation, we manually set element-wise bound x_L and x_U. eps is not used for Linf norm.
		if norm > 0:
			ptb = PerturbationLpNorm(norm=norm, eps=eps, x_L=data_lb, x_U=data_ub)
		elif norm == 0:
			ptb = PerturbationL0Norm(eps = eps_scheduler.get_max_eps(), ratio = eps_scheduler.get_eps()/eps_scheduler.get_max_eps())
		x = BoundedTensor(data, ptb)

		for j in range(m): model[j](x)

		output = ens(x)
		regular_ce = CrossEntropyLoss()(output, labels)  # regular CrossEntropyLoss used for warming up
		
		meter.update('CE', regular_ce.item(), x.size(0))
		meter.update('Err', torch.sum(torch.argmax(output, dim=1) != labels).cpu().detach().numpy() / x.size(0), x.size(0))

		loss = 0

		if batch_method == "robust":
			robust_ce = 0
			for j in range(m):
				if bound_type == "IBP":
					lb, ub = model[j].compute_bounds(IBP=True, C=c, method=None)
				elif bound_type == "CROWN":
					lb, ub = model[j].compute_bounds(IBP=False, C=c, method="backward", bound_upper=False)
				elif bound_type == "CROWN-IBP":
					# lb, ub = model.compute_bounds(ptb=ptb, IBP=True, x=data, C=c, method="backward")  # pure IBP bound
					# we use a mixed IBP and CROWN-IBP bounds, leading to better performance (Zhang et al., ICLR 2020)
					factor = (eps_scheduler.get_max_eps() - eps) / eps_scheduler.get_max_eps()
					ilb, iub = model[j].compute_bounds(IBP=True, C=c, method=None)
					clb, cub = model[j].compute_bounds(IBP=False, C=c, method="backward", bound_upper=False)
					if factor < 1e-5:
						lb = ilb
					else:
						lb = clb * factor + ilb * (1 - factor)
				elif bound_type == "CROWN-FAST":
					# model.compute_bounds(IBP=True, C=c, method=None)
					lb, ub = model[j].compute_bounds(IBP=True, C=c, method=None)
					lb, ub = model[j].compute_bounds(IBP=False, C=c, method="backward", bound_upper=False)

				target[:, j] = (lb >= 0).all(dim=1, keepdim=True).cpu().detach().squeeze()
				# Pad zero at the beginning for each example, and use fake label "0" for all examples
			#	lb_padded = torch.cat((torch.zeros(size=(lb.size(0),1), dtype=lb.dtype, device=lb.device), lb), dim=1)
			#	fake_labels = torch.zeros(size=(lb.size(0),), dtype=torch.int64, device=lb.device)
			#	robust_ce += nn.CrossEntropyLoss()(-lb_padded, fake_labels)
			
			eilb, eub = ens.compute_bounds(IBP=True, C=c, method=None)
			elb, _ = ens.compute_bounds(IBP=False, C=c, method="backward", bound_upper=False)

			#if (factor < 1e-5): elb = eilb
			#else: elb = eclb * factor + eilb * (1 - factor)
			#if (train):
			#	if (factor < 1e-5): elb = eilb
			#	else: 
			#		elb = eclb * factor + eilb * (1 - factor)
			#else:
			#	elb = eclb

			lb_padded = torch.cat((torch.zeros(size=(elb.size(0),1), dtype=lb.dtype, device=lb.device), elb), dim=1)
			fake_labels = torch.zeros(size=(elb.size(0),), dtype=torch.int64, device=elb.device)
			robust_ce = nn.CrossEntropyLoss()(-lb_padded, fake_labels)
			loss = robust_ce# + grad_loss
		elif batch_method == "natural":
			loss = regular_ce
		if train:
			loss.backward()
			eps_scheduler.update_loss(loss.item() - regular_ce.item())
			opt.step()
		meter.update('Loss', loss.item(), data.size(0))
		if batch_method != "natural":
			meter.update('Robust_CE', robust_ce.item(), data.size(0))
			# For an example, if lower bounds of margins is >0 for all classes, the output is verifiably correct.
			# If any margin is < 0 this example is counted as an error
			
			meter.update('Certified_acc_ens', torch.sum((elb >= 0).all(dim=1)).item() / data.size(0) * 100., data.size(0))
			for j in range(m):
				meter.update('Certified_acc_%d' % (j), torch.sum(target[:,j]).item() / data.size(0) * 100., data.size(0))
		meter.update('Time', time.time() - start)
		if i % 50 == 0 and train:
			print('[{:2d}:{:4d}]: eps={:.8f} {}'.format(t, i, eps, meter))


	print('[{:2d}:{:4d}]: eps={:.8f} {}'.format(t, i, eps, meter))
	prefix = "train/"
	if (train == False): prefix = "test/"
	prefix += "%d/" % (m)
	writer.add_scalar(prefix + "Loss", meter.avg("Loss"), t)
	writer.add_scalar(prefix + "Certified_acc_ens", meter.avg("Certified_acc_ens"), t)
	for j in range(m):
		writer.add_scalar(prefix + "Certified_acc_%d" % (j), meter.avg("Certified_acc_%d" % (j)), t)
	writer.add_scalar(prefix + "Robust_CE", meter.avg("Robust_CE"), t)

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
	for i in range(m):
		model = cnn_4layer(in_ch=1, in_dim=28)
		#checkpoint = torch.load("/home/zly27/toolbox/ckpt/model%d.pt" % (2))['state_dict']
		#model.load_state_dict(checkpoint)
		model_pred = BoundedModule(model, torch.empty_like(dummy_input), device="cuda")
		#pred = model_gate(dummy_input)
		models.append(model_pred)
		core.append(model)
	ens = BoundedModule(EnsWrapper3(core[0],  core[1], core[2]), torch.empty_like(dummy_input), device="cuda")
	print("===========================")
	writer = SummaryWriter("Ablation/joint")

	param = list(core[0].parameters())
	for i in range(1, m):
		param.extend(list(core[i].parameters()))
	opt = optim.Adam(param, lr=5e-4)
	norm = float(np.inf)
	lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
	#eps_scheduler = SmoothedScheduler(0.3)


	eps_scheduler = eval("SmoothedScheduler")(0.3, "start=3,length=60")

	test_eps_scheduler = FixedScheduler(0.3)#, "start=3,length=60")
	timer = 0.0
	epochs = 100
	for t in range(1, epochs + 1):
		if eps_scheduler.reached_max_eps():
			# Only decay learning rate after reaching the maximum eps
			lr_scheduler.step()
		print("Epoch {}, learning rate {}".format(t, lr_scheduler.get_lr()))
		start_time = time.time()
		Train(models, ens, writer, t, train_data, eps_scheduler, norm, True, opt, "CROWN-IBP")
		epoch_time = time.time() - start_time
		timer += epoch_time
		print('Epoch time: {:.4f}, Total time: {:.4f}'.format(epoch_time, timer))
		print("Evaluating...")
		with torch.no_grad():
			Train(models, ens, writer, t, test_data, eps_scheduler, norm, False, None, "CROWN-IBP")
		for i in range(m):
			torch.save({'state_dict': core[i].state_dict(), 'epoch': t},
						"ens_c_0.3/model%d.pt" % (i))
__main__()
