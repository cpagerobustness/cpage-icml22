import torch
import torch.nn as nn
import torch.nn.functional as F
class CrossEntropyWrapper(nn.Module):
	def __init__(self, model):
		super(CrossEntropyWrapper, self).__init__()
		self.model = model

	def forward(self, x, labels):
		y = self.model(x)
		logits = y - torch.gather(y, dim=-1, index=labels.unsqueeze(-1))
		return torch.exp(logits).sum(dim=-1, keepdim=True)

class CrossEntropyWrapperMultiInput(nn.Module):
	def __init__(self, model):
		super(CrossEntropyWrapperMultiInput, self).__init__()
		self.model = model

	def forward(self, labels, *x):
		y = self.model(*x)
		logits = y - torch.gather(y, dim=-1, index=labels.unsqueeze(-1))
		return torch.exp(logits).sum(dim=-1, keepdim=True)

class LAdhocWrapper(nn.Module):
	def __init__(self, model):
		super(LAdhocWrapper, self).__init__()
		self.model = model

	def forward(self, x):
		y = F.softmax(self.model(x))
		y = y - torch.sum(y, dim=1, keepdim=True) / 10.
		y = torch.sum(y * y, dim=1, keepdim=True) / 10.
		return y

class UAdhocWrapper(nn.Module):
	def __init__(self, model):
		super(UAdhocWrapper, self).__init__()
		self.model = model

	def forward(self, x):
		y = F.softmax(self.model(x))
		a = torch.sum(y, dim=1, keepdim=True) / 10.
		a = a * a
		y = y * y
		y = torch.sum(y, dim=1, keepdim=True) / 100.
		return a - y

class VariancesigWrapper(nn.Module):
	def __init__(self, model):
		super(VariancesigWrapper, self).__init__()
		self.model = model

	def forward(self, x):
		y = F.sigmoid(self.model(x))
		y = y - torch.sum(y, dim=1, keepdim=True) / 10.
		y = torch.sum(y * y, dim=1, keepdim=True) / 10.
		return y

class VarianceNoneWrapper(nn.Module):
	def __init__(self, model):
		super(VarianceNoneWrapper, self).__init__()
		self.model = model

	def forward(self, x):
		y = (self.model(x))
		y = y - torch.sum(y, dim=1, keepdim=True) / 10.
		y = torch.sum(y * y, dim=1, keepdim=True) / 10.
		return y

class VarianceWrapper(nn.Module):
	def __init__(self, model):
		super(VarianceWrapper, self).__init__()
		self.model = model

	def forward(self, x):
		y = F.softmax(self.model(x))
		y = y - torch.sum(y, dim=1, keepdim=True) / 10.
		y = torch.sum(y * y, dim=1, keepdim=True) / 10.
		return y

class GateWrapper(nn.Module):
	def __init__(self, model):
		super(GateWrapper, self).__init__()
		self.model = model
		self.conv1 = nn.Conv2d(24, 4, 4, stride=2, padding=1)
		self.fc1 = nn.Linear(4 * 7 * 7, 3)
	def forward(self, x):
		x = torch.cat([self.model[i].feature(x) for i in range(3)], dim=1)
		x = F.relu(self.conv1(x))
		x = x.view(x.size(0), -1)
		x = F.sigmoid(self.fc1(x))
		return x

class RWrapper(nn.Module):
	def __init__(self, model):
		super(RWrapper, self).__init__()
		self.model = model

	def forward(self, x):
		return - self.model(x)

class EnsWrapper(nn.Module):
	def __init__(self, model1, model2):
		super(EnsWrapper, self).__init__()
		self.model1 = model1 # = []
		self.model2 = model2 # = []
		#for i in range(len(models)):
		#	self.models.append(models[i])
	def forward(self, x):
		y = (self.model1(x) + self.model2(x)) / 2.
		#for i in range(1, len(self.models)):
		#	y += self.models[i](x)
		return y

class WEnsWrapper3(nn.Module):
	def __init__(self, model1, model2, model3, w):
		super(WEnsWrapper3, self).__init__()
		self.model1 = model1 # = []
		self.model2 = model2 # = []
		self.model3 = model3
		self.w = w
		#for i in range(len(models)):
		#	self.models.append(models[i])
	def forward(self, x):
		y = self.w[0] * self.model1(x) + self.w[1] * self.model2(x) + self.w[2] * self.model3(x)
		#for i in range(1, len(self.models)):
		#	y += self.models[i](x)
		return y

class EnsWrapper3(nn.Module):
	def __init__(self, model1, model2, model3):
		super(EnsWrapper3, self).__init__()
		self.model1 = model1 # = []
		self.model2 = model2 # = []
		self.model3 = model3
		#for i in range(len(models)):
		#	self.models.append(models[i])
	def forward(self, x):
		y = (self.model1(x) + self.model2(x) + self.model3(x))/3.
		#for i in range(1, len(self.models)):
		#	y += self.models[i](x)
		return y

class EntropyWrapper(nn.Module):
	def __init__(self, model):
		super(EntropyWrapper, self).__init__()
		self.model = model

	def forward(self, x):
		y = F.softmax(self.model(x))
		y = - y * torch.log(y)
		return torch.sum(y, dim=1, keepdim=True)
