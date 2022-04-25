from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torchvision
from IPython.core.debugger import set_trace

__all__ = ['ResNet50TP', 'ResNet50TA', 'ResNet50RNN']


class ResNet50TP(nn.Module):
	def __init__(self, num_classes, loss={'cent'}, **kwargs):
		super(ResNet50TP, self).__init__()
		self.loss = loss
		resnet50 = torchvision.models.resnet50(pretrained=True)
		self.base = nn.Sequential(*list(resnet50.children())[:-2])
		self.feat_dim = 2048
		self.classifier = nn.Linear(self.feat_dim, num_classes)

	def forward(self, x):									# [4, 16, 3 , 224, 224]
		b = x.size(0)										# 4	
		t = x.size(1)										# 16
		x = x.view(b*t,x.size(2), x.size(3), x.size(4))		# [64, 3, 224, 224]
		x = self.base(x)									# [64, 2048, 7,7]
		x = F.avg_pool2d(x, x.size()[2:])					# [64, 2048, 1, 1]
		x = x.view(b,t,-1)									# [4, 16, 2048]
		x=x.permute(0,2,1)									# [4, 2048, 16]
		f = F.avg_pool1d(x,t)								# [4, 2048, 1]								
		f = f.view(b, self.feat_dim)						# [4, 2048]
		# if not self.training:
		# 	return f
		y = self.classifier(f)		
		return y, f

		# if self.loss == {'xent'}:
		# 	return y
		# elif self.loss == {'xent', 'htri'}:
		# 	return y, f
		# elif self.loss == {'cent'}:
		# 	return y, f
		# else:
		# 	raise KeyError("Unsupported loss: {}".format(self.loss))


class ResNet50TA(nn.Module):
	def __init__(self, num_classes, loss={'xent'}, **kwargs):
		super(ResNet50TA, self).__init__()
		self.loss = loss
		resnet50 = torchvision.models.resnet50(pretrained=True)
		self.base = nn.Sequential(*list(resnet50.children())[:-2])
		self.att_gen = 'softmax' # method for attention generation: softmax or sigmoid
		self.feat_dim = 2048 # feature dimension
		self.middle_dim = 256 # middle layer dimension
		self.classifier = nn.Linear(self.feat_dim, num_classes)
		self.attention_conv = nn.Conv2d(self.feat_dim, self.middle_dim, [7,7]) # 7,4 cooresponds to 224, 112 input image size
		self.attention_tconv = nn.Conv1d(self.middle_dim, 1, 3, padding=1)
	def forward(self, x): # x.shape [4, 16, 3, 224, 224]	
		#set_trace()	
		b = x.size(0)
		t = x.size(1)
		x = x.view(b*t, x.size(2), x.size(3), x.size(4))  	# x out [64, 3, 224, 224]
		x = self.base(x)									# x out [64, 2048, 7, 7]
		a = F.relu(self.attention_conv(x))					# a out [64, 256, 1, 1]
		a = a.view(b, t, self.middle_dim)					# a out [4,16, 256]
		a = a.permute(0,2,1)								# a out [4, 256, 16]
		a = F.relu(self.attention_tconv(a))					# a out [4, 1, 16]
		a = a.view(b, t)									# a out [4, 16]
		
		x = F.avg_pool2d(x, x.size()[2:])					# x out [160, 2048, 1, 1]			
		if self. att_gen=='softmax':
			a = F.softmax(a, dim=1)
		elif self.att_gen=='sigmoid':
			a = F.sigmoid(a)
			a = F.normalize(a, p=1, dim=1)
		else: 
			raise KeyError("Unsupported attention generation function: {}".format(self.att_gen))
		x = x.view(b, t, -1)								# x out [16, 10, 2048]
		a = torch.unsqueeze(a, -1)							# a out [16, 10, 1]
		_a=a
		a = a.expand(b, t, self.feat_dim)					# a out [16, 10, 2048]
		att_x = torch.mul(x,a)								# att_x [16, 10, 2048]
		att_x = torch.sum(att_x,1)							# att_x [16, 2048]		
		f = att_x.view(b,self.feat_dim)						# f out [16, 2048]
		# if not self.training:
		# 	return f
		y = self.classifier(f)	
		# return _a, f			
		return y, f
		
		


class ResNet50RNN(nn.Module):
	def __init__(self, num_classes, loss={'xent'}, **kwargs):
		super(ResNet50RNN, self).__init__()
		self.loss = loss
		resnet50 = torchvision.models.resnet50(pretrained=True)
		self.base = nn.Sequential(*list(resnet50.children())[:-2])
		self.hidden_dim = 512
		self.feat_dim = 2048
		self.classifier = nn.Linear(self.hidden_dim, num_classes)
		self.lstm = nn.LSTM(input_size=self.feat_dim, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)
	def forward(self, x): 										# [4, 16, 3, 224, 224] batch=4		
		b = x.size(0)											# 4
		t = x.size(1)											# 16
		x = x.view(b*t,x.size(2), x.size(3), x.size(4)) 		# [64, 3, 224, 224]
		x = self.base(x)										# [64, 2048, 7, 7]
		x = F.avg_pool2d(x, x.size()[2:])						# [64, 2048, 1, 1]							
		x = x.view(b,t,-1)										# [4, 16, 2048]
		output, (h_n, c_n) = self.lstm(x)						# output [4, 16, 512]
		output = output.permute(0, 2, 1)						# [4, 512, 16]	
		f = F.avg_pool1d(output, t)							# [4, 512, 1]	avg pooling
		# f=output[:,:,t-1]										# [4, 512, 1]	last node
		f = f.view(b, self.hidden_dim)							# [4, 512]
		# if not self.training:
		# 	return f
		y = self.classifier(f)
		return y, f
			
		# if self.loss == {'xent'}:
		# 	return y
		# elif self.loss == {'xent', 'htri'}:
		# 	return y, f
		# elif self.loss == {'cent'}:
		# 	return y, f
		# else:
		# 	raise KeyError("Unsupported loss: {}".format(self.loss))

