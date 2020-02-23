import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchsummary import summary
import argparse
import numpy as np 
import matplotlib.pyplot as plt
from math import sqrt, floor
# argument parser
parser = argparse.ArgumentParser(description= 'ML_CODESIGN Lab1 - MNIST example' )
parser.add_argument( '--batch-size' , type=int, default= 100 , help= 'Number of samples per mini-batch' )
parser.add_argument( '--epochs' , type=int, default= 10 , help= 'Number of epoch to train' )
parser.add_argument( '--lr' , type=float, default= 0.1 , help= 'Learning rate' )
parser.add_argument( '--kernel_size', type=int, default = 3, help = 'kernel size')
parser.add_argument( '--enable_cuda', type = int, default = 1, help = '1 to enable cuda')
parser.add_argument( '--thr', type = float, default = 0.01, help = 'threshold for pruning')
parser.add_argument( '--frac', type = float, default = 10, help = 'fraction for pruning')
parser.add_argument( '--enable_thr_pruning', type = int, default = 0, help = 'enable frac pruning by default')
args = parser.parse_args()

# Hyper Parameters
input_size = 784
num_classes = 10
num_epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.lr
kernel_size = args.kernel_size
thr = args.thr
frac = args.frac
print(args.enable_thr_pruning)
args.device = None
print('learning rate: %f' %learning_rate)
if args.enable_cuda == 1  and torch.cuda.is_available():
	args.device = torch.device('cuda')
else:
	args.device = torch.device('cpu')

print('enable cuda is', args.enable_cuda, 'device is', args.device)

# MNIST Dataset (Images and Labels)
test_dataset = dsets.MNIST(root = './data' ,
	train = False ,
	transform = transforms.ToTensor())

# Dataset Loader (Input Pipeline)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
	batch_size = batch_size,
	shuffle = False )

#Model
class SimpleNet (nn.Module):
	def __init__ (self, args):
		super(SimpleNet, self).__init__()
		self.features1 = nn.Sequential()
		self.features1.add_module( "conv1" , nn.Conv2d( 1 , 4 , args.kernel_size, stride= 1 , padding= 1 ))
		self.features1.add_module( "bn1", nn.BatchNorm2d(num_features = 4))
		self.features1.add_module( "relu1", nn.ReLU())
			
		self.features1.add_module( "conv2" , nn.Conv2d( 4 , 16 , args.kernel_size, stride= 1 , padding= 1 ))
		self.features1.add_module( "bn2", nn.BatchNorm2d(num_features = 16))
		self.features1.add_module( "relu2", nn.ReLU())
		
		self.features1.add_module( "conv3" , nn.Conv2d( 16 , 16 , args.kernel_size, stride= 1 , padding= 1 ))
		self.features1.add_module( "bn3", nn.BatchNorm2d(num_features = 16))
		self.features1.add_module( "relu3", nn.ReLU())
		#input(1,28,28), output(16,28,28) need to add (1,1) conv layer to convert
		self.shortcut1 =  nn.Sequential(nn.Conv2d(1, 16, kernel_size = 1, stride = 1, padding = 0),
								nn.BatchNorm2d(num_features = 16))	

		self.pool1 = nn.MaxPool2d(kernel_size= 2 , stride= 2 )
		
		self.features2 = nn.Sequential()
		self.features2.add_module( "conv4" , nn.Conv2d( 16 , 16 , args.kernel_size, stride= 1 , padding= 1 ))
		self.features2.add_module( "bn4", nn.BatchNorm2d(num_features = 16));
		self.features2.add_module( "relu4", nn.ReLU())
		
		self.features2.add_module( "conv5" , nn.Conv2d( 16 , 16 , args.kernel_size, stride= 1 , padding= 1 ))
		self.features2.add_module( "bn5", nn.BatchNorm2d(num_features = 16));
		self.features2.add_module( "relu5", nn.ReLU())
		
		self.features2.add_module( "conv6" , nn.Conv2d( 16 , 16 , args.kernel_size, stride= 1 , padding= 1 ))
		self.features2.add_module( "bn6", nn.BatchNorm2d(num_features = 16));
		self.features2.add_module( "relu6", nn.ReLU())
		
		self.shortcut2 = nn.Identity()#no input/output dimension change, just bypass
		
		self.pool2 = nn.MaxPool2d(kernel_size= 2 , stride= 2 )
	
		self.features3 = nn.Sequential()	
		self.features3.add_module( "conv7" , nn.Conv2d( 16 , 16 , args.kernel_size, stride= 1 , padding= 1 ))
		self.features3.add_module( "bn7", nn.BatchNorm2d(num_features = 16));
		self.features3.add_module( "relu7", nn.ReLU())
	
		self.lin8 = nn.Linear( 7 * 7 * 16 , 4096 )
		self.relu8 = nn.ReLU()

		self.lin9 = nn.Linear( 4096 , 4096 )
		self.relu9 = nn.ReLU()

		self.lin10 = nn.Linear( 4096 , 4096 )
		self.relu10 = nn.ReLU()

		self.lin11 = nn.Linear( 4096 , 4096 )
		self.relu11 = nn.ReLU()

		self.lin12 = nn.Linear( 4096 , 4096 )
		self.relu12 = nn.ReLU()
		#self.features.add_module( "pool2" , nn.MaxPool2d(kernel_size= 2 , stride= 2 ))
		self.lin = nn.Linear( 4096 , 10 )#input_dimension, output_dimension
		
	def forward (self, x):
		res1 = x;
		out = self.features1(x)
		out = out + self.shortcut1(res1)
		out = self.pool1(out)
		
		res2 = out
		out = self.features2(out)
		out = out + self.shortcut2(res2)
		out = self.pool2(out)

		out = self.features3(out)
		out = out.view(out.size( 0 ), -1 )
		out = self.lin8(out)
		out =self.relu8(out)
		out = self.lin9(out)
		out =self.relu9(out)
		out = self.lin10(out)
		out =self.relu10(out)
		out = self.lin11(out)
		out =self.relu11(out)
		out = self.lin12(out)
		out =self.relu12(out)
		out = self.lin(out)
		return out

#def plot_histogram(data, fignum, scale):
#	num_of_element = len(data)
#	range_of_element = max(data) - min(data)
#	num_of_interval = floor(sqrt(num_of_element))
#	width_of_intervals = range_of_element/num_of_interval

#	plt.figure(fignum)
#	plt.yscale(scale)
#	plt.hist(data, bins = num_of_interval)

#def con2data(input_t):
#	data = input_t.data
#	data = torch.reshape(data,(-1,))
#	data = data.tolist()
#	output_list = []
#	for x in data:
#		output_list.append(x)
#	return output_list


def elw_threshold_pruning(model, thr = 0.5):
	print('thr')
	print(thr)
	for layer in model.modules():
		if isinstance(layer,torch.nn.Conv2d):
			print('Target conv layer')
			for weights in layer.weight:
				weights.masked_fill_(torch.le(torch.abs(weights), thr),0)
		else:
			print('Not a conv layer')

def elw_fraction_pruning(model, frac = 100):
	for layer in model.modules():
		if isinstance(layer, torch.nn.Conv2d):
			print('Target conv layer')
			thr = np.percentile(layer.weight.cpu().detach().numpy(), frac)
			for weights in layer.weight:
				weights.masked_fill_(torch.le(weights, thr), 0)
		else:
			print('Not a conv layer')

model = SimpleNet(args)
model.load_state_dict(torch.load('fineTuneModel.pt'))
#model.load_state_dict(torch.load('pruning.pt', map_location = args.device))
model.to(args.device)
model.eval()
summary(model, (1,28,28))
#fig_idx = 1
#plot weights for conv layer
#for layer in model.modules():
#	if isinstance(layer, torch.nn.Conv2d):	
#		print('inside' + layer.__class__.__name__ )
#		plot_histogram(con2data(layer.weight), fig_idx,'log') 
#		fig_idx+=1

if args.enable_thr_pruning == 0:
	print('frac pruning')
	print(frac)
	elw_fraction_pruning(model, frac)
else:
	print('thr pruning')
	print(thr)
	elw_threshold_pruning(model, thr)

# Test the Model
correct = 0
total = 0
for images, labels in test_loader:
	images = Variable(images).to(args.device)
	labels = Variable(labels).to(args.device)
	outputs = model(images)
	_, predicted = torch.max(outputs.data, 1 )
	total += labels.size( 0 )
	correct += (predicted.to(args.device) == labels.to(args.device)).sum()
print('total :%d, correct : %d' %(total, correct))
print( 'Accuracy of the model on the 10000 test images: % f %%' % ( float(100) * correct /total))

plt.show()
