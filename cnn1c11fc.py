import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchsummary import summary
import argparse

# argument parser
parser = argparse.ArgumentParser(description= 'ML_CODESIGN Lab1 - MNIST example' )
parser.add_argument( '--batch-size' , type=int, default= 100 , help= 'Number of samples per mini-batch' )
parser.add_argument( '--epochs' , type=int, default= 10 , help= 'Number of epoch to train' )
parser.add_argument( '--lr' , type=float, default= 0.1 , help= 'Learning rate' )
parser.add_argument( '--kernel_size', type=int, default = 3, help = 'kernel size')
parser.add_argument( '--enable_cuda', type = bool, default = 1, help = '1 to enable cuda')
args = parser.parse_args()

# Hyper Parameters
input_size = 784
num_classes = 10
num_epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.lr
kernel_size = args.kernel_size
args.device = None
print('learning rate: %f' %learning_rate)
if args.enable_cuda and torch.cuda.is_available():
	args.device = torch.device('cuda')
else:
	args.device = torch.device('cpu')

print('enable cuda is', args.enable_cuda, 'device is', args.device)
# MNIST Dataset (Images and Labels)
train_dataset = dsets.MNIST(root = './data' ,
	train = True ,
	transform = transforms.ToTensor(),
	download = True )

test_dataset = dsets.MNIST(root = './data' ,
	train = False ,
	transform = transforms.ToTensor())

# Dataset Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
	batch_size = batch_size,
	shuffle = True )

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
	batch_size = batch_size,
	shuffle = False )

#Model
class SimpleNet (nn.Module):
	def __init__ (self, args):
		super(SimpleNet, self).__init__()
		self.features = nn.Sequential()
		self.features.add_module( "conv1" , nn.Conv2d( 1 , 4 , args.kernel_size, stride= 1 , padding= 1 ))
		self.features.add_module( "bn1", nn.BatchNorm2d(num_features = 4))
		self.features.add_module( "relu1", nn.ReLU())
		
		self.features.add_module( "pool1" , nn.MaxPool2d(kernel_size= 2 , stride= 2 ))
		
		#self.features.add_module( "conv2" , nn.Conv2d( 4 , 16 , args.kernel_size, stride= 1 , padding= 1 ))
		#self.features.add_module( "bn2", nn.BatchNorm2d(num_features = 16))
		#self.features.add_module( "relu2", nn.ReLU())
		
		#self.features.add_module( "conv3" , nn.Conv2d( 16 , 16 , args.kernel_size, stride= 1 , padding= 1 ))
		#self.features.add_module( "bn3", nn.BatchNorm2d(num_features = 16))
		#self.features.add_module( "relu3", nn.ReLU())
		
		#self.features.add_module( "conv4" , nn.Conv2d( 16 , 16 , args.kernel_size, stride= 1 , padding= 1 ))
		#self.features.add_module( "bn4", nn.BatchNorm2d(num_features = 16));
		#self.features.add_module( "relu4", nn.ReLU())
		
		#self.features.add_module( "conv5" , nn.Conv2d( 16 , 16 , args.kernel_size, stride= 1 , padding= 1 ))
		#self.features.add_module( "bn5", nn.BatchNorm2d(num_features = 16));
		#self.features.add_module( "relu5", nn.ReLU())
		
		#self.features.add_module( "conv6" , nn.Conv2d( 16 , 16 , args.kernel_size, stride= 1 , padding= 1 ))
		#self.features.add_module( "bn6", nn.BatchNorm2d(num_features = 16));
		#self.features.add_module( "relu6", nn.ReLU())
		
		#self.features.add_module( "conv7" , nn.Conv2d( 16 , 16 , args.kernel_size, stride= 1 , padding= 1 ))
		#self.features.add_module( "bn7", nn.BatchNorm2d(num_features = 16));
		#self.features.add_module( "relu7", nn.ReLU())
	
		#self.features.add_module( "conv8" , nn.Conv2d( 16 , 16 , args.kernel_size, stride= 1 , padding= 1 ))
		#self.features.add_module( "bn8", nn.BatchNorm2d(num_features = 16));
		#self.features.add_module( "relu8", nn.ReLU())
		
		#self.features.add_module( "conv9" , nn.Conv2d( 16 , 16 , args.kernel_size, stride= 1 , padding= 1 ))
		#self.features.add_module( "bn9", nn.BatchNorm2d(num_features = 16));
		#self.features.add_module( "relu9", nn.ReLU())
		
		#self.features.add_module( "conv10" , nn.Conv2d( 16 , 16 , args.kernel_size, stride= 1 , padding= 1 ))
		#self.features.add_module( "bn10", nn.BatchNorm2d(num_features = 16));
		#self.features.add_module( "relu10", nn.ReLU())
		
		#self.features.add_module( "conv11" , nn.Conv2d( 16 , 16 , args.kernel_size, stride= 1 , padding= 1 ))
		#self.features.add_module( "bn11", nn.BatchNorm2d(num_features = 16));
		#self.features.add_module( "relu11", nn.ReLU())
		
		#self.features.add_module( "conv12" , nn.Conv2d( 16 , 16 , args.kernel_size, stride= 1 , padding= 1 ))
		#self.features.add_module( "bn12", nn.BatchNorm2d(num_features = 16));
		#self.features.add_module( "relu12", nn.ReLU())

		self.lin2 = nn.Linear( 4 * 14 * 14 , 4096 )
		self.relu2 = nn.ReLU()

		self.lin3 = nn.Linear( 4096 , 4096 )
		self.relu3 = nn.ReLU()

		self.lin4 = nn.Linear( 4096 , 4096 )
		self.relu4 = nn.ReLU()

		self.lin5 = nn.Linear( 4096 , 4096 )
		self.relu5 = nn.ReLU()

		self.lin6 = nn.Linear( 4096 , 4096 )
		self.relu6 = nn.ReLU()

		self.lin7 = nn.Linear( 4096 , 4096 )
		self.relu7 = nn.ReLU()

		self.lin8 = nn.Linear( 4096 , 4096 )
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
		out = self.features(x)
		out = out.view(out.size( 0 ), -1 )
		out = self.lin2(out)
		out =self.relu2(out)
		out = self.lin3(out)
		out =self.relu3(out)
		out = self.lin4(out)
		out =self.relu4(out)
		out = self.lin5(out)
		out =self.relu5(out)
		out = self.lin6(out)
		out =self.relu6(out)
		out = self.lin7(out)
		out =self.relu7(out)
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
#norm = []
#def printnorm(self, input, output):
	#input is a tuple of packed inputs
	#output is a Variable. output.data is the Tensor we are interested
	#print('Inside ' + self.__class__.__name__ + ' forward')
#	print('output norm:', output.data.norm())
#	norm.append(output.data.norm().item())


model = SimpleNet(args).to(args.device)
print(model)
#model.features.conv2.register_forward_hook(printnorm)
#model.features.relu2.register_forward_hook(printnorm)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

summary(model, (1,28,28))

#Training the model
for epoch in range(num_epochs):
	for i, (images, labels) in enumerate(train_loader):
		#print(images.size()) #tensor size(100*1*28*28)
		#image size should be (batch_size, channel, 28, 28)
		#images = Variable(images.view( -1 , 28 * 28 ))#reshape the tensor
		#For fully connected layer is like 
		#100 * 2D arrays transform to (100,784) single 2D array
		images = Variable(images).to(args.device)
		labels = Variable(labels).to(args.device)
		# Forward + Backward + Optimize
		optimizer.zero_grad()
		outputs = model(images)
		loss = criterion(outputs, labels)
		# (1)
		loss.backward()
		# (2)
		optimizer.step()
		# (3)
		if (i + 1 ) % 100 == 0 :
			print( 'Epoch: [% d/% d], Step: [% d/% d], Loss: %.4f'
				% (epoch + 1 , num_epochs, i + 1 ,
				len(train_dataset) // batch_size, loss.data.item()))
	print('part3.8 -- Epoch:[%d/%d], Loss: %.4f' %(epoch + 1, num_epochs, loss.data.item()))


# Test the Model
correct = 0
total = 0
epochTest = 0
for images, labels in test_loader:
	images = Variable(images).to(args.device)
	labels = Variable(labels).to(args.device)
	outputs = model(images)
	_, predicted = torch.max(outputs.data, 1 )
	total += labels.size( 0 )
	correct += (predicted.to(args.device) == labels.to(args.device)).sum()
	if(epochTest +1)% 10 == 0 :
		print('Epoch:[%d/%d], accuracy: %d %%' %((epochTest + 1)/10, num_epochs, (100 * correct/total)))
	epochTest = epochTest + 1
print('total :%d, correct : %d' %(total, correct))
print( 'Accuracy of the model on the 10000 test images: % f %%' % ( float(100) * correct /total))
