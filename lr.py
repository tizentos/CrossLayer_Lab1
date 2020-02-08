import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import argparse

# argument parser
parser = argparse.ArgumentParser(description= 'ML_CODESIGN Lab1 - MNIST example' )
parser.add_argument( '--batch-size' , type=int, default= 100 , help= 'Number of samples per mini-batch' )
parser.add_argument( '--epochs' , type=int, default= 10 , help= 'Number of epoch to train' )
parser.add_argument( '--lr' , type=float, default= 0.001 , help= 'Learning rate' )
parser.add_argument( '--enable_cuda', help= 'Enable GPU')
args = parser.parse_args()
args.device = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

#select device for training
if args.enable_cuda and torch.cuda.is_available():
	args.device = torch.device('cuda')
else:
	args.device = torch.device('cpu')
# Hyper Parameters

input_size = 784
num_classes = 10
num_epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.lr

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

# Model
class LogisticRegression (nn.Module):
	def __init__ (self, input_size, num_classes):
		super(LogisticRegression, self).__init__()
		self.linear = nn.Linear(input_size, num_classes)
	def forward (self, x):
		out = self.linear(x)
		return out

model = LogisticRegression(input_size, num_classes).to(device = args.device)

# Loss and Optimizer
# Softmax is internally computed.
# Set parameters to be updated.

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

# Training the Model
for epoch in range(num_epochs):
	for i, (images, labels) in enumerate(train_loader):
		images = Variable(images.view( -1 , 28 * 28 )).to(device = args.device)
		labels = Variable(labels).to(device = args.device)
		
		# Forward + Backward + Optimize
		optimizer.zero_grad()
		outputs = model(images)
		loss = criterion(outputs, labels)
		# (1)
		loss = loss + torch.norm(model.linear.weight,p=1)/3000
		loss.backward()
		# (2)
		optimizer.step()
		# (3)
		if (i + 1 ) % 100 == 0 :
			print( 'Epoch: [% d/% d], Step: [% d/% d], Loss: %.4f'
				% (epoch + 1 , num_epochs, i + 1 ,
				len(train_dataset) // batch_size, loss.data.item()))

#access weight in linear layer
print(model.linear.weight)
print(torch.norm(model.linear.weight,p=1))
# Test the Model
correct = 0
total = 0
for images, labels in test_loader:
	images = Variable(images.view( -1 , 28 * 28 ))
	outputs = model(images)
	_, predicted = torch.max(outputs.data, 1 )
	total += labels.size( 0 )
	correct += (predicted == labels).sum()

print( 'Accuracy of the model on the 10000 test images: % d %%' % ( 100 * correct /
total))
