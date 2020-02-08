import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import argparse

# argument parser
parser = argparse.ArgumentParser(description='ML_CODESIGN Lab1 - MNIST example')
parser.add_argument('--batch-size', type=int, default=100, help='Number of samples per mini-batch')
parser.add_argument('--epochs', type=int, default=10, help='Number of epoch to train')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--enable_cuda', type=bool, default=1, help='1 to enable cuda')
args = parser.parse_args()

# Hyper Parameters
input_size = 784
num_classes = 10
num_epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.lr
args.device = None
if args.enable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
else:
        args.device = torch.device('cpu')  

print('enable cuda is ', args.enable_cuda, ' device is ', args.device)

# MNIST Dataset (Images and Labels)
train_dataset = dsets.MNIST(root ='./data',
        train = True,
        transform = transforms.ToTensor(),
        download = True)

test_dataset = dsets.MNIST(root ='./data',
        train = False,
        transform = transforms.ToTensor())

# Dataset Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
        batch_size = batch_size,
        shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
        batch_size = batch_size,
        shuffle = False)

# Model
# class LogisticRegression(nn.Module):
#     def __init__(self, input_size, num_classes):
#         super(LogisticRegression, self).__init__()
#         self.linear = nn.Linear(input_size, num_classes)

#     def forward(self, x):
#         out = self.linear(x)
#         return out


class SimpleNet(nn.Module):
    def __init__(self, args):
        super(SimpleNet, self).__init__()
        self.features = nn.Sequential()
        self.features.add_module("conv1", nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1))
        self.features.add_module("relu1", nn.ReLU())
        self.features.add_module("pool1", nn.MaxPool2d(kernel_size=2, stride=2))
        self.features.add_module("conv2", nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1))
        self.features.add_module("relu2", nn.ReLU())
        self.features.add_module("pool2", nn.MaxPool2d(kernel_size=2, stride=2))
        self.lin1 = nn.Linear(7 * 7 * 16, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.lin1(out)
        return out

norms = []
def printnorm(self, input, output):
    # input is a tuple of packed inputs
    # output is a Variable. output.data is the Tensor we are interested
    # print('Inside ' + self.__class__.__name__ + ' forward')
    # print('')
    # print('input: ', type(input))
    # print('input[0]: ', type(input[0]))
    # print('output: ', type(output))
    # print('')
    # print('input size:', input[0].size())
    # print('output size:', output.data.size())
    # print('output norm:', output.data.norm())
    #print('output norm:', output.data.norm().item())
    norms.append(output.data.norm().item())


visualization = {}
visualization2 = {}


def printactivation(m, i, o):
    visualization[m] = o
    print("after conv")
    print(o)

def printactivation2(m, i, o):
    visualization2[m] = o
    print("after relu")
    print(o)


print(norms)

model = SimpleNet(args).to(args.device)
print(model)
model.features.conv2.register_forward_hook(printnorm)

model.features.conv2.register_forward_hook(printactivation)
model.features.relu2.register_forward_hook(printactivation2)

#print (model.conv1.weight.data.norm())


# Loss and Optimizer
# Softmax is internally computed.
# Set parameters to be updated.
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)


# Training the Model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        ##images = Variable(images.view(-1, 28 * 28)).to(args.device)
        labels = Variable(labels).to(args.device)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        # (1)
        #l1_regularization = torch.tensor(0)
        #for param in model.parameters():
        #        l1_regularization += torch.norm(param, 1)
        #print(l1_regularization)
               #loss = (.1 * l1_regularization) + loss
        #l1_norm = torch.tensor
        #for param in model.parameters():
        #        l1_norm += torch.norm(param, p=1)


        ##l1_norm = torch.norm(model.linear.weight, p=1) *.01
        ##loss += l1_norm
        
        #print(model.linear.weight)


        #print(l1_norm)


        loss.backward()
        # (2)
        optimizer.step()
        # (3)

        if (i + 1) % 100 == 0:
            print('Epoch: [% d/% d], Step: [% d/% d], Loss: %.4f'
                    % (epoch + 1, num_epochs, i + 1,
                       len(train_dataset) // batch_size, loss.data.item()))

print(norms)

# Test the Model
correct = 0
total = 0
for images, labels in test_loader:
    ##images = Variable(images.view(-1, 28 * 28)).to(args.device)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.to(args.device) == labels.to(args.device)).sum()

print('Accuracy of the model on the 10000 test images: % d %%' % (100 * correct / total))