import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import argparse
from simpleNet import SimpleNet
import matplotlib.pyplot as plt
from math import sqrt,floor

# argument parser
parser = argparse.ArgumentParser(description='ML_CODESIGN Lab1 - MNIST example')
parser.add_argument('--batch-size', type=int, default=100, help='Number of samples per mini-batch')
parser.add_argument('--epochs', type=int, default=10, help='Number of epoch to train')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--enable-cuda',type=bool,default=False,help='Enable cuda')

args = parser.parse_args()

# Hyper Parameters
input_size = 784
num_classes = 10
num_epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.lr
enable_cuda = args.enable_cuda
# enable_cuda =False
lambda_value = 1

# a =[1,2,3,4,5,6,7,8,9.9]
# length = range(len(a))
# plt.plot(range(len(a)),a)
# plt.show()


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
# count = 0
norms = []
reludata = []
conv2data = []
bndata = []
counter = False
# fig = plt.figure()
# ax = fig.add_subplot(111)
# Ln, = ax.plot(norms)


def conv2Hook(self, input, output):
    x = output.data.norm()
    x = x.item()
    norms.append(x)
    if (counter == True):
        print("capture conv2 data")
        data = output.data
        data = torch.reshape(data,(-1,))
        data = data.tolist()
        for x in data:
            conv2data.append(x)
        a = len(conv2data)

def relu2Hook(self, input, output):
    if (counter == True):   
        print("capture relu2 data")       
        x = output.data.norm()
        data = output.data
        data = torch.reshape(data,(-1,))
        data = data.tolist()
        for x in data:
            reludata.append(x)
def bn2Hook(self, input, output):
    if (counter == True):   
        print("capture bn2 data")       
        x = output.data.norm()
        data = output.data
        data = torch.reshape(data,(-1,))
        data = data.tolist()
        for x in data:
            bndata.append(x)

def plot_histogram(data,fignum,scale):
    num_of_element = len(data)
    range_of_element = max(data) - min(data)
    num_of_interval = floor(sqrt(num_of_element))
    width_of_intervals = range_of_element/num_of_interval

    plt.figure(fignum)
    plt.yscale(scale)
    plt.hist(data,bins= num_of_interval)


# Model
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out

# model = LogisticRegression(input_size, num_classes)
model = SimpleNet(args)



device = torch.device("cpu")

if enable_cuda:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Running with Cuda\n\n')
    # model.cuda()
    print(device)
    model.to(device)


# Loss and Optimizer
# Softmax is internally computed.
# Set parameters to be updated.
criterion = nn.CrossEntropyLoss()

print(model)
print(model.features[2])
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
handle = model.features.conv2.register_forward_hook(conv2Hook)
relu_handle = model.features.relu2.register_forward_hook(relu2Hook)
bn2_handle = model.features.bn2.register_forward_hook(bn2Hook)

# Training the Model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # images = Variable(images.view(-1, 28 * 28))
        labels = Variable(labels)
        # torch.device(device)
        images = images.to(device)
        # images = images.unsqueeze(0).unsqueeze(0)
        labels = labels.to(device)
        # images.device(device)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        if (i+1  == len(train_dataset)/batch_size): 
            counter = True
        else:
            counter = False
        outputs = model(images)
        loss = criterion(outputs, labels)

        # (1)
        loss.backward()
        # (2)
        optimizer.step()
        # (3)
        # loss = loss + (torch.norm(model.linear.weight,p=1)*0.0011)
        # count+= 1

        if (i + 1) % 100 == 0:
            print('Epoch: [% d/% d], Step: [% d/% d], Loss: %.4f'
                    % (epoch + 1, num_epochs, i + 1,
                       len(train_dataset) // batch_size, loss.data.item()))



# norm = torch.norm(model.linear.weight, p = 1)
# print (norm)
# Test the Model

handle.remove()
relu_handle.remove()
bn2_handle.remove()


# plt.ion()


correct = 0
total = 0
for images, labels in test_loader:
    # images = Variable(images.view(-1, 28 * 28))
    images = images.to(device)
    labels = labels.to(device)

    outputs = model(images)
    # loss = criterion(outputs,labels)

    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the model on the 10000 test images: % d %%' % (100 * correct / total))

# plt.figure(1)
# plt.plot(range(len(norms)),norms)

# plot_histogram(reludata,2,'log')
plot_histogram(conv2data,3,'log')
plot_histogram(bndata,4,'log')
plt.show()
