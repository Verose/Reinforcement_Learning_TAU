import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Hyper Parameters
input_size = 784
num_classes = 10
num_epochs = 20
batch_size = 100
learning_rate = 1e-3
hidden_layer_size = 500

# MNIST Dataset
train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# Neural Network Model
class Net(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        return out

#Deep Neural Network Model
class DeepNet(nn.Module):
    def __init__(self, input_size,hidden_layer_size ,num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_layer_size)
        self.fc2 = nn.Linear(input_size, hidden_layer_size)
        self.fc3 = nn.Linear(hidden_layer_size, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out

def train_test_NN(net, criterion, optimizer, num_epochs):
    # Train the Model
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Convert torch tensor to Variable
            images = Variable(images.view(-1, 28*28))
            labels = Variable(labels)

            # Forward + Backward + Optimize
            # TODO: implement training code

            net.train()
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output = net(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    # Test the Model
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images.view(-1, 28*28))
        # TODO: implement evaluation code - report accuracy
        total += labels.size(0)

        #with torch.no_grad():
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))


    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

    # Save the Model
    torch.save(net.state_dict(), 'model.pkl')

#create the net
net = Net(input_size, num_classes)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

#train and test the model
train_test_NN(net, criterion, optimizer, num_epochs)

#Find a better optimization configuration
list_learning_rate = [0.1, 0.01, 1e-3, 0.0001]
list_num_epochs = [100, 150, 170, 200]
optimzer_list = []
for i in list_learning_rate:
    optimzer_list.append(torch.optim.SGD(net.parameters(), lr=i))
for epoch_conf in list_num_epochs:
    for optimizer_conf in optimzer_list:
        # train and test the model
        train_test_NN(net, criterion, optimizer_conf, epoch_conf)

