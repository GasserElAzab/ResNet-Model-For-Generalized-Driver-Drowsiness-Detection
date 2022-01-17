import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import tools
import tests

data_dir = tools.select_data_dir('../')
source_dir = ''
data_dir1 = os.path.join(data_dir,'data/all_data')

device = torch.device('cuda:0') if (torch.cuda.is_available()) else torch.device('cpu')




data_transform = transforms.Compose([
        transforms.Resize((28,28)),    
        transforms.Grayscale(),               
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Scale images to [-1, 1]
    ])

all_dataset = datasets.ImageFolder(data_dir1, transform=data_transform)

print(len(all_dataset))
trainset, testset = torch.utils.data.random_split(all_dataset, [len(all_dataset)-780, 780])

classes = ['alert','closed_eyes', 'no_yawn', 'non_vigilant', 'open_eyes', 'tired', 'yawn']

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=5, shuffle=False)

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Args:
          in_channels (int):  Number of input channels.
          out_channels (int): Number of output channels.
          stride (int):       Controls the stride.
        """
        super(Block, self).__init__()
        
        self.block_layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels , 3, stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # YOUR CODE HERE
        if in_channels == out_channels:
            if stride != 1:
                #print("Stride > 1, same channels  ", stride)
                self.block_skip = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels , 1, stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.block_skip = nn.Sequential()
        else:
            #print("Unequal channels  ", stride)
            self.block_skip = nn.Sequential(    
                nn.Conv2d(in_channels, out_channels , 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
        self.block_relu = nn.Sequential(
            nn.ReLU()
        )
        
    def forward(self, x):
        # YOUR CODE HERE
        #print("Input :" , x.shape)
        y = self.block_layers(x)
        #print("Layers :" , y.shape)
        a = self.block_skip(x)
        #print("Skip :" , a.shape)
        y = y + a
        #print("Before relu :" , y.shape)
        y = self.block_relu(y)
        #print("After relu :" , y.shape)
        return y
    
    
def test_Block_shapes():

    # The number of channels and resolution do not change
    batch_size = 20
    x = torch.zeros(batch_size, 16, 28, 28)
    block = Block(in_channels=16, out_channels=16)
    y = block(x)
    assert y.shape == torch.Size([batch_size, 16, 28, 28]), "Bad shape of y: y.shape={}".format(y.shape)

    # Increase the number of channels
    block = Block(in_channels=16, out_channels=32)
    y = block(x)
    assert y.shape == torch.Size([batch_size, 32, 28, 28]), "Bad shape of y: y.shape={}".format(y.shape)

    # Decrease the resolution
    block = Block(in_channels=16, out_channels=16, stride=2)
    y = block(x)
    assert y.shape == torch.Size([batch_size, 16, 14, 14]), "Bad shape of y: y.shape={}".format(y.shape)

    # Increase the number of channels and decrease the resolution
    block = Block(in_channels=16, out_channels=32, stride=2)
    y = block(x)
    assert y.shape == torch.Size([batch_size, 32, 14, 14]), "Bad shape of y: y.shape={}".format(y.shape)

    print('Success')


# We implement a group of blocks in this cell
class GroupOfBlocks(nn.Module):
    def __init__(self, in_channels, out_channels, n_blocks, stride=1):
        super(GroupOfBlocks, self).__init__()

        first_block = Block(in_channels, out_channels, stride)
        other_blocks = [Block(out_channels, out_channels) for _ in range(1, n_blocks)]
        self.group = nn.Sequential(first_block, *other_blocks)

    def forward(self, x):
        return self.group(x)
    


class ResNet(nn.Module):
    def __init__(self, n_blocks, n_channels=64, num_classes=7):
        """
        Args:
          n_blocks (list):   A list with three elements which contains the number of blocks in 
                             each of the three groups of blocks in ResNet.
                             For instance, n_blocks = [2, 4, 6] means that the first group has two blocks,
                             the second group has four blocks and the third one has six blocks.
          n_channels (int):  Number of channels in the first group of blocks.
          num_classes (int): Number of classes.
        """
        print(len(n_blocks) == 3)
        assert len(n_blocks) == 3, "The number of groups should be three."
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=n_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.group1 = GroupOfBlocks(n_channels, n_channels, n_blocks[0])
        self.group2 = GroupOfBlocks(n_channels, 2*n_channels, n_blocks[1], stride=2)
        self.group3 = GroupOfBlocks(2*n_channels, 4*n_channels, n_blocks[2], stride=2)

        self.avgpool = nn.AvgPool2d(kernel_size=4, stride=1)
        self.fc = nn.Linear(4*n_channels, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, verbose=False):
        """
        Args:
          x of shape (batch_size, 1, 28, 28): Input images.
          verbose: True if you want to print the shapes of the intermediate variables.
        
        Returns:
          y of shape (batch_size, 10): Outputs of the network.
        """
        if verbose: print(x.shape)
        x = self.conv1(x)
        if verbose: print('conv1:  ', x.shape)
        x = self.bn1(x)
        if verbose: print('bn1:    ', x.shape)
        x = self.relu(x)
        if verbose: print('relu:   ', x.shape)
        x = self.maxpool(x)
        if verbose: print('maxpool:', x.shape)

        x = self.group1(x)
        if verbose: print('group1: ', x.shape)
        x = self.group2(x)
        if verbose: print('group2: ', x.shape)
        x = self.group3(x)
        if verbose: print('group3: ', x.shape)

        x = self.avgpool(x)
        if verbose: print('avgpool:', x.shape)

        x = x.view(-1, self.fc.in_features)
        if verbose: print('x.view: ', x.shape)
        x = self.fc(x)
        if verbose: print('out:    ', x.shape)

        return x


def test_ResNet_shapes():
    # Create a network with 2 block in each of the three groups
    parser = argparse.ArgumentParser(description='Training the neural network')
    n_blocks =  parser.add_argument('-b', '--blocks', default=[2,2,2], help="the number of blocks chosen")
    print("chosen number of blocks is {}".format(n_blocks))
    net = ResNet(n_blocks, n_channels=10)
    net.to(device)

    # Feed a batch of images from the training data to test the network
    with torch.no_grad():
        images, labels = iter(trainloader).next()
        images = images.to(device)
        print('Shape of the input tensor:', images.shape)

        y = net.forward(images, verbose=True)
        print(y.shape)
        assert y.shape == torch.Size([trainloader.batch_size, 7]), "Bad shape of y: y.shape={}".format(y.shape)

    print('Success')

# test_ResNet_shapes()

# This function computes the accuracy on the test dataset
def compute_accuracy(net, testloader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# Create the network
parser = argparse.ArgumentParser(description='Training the neural network')
parser.add_argument('-b', '--blocks',nargs='+', default=[2,2,2], help="the number of blocks chosen")
parser.add_argument('-lr', '--lr',nargs='+', default=0.001, help="the number of blocks chosen")
args = parser.parse_args()
n_blocks_str=args.blocks
lr = float(args.lr)
int_map= map(int,n_blocks_str)
n_blocks= list(int_map)

print("chosen number of blocks is {}".format(n_blocks))
skip_training=False
net = ResNet(n_blocks, n_channels=16)
net.to(device)

if not skip_training:
    
    # YOUR CODE HERE
    iteration=[]
    train_accu=[]
    losses=[]
    epochs_arr=[]
    train_accu_per_epoch=[]
    loss_per_epoch=[]
    optimizer = optim.Adam(net.parameters(), lr)
    criterion = nn.CrossEntropyLoss()
    epochs = 30
    net.train()
    for epoch in range(epochs):
        print("Epoch number:  ", epoch)
        for i, data in enumerate(trainloader, 0):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = net(images)
            output.to(device)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            iteration.append(i)
            train_accu.append(compute_accuracy(net,testloader))
            losses.append(loss)
            if i % 32 == 31:
                acc = compute_accuracy(net, testloader)
                print("Accuracy:  ", acc)
                net.train()
        epochs_arr.append(epoch)
        train_accu_per_epoch.append(acc)
        loss_per_epoch.append(loss)
        

#zipped = zip(iteration, train_accu,losses,epochs_arr, train_accu_per_epoch, loss_per_epoch )

#np.savetxt('training_data.csv', zipped, fmt='%i,%i,%i,%i,%i,%i')

dict_saving= {"iteration":iteration,"train_accu":train_accu,"losses":losses, "epochs_arr":epochs_arr,\
    "train_accu_per_epoch":train_accu_per_epoch,"loss_per_epoch":loss_per_epoch}
np.save("training_result_{}.npy".format(n_blocks),dict_saving)
#for loading data2=np.load("...",allow_pickle=True), data2[()]["iteration"]
# Save the model to disk (the pth-files will be submitted automatically together with your notebook)
# Set confirm=False if you do not want to be asked for confirmation before saving.
if not skip_training:
    path = os.path.join(source_dir, 'resnet_all.pth')
    tools.save_model(net, path, confirm=False)
else:
    net = ResNet(n_blocks, n_channels=16)
    tools.load_model(net, 'resnet_all.pth', device)
  
