
import timeit
from collections import OrderedDict

import torch
from torchvision import transforms, datasets

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split


#TO DO: Complete this with your CNN architecture. Make sure to complete the architecture requirements.
#The init has in_channels because this changes based on the dataset.

class Net(nn.Module):
    def __init__(self, in_channels):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)

        if in_channels == 3:
            number = 1600
        else:
            number = 1024
        self.fc1 = nn.Linear(number, 100)
        self.fc2 = nn.Linear(100, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x,dim=1)


#Function to get train and validation datasets. Please do not make any changes to this function.
def load_dataset(
        dataset_name: str,
):
    if dataset_name == "MNIST":
        full_dataset = datasets.MNIST('./data', train=True, download=True,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))]))

        train_dataset, valid_dataset = random_split(full_dataset, [48000, 12000])

    elif dataset_name == "CIFAR10":
        full_dataset = datasets.CIFAR10(root='./data', train=True, download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

        train_dataset, valid_dataset = random_split(full_dataset, [38000, 12000])

    else:
        raise Exception("Unsupported dataset.")

    return train_dataset, valid_dataset



#TO DO: Complete this function. This should train the model and return the final trained model.
#Similar to Assignment-1, make sure to print the validation accuracy to see
#how the model is performing.

def train(
        model,
        train_dataset,
        valid_dataset,
        device,
        dataset_name
):
    #Make sure to fill in the batch size.

  if dataset_name == "MNIST":
    n_epochs = 10
    batch_size_train = 100
    batch_size_test = 1000
    learning_rate = 1e-4
    momentum = 0.5
    log_interval = 100
    WeightDecay = 0.0001

    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    # Checking GPU availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=WeightDecay)

    one_hot = torch.nn.functional.one_hot
    one_hot = torch.nn.functional.one_hot

  else:
    n_epochs = 15
    batch_size_train = 100
    batch_size_test = 1000
    learning_rate = 1e-3
    momentum = 0.5
    log_interval = 100
    WeightDecay = 0.0001

    # Checking GPU availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=WeightDecay)
  one_hot = torch.nn.functional.one_hot
  def train(epoch,data_loader,model,optimizer):

    for batch_idx, (data, target) in enumerate(data_loader):
      data = data.to(device)
      target = target.to(device)
      optimizer.zero_grad()
      output = model(data)
      loss = F.cross_entropy(output, one_hot(target,num_classes=10).float())
      loss.backward()
      optimizer.step()
      if batch_idx % log_interval == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch, batch_idx * len(data), len(data_loader.dataset),
          100. * batch_idx / len(data_loader), loss.item()))
  train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size_train, shuffle=True)

  valid_loader = torch.utils.data.DataLoader(
      valid_dataset, batch_size=batch_size_train, shuffle=False)

  def eval(data_loader,model,dataset):
    loss = 0
    correct = 0
    with torch.no_grad(): # notice the use of no_grad
      for data, target in data_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
        loss += F.mse_loss(output, one_hot(target,num_classes=10).float(), size_average=False).item()
    loss /= len(data_loader.dataset)
    print(dataset+'set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(loss, correct, len(data_loader.dataset), 100. * correct / len(data_loader.dataset)))

  eval(valid_loader,model,"Validation")
  for epoch in range(1, n_epochs + 1):
    train(epoch,train_loader,model,optimizer)
    eval(valid_loader,model,"Validation")



  results = dict(
      model=model
  )

  return results

def CNN(dataset_name, device):

    #CIFAR-10 has 3 channels whereas MNIST has 1.
    if dataset_name == "CIFAR10":
        in_channels= 3
    elif dataset_name == "MNIST":
        in_channels = 1
    else:
        raise AssertionError(f'invalid dataset: {dataset_name}')

    model = Net(in_channels).to(device)

    train_dataset, valid_dataset = load_dataset(dataset_name)

    results = train(model, train_dataset, valid_dataset, device, dataset_name)

    return results
