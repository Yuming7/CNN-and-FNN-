import torch

import torch.nn as nn
import torch.nn.functional as F


class FNN(nn.Module):
    def __init__(self, loss_type, num_classes):
        super(FNN, self).__init__()
        self.loss_type = loss_type
        self.num_classes = num_classes
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)



        """add your code here"""


    def forward(self, x):
        output = None
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.tanh(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)



        output = F.log_softmax(x, dim=1)

        """add your code here"""

        return output

    def get_loss(self, output, target):
        loss = None
        if(self.loss_type == "ce"):
            loss = F.cross_entropy(output,target)

        else:
            one_hot = torch.nn.functional.one_hot
            target = target.view(target.size(0), -1)
            loss = F.mse_loss(output, one_hot(target,num_classes=10).float())


        """add your code here"""

        return loss