import torch
import torch.nn.functional as F


class EdgeToEdgeElementWis(torch.nn.Module):
    """(X.shape[0], 1, 90, 90)"""

    def __init__(self, in_planes, planes, example, bias=False):
        super(EdgeToEdgeElementWis, self).__init__()
        self.d = example.size(3)  # 90
        self.in_planes = in_planes
        self.planes = planes
        self.weight = torch.nn.Parameter(torch.Tensor(self.planes, self.in_planes, self.d, self.d))   # O I 90 90
        torch.nn.init.xavier_uniform_(self.weight)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(self.planes))  # O I 90 90
        else:
            self.bias = None

    def forward(self, x):

        outputs1 = []
        input_slices = torch.split(x, 1, dim=3)
        kernel_slices = torch.split(self.weight, 1, dim=3)
        for input_slice, kernel_slice in zip(input_slices, kernel_slices):
            outputs1.append(F.conv2d(input=input_slice, weight=kernel_slice, bias=self.bias))

        outputs2 = []
        input_slices = torch.split(x, 1, dim=2)
        kernel_slices = torch.split(self.weight, 1, dim=2)
        for input_slice, kernel_slice in zip(input_slices, kernel_slices):
            outputs2.append(F.conv2d(input=input_slice, weight=kernel_slice, bias=self.bias))

        row = torch.cat(outputs1, 3) #1*90
        colum = torch.cat(outputs2, 2) #90*1

        return torch.cat([row] * self.d, 2) + torch.cat([colum] * self.d, 3)


class EdgeToNodeElementWis(torch.nn.Module):
    """(X.shape[0], 1, 90, 90)"""

    def __init__(self, in_planes, planes, example, bias=False):
        super(EdgeToNodeElementWis, self).__init__()
        self.d = example.size(3)#90
        self.in_planes = in_planes
        self.planes = planes
        self.weight = torch.nn.Parameter(torch.Tensor(self.planes, self.in_planes, self.d, self.d))  # O I 90 90
        torch.nn.init.xavier_uniform_(self.weight)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(self.planes))  # O I 90 90
        else:
            self.bias = None

    def forward(self, x):
        """
        outputs1 = []
        input_slices = torch.split(x, 1, dim=3)
        kernel_slices = torch.split(self.weight, 1, dim=3)
        for input_slice, kernel_slice in zip(input_slices, kernel_slices):
            outputs1.append(F.conv2d(input=input_slice, weight=kernel_slice, bias=self.bias))
        return torch.cat(outputs1, 3)
        """
        outputs2 = []
        input_slices = torch.split(x, 1, dim=2)
        kernel_slices = torch.split(self.weight, 1, dim=2)
        for input_slice, kernel_slice in zip(input_slices, kernel_slices):
            outputs2.append(F.conv2d(input=input_slice, weight=kernel_slice, bias=self.bias))
        return torch.cat(outputs2, 2)


class E2EBlock(torch.nn.Module):
    """(X.shape[0], 1, 90, 90)"""

    def __init__(self, in_planes, planes, example, bias=False):
        super(E2EBlock, self).__init__()
        self.d = example.size(3)  # 90
        self.cnn1 = torch.nn.Conv2d(in_planes, planes, (1, self.d), bias=bias)
        self.cnn2 = torch.nn.Conv2d(in_planes, planes, (self.d, 1), bias=bias)

    def forward(self, x):
        a = self.cnn1(x)
        b = self.cnn2(x)
        return torch.cat([a] * self.d, 3) + torch.cat([b] * self.d, 2)


class BrainNetCNNEWScore(torch.nn.Module):
    def __init__(self, example):
        super(BrainNetCNNEWScore, self).__init__()
        self.in_planes = example.size(1)
        self.d = example.size(3)
        self.e2econv1 = EdgeToEdgeElementWis(1, 32, example, bias=False)
        self.E2N = EdgeToNodeElementWis(32, 64, example, bias=True)
        self.N2G = torch.nn.Conv2d(64, 128, (self.d, 1))
        self.dense1 = torch.nn.Linear(128, 256)
        self.dense2 = torch.nn.Linear(256, 2)

    def forward(self, x):
        out = F.leaky_relu(self.e2econv1(x),negative_slope=0.33)
        out = self.E2N(out)
        out = F.leaky_relu(out,negative_slope=0.33)
        out = self.N2G(out)
        out = F.leaky_relu(out,negative_slope=0.33)
        out = out.view(out.size(0), -1)
        concept = F.leaky_relu(self.dense1(out), negative_slope=0.33)
        out = F.leaky_relu(self.dense2(concept), negative_slope=0.33)
        return F.log_softmax(out, dim=1), concept

class BrainNetCNNEWAUCScore(torch.nn.Module):
    def __init__(self, example):
        super(BrainNetCNNEWAUCScore, self).__init__()
        self.in_planes = example.size(1)
        self.d = example.size(3)
        self.e2econv1 = EdgeToEdgeElementWis(1, 32, example, bias=False)
        self.E2N = EdgeToNodeElementWis(32, 64, example, bias=True)
        self.N2G = torch.nn.Conv2d(64, 128, (self.d, 1))
        self.dense1 = torch.nn.Linear(128, 256)
        self.dense2 = torch.nn.Linear(256, 1)

    def forward(self, x):
        out = F.leaky_relu(self.e2econv1(x),negative_slope=0.33)
        out = self.E2N(out)
        out = F.leaky_relu(out,negative_slope=0.33)
        out = self.N2G(out)
        out = F.leaky_relu(out,negative_slope=0.33)
        out = out.view(out.size(0), -1)
        concept = F.leaky_relu(self.dense1(out), negative_slope=0.33)
        out = torch.sigmoid(self.dense2(concept))
        return out, concept

class BrainNetCNNEW(torch.nn.Module):
    def __init__(self, example):
        super(BrainNetCNNEW, self).__init__()
        self.in_planes = example.size(1)
        self.d = example.size(3)
        self.e2econv1 = EdgeToEdgeElementWis(1, 32, example, bias=False)
        self.E2N = EdgeToNodeElementWis(32, 64, example, bias=True)
        self.N2G = torch.nn.Conv2d(64, 128, (self.d, 1))
        self.dense1 = torch.nn.Linear(128, 64)
        self.dense2 = torch.nn.Linear(64, 2)

    def forward(self, x):
        out = F.leaky_relu(self.e2econv1(x),negative_slope=0.33)
        out = self.E2N(out)
        out = F.leaky_relu(out,negative_slope=0.33)
        out = self.N2G(out)
        out = F.leaky_relu(out,negative_slope=0.33)
        out = out.view(out.size(0), -1)
        out = F.leaky_relu(self.dense1(out), negative_slope=0.33)
        out = F.leaky_relu(self.dense2(out), negative_slope=0.33)
        return F.log_softmax(out, dim=1)

class BrainNetCNNScore(torch.nn.Module):
    def __init__(self, example):
        super(BrainNetCNNScore, self).__init__()
        self.in_planes = example.size(1)
        self.d = example.size(3)
        self.e2econv1 = E2EBlock(1, 32, example, bias=True)
        self.E2N = torch.nn.Conv2d(32, 64, (1, self.d))
        self.N2G = torch.nn.Conv2d(64, 128, (self.d, 1))
        self.dense1 = torch.nn.Linear(128, 256)
        #self.dense2 = torch.nn.Linear(256, 128)
        #self.dense3 = torch.nn.Linear(128, 256)
        self.dense4 = torch.nn.Linear(256, 2)

    def forward(self, x):
        out = F.leaky_relu(self.e2econv1(x), negative_slope=0.33)
        out = self.E2N(out)
        out = F.leaky_relu(out, negative_slope=0.33)
        out = self.N2G(out)
        out = F.leaky_relu(out, negative_slope=0.33)
        out = out.view(out.size(0), -1)
        concept = F.leaky_relu(self.dense1(out), negative_slope=0.33)
        #out = F.leaky_relu(self.dense2(out), negative_slope=0.33)
        #concept = F.leaky_relu(self.dense3(out), negative_slope=0.33)
        out = F.leaky_relu(self.dense4(concept), negative_slope=0.33)
        return F.log_softmax(out, dim=1), concept


class BrainNetCNN(torch.nn.Module):
    def __init__(self, example):
        super(BrainNetCNN, self).__init__()
        self.in_planes = example.size(1)
        self.d = example.size(3)
        self.e2econv1 = E2EBlock(1, 32, example, bias=True)
        self.E2N = torch.nn.Conv2d(32, 64, (1, self.d))
        self.N2G = torch.nn.Conv2d(64, 128, (self.d, 1))
        self.dense1 = torch.nn.Linear(128, 64)
        self.dense2 = torch.nn.Linear(64, 2)

    def forward(self, x):
        out = F.leaky_relu(self.e2econv1(x),negative_slope=0.33)
        out = self.E2N(out)
        out = F.leaky_relu(out,negative_slope=0.33)
        out = self.N2G(out)
        out = F.leaky_relu(out,negative_slope=0.33)
        out = out.view(out.size(0), -1)
        out = F.leaky_relu(self.dense1(out), negative_slope=0.33)
        out = F.leaky_relu(self.dense2(out), negative_slope=0.33)
        return F.log_softmax(out, dim=1)


class CNNScore(torch.nn.Module):
    def __init__(self, example, num_classes=2):
        super(CNNScore, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=3,
                padding=0),
            torch.nn.LeakyReLU(negative_slope=0.33),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(8, 16, kernel_size=3, padding=0),
            torch.nn.LeakyReLU(negative_slope=0.33),
            torch.nn.MaxPool2d(2)
        )
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(7056, 256),
            torch.nn.LeakyReLU(negative_slope=0.33),
        )
        self.fc2 = torch.nn.Linear(256, 2)

    def forward(self, x):
        out = self.conv(x)
        out = torch.flatten(out, start_dim=1)
        concept = self.fc1(out)
        out = self.fc2(concept)
        return F.log_softmax(out, dim=1), concept


class CNN(torch.nn.Module):
    def __init__(self, example, num_classes=2):
        super(CNN, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=3,
                padding=0),
            torch.nn.LeakyReLU(negative_slope=0.33),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(8, 16, kernel_size=3, padding=0),
            torch.nn.LeakyReLU(negative_slope=0.33),
            torch.nn.MaxPool2d(2)
        )
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(576, 64),
            torch.nn.LeakyReLU(negative_slope=0.33),
        )
        self.fc2 = torch.nn.Linear(64, 2)

    def forward(self, x):
        out = self.conv(x)
        out = torch.flatten(out, start_dim=1)
        out = self.fc1(out)
        out = self.fc2(out)
        return F.log_softmax(out, dim=1)
