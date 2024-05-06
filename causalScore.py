import numpy as np
import torch
import torch.nn.functional as F
import copy


class Score(object):
    def __init__(self, inputdataP, inputdataS):
        self.inputdataP = inputdataP
        self.inputdataS = inputdataS

    def setParameters(self, inputdataP, inputdataS):
        self.inputdataP = inputdataP
        self.inputdataS = inputdataS

    def train(self):
        pass

    def update(self):
        pass

    def calScore(self):
        return 0


class GlobalAvgPool2d(torch.nn.Module):

    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])


class FlattenLayer(torch.nn.Module):

    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):
        # x shape: (batch_size, *, *, ...)
        return x.view(x.shape[0], -1)


class BaseNet(torch.nn.Module):   # (-1, 1, 89, 1)
    def __init__(self):
        super(BaseNet, self).__init__()
        self.cn1 = torch.nn.Conv2d(1, 16, kernel_size=(3, 1), padding=1, stride=2)
        self.cn2 = torch.nn.Conv2d(16, 32, kernel_size=(3, 1), padding=1, stride=2)
        self.cn3 = torch.nn.Conv2d(32, 64, kernel_size=(3, 1), padding=1, stride=2)
        self.global_avg_pool = GlobalAvgPool2d()
        self.fc = torch.nn.Linear(64, 1)

    def forward(self, x):
        out = F.leaky_relu(self.cn1(x.reshape((-1, 1, 89, 1))), negative_slope=0.33)
        out = F.leaky_relu(self.cn2(out), negative_slope=0.33)
        out = F.leaky_relu(self.cn3(out), negative_slope=0.33)
        out = self.global_avg_pool(out)
        out = out.view(out.size(0), -1)
        out = F.leaky_relu(self.fc(out), negative_slope=0.33)
        return out


class NolinerSEM(torch.nn.Module):
    def __init__(self):
        super(NolinerSEM, self).__init__()
        self.models = torch.nn.ModuleList([BaseNet() for i in range(90)])

    def forward(self, x):
        outs = []
        for i in range(90):
            col = np.zeros((90, 90))
            for j in range(90):
                col[i, j] = 1
            col[i, i] = 0
            cols_TrueFalse = col > 0.5
            X_train = x[:, 0, cols_TrueFalse]
            outs.append(self.models[i](X_train))
        return torch.cat(outs, dim=1)



class BICN(Score):

    def __init__(self, inputdataP=np.ones((5,5)), inputdataS=np.ones((1,1)), lr=1e-5, wd=1e-5):
        super(BICN, self).__init__(inputdataP, inputdataS)
        self.model = NolinerSEM().cuda()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        self.criterion = torch.nn.SmoothL1Loss()
        self.model_state_dict = [copy.deepcopy(self.model.state_dict()) for i in range(90)]
        self.optimizer_state_dict = [copy.deepcopy(self.optimizer.state_dict()) for i in range(90)]

    def setParameters(self, inputdataP, inputdataS):
        super().setParameters(inputdataP, inputdataS)

    def train(self, iterator=1, trainloader=None, validloader=None, net=None):
        for i in range(90):
            self.model.load_state_dict(self.model_state_dict[i])
            self.optimizer.load_state_dict(self.optimizer_state_dict[i])
            model_state_dict = copy.deepcopy(self.model.state_dict())
            optimizer_state_dict = copy.deepcopy(self.optimizer.state_dict())
            minloss = 0
            self.model.eval()
            for batch_idx, (input, _) in enumerate(validloader):
                with torch.no_grad():
                    noise = np.random.normal(0, 1, size=input.shape).astype(np.float32)
                    #noise = torch.normal(0, 1, size=input.shape)
                    for a in range(89):
                        for b in range(a, 90):
                            tempt = (noise[:, 0, a, b] + noise[:, 0, b, a])/2
                            noise[:, 0, a, b] = tempt
                            noise[:, 0, b, a] = tempt
                    input += noise
                    input = input.cuda()
                    output, concept = net(input)
                    X_train = input
                    x = X_train.clone().detach()
                    y = torch.cat([concept[:, i].clone().detach().reshape((-1, 1)) for k in range(90)], dim=1)
                    output = self.model(x)
                    loss_ = self.criterion(output, y)
                minloss += loss_.item()
            index = -1

            for j in range(iterator):
                loss = 0
                self.model.train()
                for batch_idx, (input, _) in enumerate(trainloader):
                    with torch.no_grad():
                        noise = np.random.normal(0, 1, size=input.shape).astype(np.float32)
                        #noise = torch.normal(0, 1, size=input.shape)
                        for a in range(89):
                            for b in range(a, 90):
                                tempt = (noise[:, 0, a, b] + noise[:, 0, b, a]) / 2
                                noise[:, 0, a, b] = tempt
                                noise[:, 0, b, a] = tempt
                        input += noise
                        input = input.cuda()
                        output, concept = net(input)
                    X_train = input
                    x = X_train.clone().detach()
                    y = torch.cat([concept[:, i].clone().detach().reshape((-1, 1)) for k in range(90)], dim=1)
                    self.optimizer.zero_grad()
                    output = self.model(x)
                    loss_ = self.criterion(output, y)
                    loss_.backward()
                    self.optimizer.step()

                self.model.eval()
                for batch_idx, (input, _) in enumerate(validloader):
                    with torch.no_grad():
                        noise = np.random.normal(0, 1, size=input.shape).astype(np.float32)
                        #noise = torch.normal(0, 1, size=input.shape)
                        for a in range(89):
                            for b in range(a, 90):
                                tempt = (noise[:, 0, a, b] + noise[:, 0, b, a]) / 2
                                noise[:, 0, a, b] = tempt
                                noise[:, 0, b, a] = tempt
                        input += noise
                        input = input.cuda()
                        output, concept = net(input)
                        X_train = input
                        x = X_train.clone().detach()
                        y = torch.cat([concept[:, i].clone().detach().reshape((-1, 1)) for k in range(90)], dim=1)
                        output = self.model(x)
                        loss_ = self.criterion(output, y)
                    loss += loss_.item()

                if minloss > loss:
                    minloss = loss
                    index = j
                    model_state_dict = copy.deepcopy(self.model.state_dict())
                    optimizer_state_dict = copy.deepcopy(self.optimizer.state_dict())

            print(index)
            self.model_state_dict[i] = model_state_dict
            self.optimizer_state_dict[i] = optimizer_state_dict

    def update(self, trainloader=None, net=None):
        for i in range(90):
            self.model.load_state_dict(self.model_state_dict[i])
            self.optimizer.load_state_dict(self.optimizer_state_dict[i])

            self.model.train()
            for batch_idx, (input, _) in enumerate(trainloader):
                with torch.no_grad():
                    noise = np.random.normal(0, 1, size=input.shape).astype(np.float32)
                    #noise = torch.normal(0, 1, size=input.shape)
                    for a in range(89):
                        for b in range(a, 90):
                            tempt = (noise[:, 0, a, b] + noise[:, 0, b, a])/2
                            noise[:, 0, a, b] = tempt
                            noise[:, 0, b, a] = tempt
                    input += noise
                    input = input.cuda()
                    output, concept = net(input)
                X_train = input
                x = X_train.clone().detach()
                y = torch.cat([concept[:, i].clone().detach().reshape((-1, 1)) for k in range(90)], dim=1)
                self.optimizer.zero_grad()
                output = self.model(x)
                loss_ = self.criterion(output, y)
                loss_.backward()
                self.optimizer.step()

            self.model_state_dict[i] = copy.deepcopy(self.model.state_dict())
            self.optimizer_state_dict[i] = copy.deepcopy(self.optimizer.state_dict())

    def calScore(self):
        X_train = self.inputdataP
        x = X_train.clone().detach()
        RSS_ls = []
        for i in range(90):
            self.model.load_state_dict(self.model_state_dict[i])
            self.optimizer.load_state_dict(self.optimizer_state_dict[i])
            self.model.eval()
            with torch.no_grad():
                pred = self.model(x)
                RSS_ls.append(torch.reshape(pred.clone().detach(), (-1, 1, 90)))
        return torch.cat(RSS_ls, 1)





