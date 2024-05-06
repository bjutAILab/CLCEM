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
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=(3, 1), padding=1, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=(3, 1), padding=1, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=(3, 1), padding=1, stride=2),
            torch.nn.ReLU(),
        )
        self.model.add_module("global_avg_pool", GlobalAvgPool2d())
        self.model.add_module("fc", torch.nn.Sequential(FlattenLayer(), torch.nn.Linear(64, 1)))

    def forward(self, x):
        return self.model(x.reshape((-1, 1, 89, 1)))


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


class NolinerSEMS(torch.nn.Module):
    def __init__(self):
        super(NolinerSEMS, self).__init__()
        self.models = torch.nn.ModuleList([NolinerSEM() for i in range(90)])

    def forward(self, x):
        outs = []
        for i in range(90):
            outs.append(self.models[i](x).reshape((-1, 1, 90)))
        return torch.cat(outs, dim=1)


class BIC(Score):

    def __init__(self, inputdataP=np.ones((5,5)), inputdataS=np.ones((1,1)), lr=1e-5, wd=1e-5):
        super(BIC, self).__init__(inputdataP, inputdataS)
        #self.model = NolinerSEM().cuda()
        self.model = BaseNet().cuda()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        #self.criterion = torch.nn.MSELoss()
        #self.criterion = torch.nn.L1Loss()
        self.criterion = torch.nn.SmoothL1Loss()
        self.model_state_dict = [copy.deepcopy(self.model.state_dict()) for i in range(90)]
        self.optimizer_state_dict = [copy.deepcopy(self.optimizer.state_dict()) for i in range(90)]

    def setParameters(self, inputdataP, inputdataS):
        super().setParameters(inputdataP, inputdataS)

    def update(self, iterator=1, trainloader=None, validloader=None, net=None):
        for i in range(90):
            col = np.zeros((90, 90))
            for j in range(90):
                col[i, j] = 1
            col[i, i] = 0
            cols_TrueFalse = col > 0.5
            self.model.load_state_dict(self.model_state_dict[i])
            self.optimizer.load_state_dict(self.optimizer_state_dict[i])
            model_state_dict = copy.deepcopy(self.model.state_dict())
            optimizer_state_dict = copy.deepcopy(self.optimizer.state_dict())
            minloss = 0
            self.model.eval()
            for batch_idx, (input, target) in enumerate(validloader):
                with torch.no_grad():
                    input, target = input.cuda(), target.cuda()
                    output, concept = net(input)
                    X_train = input[:, 0, cols_TrueFalse]
                    y_train = concept[:, i]
                    x = X_train.clone().detach()
                    y = y_train.clone().detach().reshape((-1, 1))
                    output = self.model(x)
                    loss_ = self.criterion(output, y)
                    minloss += loss_.item()
            index = -1

            for j in range(iterator):
                loss = 0
                self.model.train()
                for batch_idx, (input, target) in enumerate(trainloader):
                    with torch.no_grad():
                        input, target = input.cuda(), target.cuda()
                        output, concept = net(input)
                    X_train = input[:, 0, cols_TrueFalse]
                    y_train = concept[:, i]
                    x = X_train.clone().detach()
                    y = y_train.clone().detach().reshape((-1, 1))
                    self.optimizer.zero_grad()
                    output = self.model(x)
                    loss_ = self.criterion(output, y)
                    loss_.backward()
                    self.optimizer.step()

                self.model.eval()
                for batch_idx, (input, target) in enumerate(validloader):
                    with torch.no_grad():
                        input, target = input.cuda(), target.cuda()
                        output, concept = net(input)
                        X_train = input[:, 0, cols_TrueFalse]
                        y_train = concept[:, i]
                        x = X_train.clone().detach()
                        y = y_train.clone().detach().reshape((-1, 1))
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

    def calScore(self):
        RSS_ls = []
        for i in range(90):
            col = np.zeros((90, 90))
            for j in range(90):
                col[i, j] = 1
            col[i, i] = 0
            cols_TrueFalse = col > 0.5
            X_train = self.inputdataP[:, 0, cols_TrueFalse]
            y_train = self.inputdataS[:, i]
            x = X_train.clone().detach()
            y = y_train.clone().detach().reshape((-1, 1))

            self.model.load_state_dict(self.model_state_dict[i])
            self.optimizer.load_state_dict(self.optimizer_state_dict[i])
            self.model.eval()
            with torch.no_grad():
                pred = self.model(x)
                RSS_ls.append(torch.reshape(pred.clone().detach(), (-1, 1)))
        return torch.cat(RSS_ls, 1)


class BIC2(Score):
    def __init__(self, inputdataP=np.ones((5,5)), inputdataS=np.ones((1,1)), lr=1e-5, wd=1e-5):
        super(BIC2, self).__init__(inputdataP, inputdataS)
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
            for batch_idx, (input, target) in enumerate(validloader):
                with torch.no_grad():
                    input, target = input.cuda(), target.cuda()
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
                for batch_idx, (input, target) in enumerate(trainloader):
                    with torch.no_grad():
                        input, target = input.cuda(), target.cuda()
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
                for batch_idx, (input, target) in enumerate(validloader):
                    with torch.no_grad():
                        input, target = input.cuda(), target.cuda()
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
            for batch_idx, (input, target) in enumerate(trainloader):
                with torch.no_grad():
                    input, target = input.cuda(), target.cuda()
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


class BIC3(Score):

    def __init__(self, inputdataP=np.ones((5,5)), inputdataS=np.ones((1,1)), lr=1e-5, wd=1e-5):
        super(BIC3, self).__init__(inputdataP, inputdataS)
        self.model = NolinerSEM().cuda()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        self.criterion = torch.nn.SmoothL1Loss()
        self.model_state_dict = None
        self.optimizer_state_dict = None

    def setParameters(self, inputdataP, inputdataS):
        super().setParameters(inputdataP, inputdataS)

    def update(self, iterator=1, trainloader=None, validloader=None, net=None):
        model_state_dict = copy.deepcopy(self.model.state_dict())
        optimizer_state_dict = copy.deepcopy(self.optimizer.state_dict())
        minloss = 0
        self.model.eval()
        for batch_idx, (input, target) in enumerate(validloader):
            with torch.no_grad():
                input, target = input.cuda(), target.cuda()
                output, concept = net(input)
                X_train = input
                y_train = concept
                x = X_train.clone().detach()
                y = y_train.clone().detach()
                output = self.model(x)
                loss_ = self.criterion(output, y)
            minloss += loss_.item()
        index = -1

        for j in range(iterator):
            loss = 0
            self.model.train()
            for batch_idx, (input, target) in enumerate(trainloader):
                with torch.no_grad():
                    input, target = input.cuda(), target.cuda()
                    output, concept = net(input)
                X_train = input
                y_train = concept
                x = X_train.clone().detach()
                y = y_train.clone().detach()
                self.optimizer.zero_grad()
                output = self.model(x)
                loss_ = self.criterion(output, y)
                loss_.backward()
                self.optimizer.step()

            self.model.eval()
            for batch_idx, (input, target) in enumerate(validloader):
                with torch.no_grad():
                    input, target = input.cuda(), target.cuda()
                    output, concept = net(input)
                    X_train = input
                    y_train = concept
                    x = X_train.clone().detach()
                    y = y_train.clone().detach()
                    output = self.model(x)
                    loss_ = self.criterion(output, y)
                loss += loss_.item()

            if minloss > loss:
                minloss = loss
                index = j
                model_state_dict = copy.deepcopy(self.model.state_dict())
                optimizer_state_dict = copy.deepcopy(self.optimizer.state_dict())

        print(index)
        self.model.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(optimizer_state_dict)

    def calScore(self):
        X_train = self.inputdataP
        x = X_train.clone().detach()
        self.model.eval()
        with torch.no_grad():
            pred = self.model(x)
        return pred.clone().detach()

if __name__ == '__main__':
    input = np.loadtxt('C(246,5520).txt')[:, :10]
    graph = np.random.randint(2, size=(10, 10))
    print(graph)
    input = torch.from_numpy(input).cuda().float()
    SCORE = BIC(10, input, reg_type='GPR')
    print(SCORE.calScore())




