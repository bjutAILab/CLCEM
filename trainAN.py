import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np
import os
import scipy.io as scio
import random
import layers
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
import copy
import matplotlib.pyplot as plt

model_state_dicts = []
optimizer_state_dicts = []
cnum = 256


def autoBalance(loss1, loss2):
    a = abs(loss1)
    b = abs(loss2)

    mula = 1
    while a < 1:
        a *= 10
        mula *= 10
    while a > 10:
        a /= 10
        mula /= 10

    mulb = 1
    while b < 1:
        b *= 10
        mulb *= 10
    while b > 10:
        b /= 10
        mulb /= 10

    return mulb / mula


class GlobalAvgPool2d(torch.nn.Module):

    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])


"""
class BaseNet(torch.nn.Module):   # (-1, 1, 90, 1)
    def __init__(self):
        super(BaseNet, self).__init__()
        self.cn1 = torch.nn.Conv2d(1, 16, kernel_size=(3, 1), padding=1, stride=2)
        self.cn2 = torch.nn.Conv2d(16, 32, kernel_size=(3, 1), padding=1, stride=2)
        self.cn3 = torch.nn.Conv2d(32, 64, kernel_size=(3, 1), padding=1, stride=2)
        self.global_avg_pool = GlobalAvgPool2d()
        self.fc = torch.nn.Linear(64, 1)

    def forward(self, x):
        out = F.leaky_relu(self.cn1(x.reshape((-1, 1, 90, 1))), negative_slope=0.33)
        out = F.leaky_relu(self.cn2(out), negative_slope=0.33)
        out = F.leaky_relu(self.cn3(out), negative_slope=0.33)
        out = self.global_avg_pool(out)
        out = out.view(out.size(0), -1)
        out = F.leaky_relu(self.fc(out), negative_slope=0.33)
        return out
"""


class BaseNet(torch.nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        self.fc1 = torch.nn.Linear(90, 32)
        # self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 1)

    def forward(self, x):
        out = F.leaky_relu(self.fc1(x), negative_slope=0.33)
        # out = F.leaky_relu(self.fc2(out), negative_slope=0.33)
        out = F.leaky_relu(self.fc3(out), negative_slope=0.33)
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
            # col[i, i] = 1
            cols_TrueFalse = col > 0.5
            X_train = x[:, 0, cols_TrueFalse]
            outs.append(self.models[i](X_train)) #取每一个ROI，用BaseNet（90-32-1）
        return torch.cat(outs, dim=1)#每个ROI变成一个值，由其他ROIs得到


class GoldMSI_LSD_Dataset(torch.utils.data.Dataset):

    def __init__(self, seed, mode="train", k=-1, validIndex=0):
        self.mode = mode
        (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = fc(seed)
        if k == -1:
            pass
        else:
            x = np.vstack((x_train, x_valid, x_test))
            y = np.vstack((y_train, y_valid, y_test))
            (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = kfold(x, y, k, validIndex)
        if self.mode == "train":
            x = x_train
            y = y_train
        elif self.mode == "test":
            x = x_test
            y = y_test
        elif mode == "valid":
            x = x_valid
            y = y_valid
        else:
            x = x
            y = y
        y = y.reshape((-1,))
        self.X = torch.FloatTensor(x.astype(np.float32))
        self.Y = torch.FloatTensor(y.astype(np.float32))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        sample = [self.X[idx], self.Y[idx]]
        return sample


def fc(seed):
    filePath = os.getcwd() + "/abide1_samples_aal90_1035/"
    names = os.listdir(filePath)
    names.sort()
    random.seed(seed)
    random.shuffle(names)
    x = np.empty((len(names), 1, 90, 90))
    y = np.empty((len(names), 1))
    i = 0
    for name in names:
        load_fn = filePath + name
        load_data = scio.loadmat(load_fn)
        x_data = load_data['bold']
        y_data = load_data['label']
        r_data = np.corrcoef(x_data, rowvar=False)
        for a in range(90):
            for b in range(90):
                x[i, 0, a, b] = r_data[a, b]
        y[i, 0] = y_data
        i += 1

    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = \
        (x[:828, :, :, :], y[:828, :]), (x[828:932, :, :, :], y[828:932, :]), (x[932:, :, :, :], y[932:, :])
    # (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = \
    #     (x[:40, :, :, :], y[:40, :]), (x[40:45, :, :, :], y[40:45, :]), (x[45:, :, :, :], y[45:, :])
    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)


def kfold(x, y, k, validIndex):
    # 10
    xlist = np.array_split(x, 10, axis=0)
    ylist = np.array_split(y, 10, axis=0)
    (x_test, y_test) = (xlist[k], ylist[k])
    del xlist[k]
    del ylist[k]
    (x_valid, y_valid) = (xlist[validIndex], ylist[validIndex])
    del xlist[validIndex]
    del ylist[validIndex]
    (x_train, y_train) = (np.vstack(xlist), np.vstack(ylist))
    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)


def trainCausal(score, lamba1, net, trainloader, optimizer1, optimizer2, criterion1, criterion2):
    running_loss = 0.0
    categories_loss = 0.0
    causal_loss = 0.0
    global model_state_dicts, optimizer_state_dicts
    for step in range(2):
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            # 1-5，因果估计器Score
            net.eval()
            with torch.no_grad():
                noise = np.random.normal(0, 8, size=inputs.shape).astype(np.float32)
                for a in range(89):
                    noise[:, 0, a, a] = 0
                    for b in range(a + 1, 90):
                        tempt = (noise[:, 0, a, b] + noise[:, 0, b, a]) / 2
                        noise[:, 0, a, b] = tempt
                        noise[:, 0, b, a] = tempt
                """
                noise2 = np.random.normal(0, 1, size=inputs.shape).astype(np.float32)
                for a in range(89):
                    noise2[:, 0, a, a] = 0
                    for b in range(a + 1, 90):
                        tempt = (noise2[:, 0, a, b] + noise2[:, 0, b, a]) / 2
                        noise2[:, 0, a, b] = tempt
                        noise2[:, 0, b, a] = tempt
                """
                # inputs_ = torch.multiply(inputs, torch.from_numpy(noise2 < 0)) + np.multiply(noise, (noise2 > 0))
                # inputs_ = inputs + np.multiply(noise, (noise2 > 0))
                inputs_ = inputs + noise
                # inputs_ = torch.from_numpy(noise)
                # inputs_ = inputs
                inputs_ = inputs_.cuda()
                _, concept = net(inputs_)
                X_train = inputs_
                x = X_train.clone().detach()
            for i in range(cnum):

                y = torch.cat([concept[:, i].clone().detach().reshape((-1, 1)) for k in range(90)], dim=1)
                score.load_state_dict(model_state_dicts[i])
                optimizer2.load_state_dict(optimizer_state_dicts[i])
                score.train()
                optimizer2.zero_grad()
                output = score(x)
                loss_ = criterion2(output, y)
                loss_.backward()
                optimizer2.step()
                model_state_dicts[i] = copy.deepcopy(score.state_dict())
                optimizer_state_dicts[i] = copy.deepcopy(optimizer2.state_dict())

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # G
        net.train()
        optimizer1.zero_grad()
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs, concept = net(inputs)
        loss1 = criterion1(outputs, targets.long())

        RSS_ls = []
        for i in range(cnum):
            score.load_state_dict(model_state_dicts[i])
            optimizer2.load_state_dict(optimizer_state_dicts[i])
            score.eval()
            with torch.no_grad():
                pred = score(inputs)
                RSS_ls.append(torch.reshape(pred.clone().detach(), (-1, 1, 90)))
        yerr = torch.cat(RSS_ls, 1)
        """
        loss2 = 0
        for i in range(cnum):
            tempt = []
            for j in range(90):
                tempt.append(criterion2(concept[:, i], yerr[:, i, j]))
            tempt.sort()
            loss2 += tempt[0] - tempt[1]
        """
        loss2 = 0
        for i in range(cnum):
            tempt = []
            for j in range(90):
                tempt.append(criterion2(concept[:, i], yerr[:, i, j]))
            tempt.sort()
            loss2 += (tempt[0]) / tempt[1]

        loss2 /= cnum
        loss = loss1 + lamba1 * loss2
        loss.backward()
        optimizer1.step()
        causal_loss += loss2.item()
        running_loss += loss.item()
        categories_loss += loss1.item()

    return running_loss / (batch_idx + 1), categories_loss / (batch_idx + 1), causal_loss / (batch_idx + 1)


def testCausal(score, lamba1, net, testloader, criterion1, criterion2):
    net.eval()
    global model_state_dicts, optimizer_state_dicts
    running_loss = 0.0
    categories_loss = 0.0
    causal_loss = 0.0
    preds = []
    ytrue = []
    for batch_idx, (inputs, targets) in enumerate(testloader):
        with torch.no_grad():
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs, concept = net(inputs)
            loss1 = criterion1(outputs, targets.long())
            # print(concept)

            RSS_ls = []
            for i in range(cnum):
                score.load_state_dict(model_state_dicts[i])
                score.eval()
                with torch.no_grad():
                    pred = score(inputs)
                    RSS_ls.append(torch.reshape(pred.clone().detach(), (-1, 1, 90)))
            yerr = torch.cat(RSS_ls, 1)

            loss2 = 0
            for i in range(cnum):
                tempt = []
                for j in range(90):
                    tempt.append(criterion2(concept[:, i], yerr[:, i, j]))
                tempt.sort()
                loss2 += (tempt[0]) / tempt[1]
            loss2 /= cnum

            causal_loss += loss2.item()
            loss = loss1 + lamba1 * loss2
            running_loss += loss.item()
            categories_loss += loss1.item()

            preds.append(outputs.cpu().numpy())
            ytrue.append(targets.cpu().numpy())

    return np.vstack(preds), np.hstack(ytrue), running_loss / (batch_idx + 1), categories_loss / (
                batch_idx + 1), causal_loss / (batch_idx + 1)


def runCausal(nbepochs, lr, wd=0, lamba1=0.0001, seed=0, k=-1, validIndex=None):
    global model_state_dicts, optimizer_state_dicts

    torch.cuda.empty_cache()
    # Training

    trainset = GoldMSI_LSD_Dataset(mode="train", seed=seed, k=k, validIndex=validIndex)
    net = layers.CNNScore(trainset.X).cuda(0)
    score = NolinerSEM().cuda(0)
    net = torch.nn.DataParallel(net, device_ids=[0])
    score = torch.nn.DataParallel(score, device_ids=[0])
    cudnn.benchmark = True

    criterion1 = torch.nn.NLLLoss()
    criterion2 = torch.nn.MSELoss()
    optimizer1 = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=wd)
    # optimizer1 = torch.optim.RMSprop(net.parameters(), lr=lr, weight_decay=wd)
    # optimizer2 = torch.optim.RMSprop(score.parameters(), lr=2*lr/3, weight_decay=2*wd/3)
    optimizer2 = torch.optim.AdamW(score.parameters(), lr=2 * lr / 3, betas=(0.9, 0.999), weight_decay=2 * wd / 3)

    model_state_dicts.clear()
    optimizer_state_dicts.clear()
    trainC0 = []
    trainC1 = []
    testC0 = []
    testC1 = []
    for i in range(cnum):
        model_state_dicts.append(copy.deepcopy(score.state_dict()))
        optimizer_state_dicts.append(copy.deepcopy(optimizer2.state_dict()))

    trainset = GoldMSI_LSD_Dataset(mode="train", seed=seed, k=k, validIndex=validIndex)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=0)
    validset = GoldMSI_LSD_Dataset(mode="valid", seed=seed, k=k, validIndex=validIndex)
    validloader = torch.utils.data.DataLoader(validset, batch_size=256, shuffle=True, num_workers=0)
    testset = GoldMSI_LSD_Dataset(mode="test", seed=seed, k=k, validIndex=validIndex)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=True, num_workers=0)

    minloss = 1e15
    categories_l = 0
    causal_l = 0
    bestacc = 0
    besttn = 0
    bestfp = 0
    bestfn = 0
    besttp = 0
    bestpreds = 0
    besttrue = 0
    index = -1
    model_state_dict = None
    optimizer_state_dict = None
    best_model_state_dicts = []
    for epoch in range(nbepochs):
        loss, categories_loss, causal_loss = trainCausal(score, lamba1, net, trainloader, optimizer1, optimizer2,
                                                         criterion1, criterion2)
        trainC0.append(categories_loss)
        trainC1.append(causal_loss)
        _, _, loss_valid, categories_loss_valid, causal_loss_valid = testCausal(score, lamba1, net, validloader,
                                                                                criterion1, criterion2)
        """
        testC0.append(categories_loss_valid)
        testC1.append(causal_loss_valid)
        
        plt.figure()
        ax = plt.subplot(2, 1, 1)
        plt.plot(trainC0, label='train')
        plt.plot(testC0, label='test')
        ax.set_title("categories")
        plt.legend()
        ax = plt.subplot(2, 1, 2)
        plt.plot(trainC1, label='train')
        plt.plot(testC1, label='test')
        ax.set_title("causal")
        plt.legend()
        plt.savefig("BCNNEW256_0.1_2_n2_linear.jpg")
        plt.close()
        """

        print("Epoch %d" % epoch)
        print("Train Set : loss :" + str(loss) + " categories_loss:" + str(categories_loss) + " causal_loss:" + str(
            causal_loss))
        print("Validation Set : loss :" + str(loss_valid) + " categories_loss:" + str(
            categories_loss_valid) + " causal_loss:" + str(causal_loss_valid))

        preds, y_true, loss_test, ategories_loss_test, causal_loss_test = testCausal(score, lamba1, net, testloader,
                                                                                     criterion1, criterion2)
        preds_ = preds.argmax(1)
        cm = confusion_matrix(y_true.reshape((-1,)), preds_)
        tn = cm[0, 0]
        fp = cm[0, 1]
        fn = cm[1, 0]
        tp = cm[1, 1]
        acc = (tn + tp) / (tn + fp + fn + tp)

        if minloss > loss_valid:
            index = epoch
            categories_l = categories_loss
            causal_l = causal_loss
            minloss = loss_valid
            bestacc = acc
            besttn = tn
            bestfp = fp
            bestfn = fn
            besttp = tp
            bestpreds = preds
            besttrue = y_true
            model_state_dict = copy.deepcopy(net.state_dict())
            optimizer_state_dict = copy.deepcopy(optimizer1.state_dict())
            best_model_state_dicts.clear()
            for i in range(cnum):
                best_model_state_dicts.append(copy.deepcopy(model_state_dicts[i]))
    auc = roc_auc_score(besttrue, bestpreds[:, 1])
    print(str(besttn) + "\t" + str(bestfp) + "\t" + str(bestfn) + "\t" + str(besttp) + "\t" + str(bestacc) + "\t" + str(
        auc))
    print(str(index) + '\t' + str(categories_l) + '\t' + str(causal_l))
    net.load_state_dict(model_state_dict)
    optimizer1.load_state_dict(optimizer_state_dict)
    save_path = os.getcwd() + '/save_model/CNN256_0.1_2_8_linear10/' + str(k)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(net, os.getcwd() + '/save_model/CNN256_0.1_2_8_linear10/' + str(k) + '/k=' + str(k) + 'acc=' + str(
        bestacc) + 'categories_l='
               + str(categories_l) + 'causal_l=' + str(causal_l) + 'linear.pth')

    save_path1 = os.getcwd() + '/save_model/BCNNEW256_0.1_2_8_linear10/' + str(k)
    if not os.path.exists(save_path1):
        os.makedirs(save_path1)
    for i in range(cnum):
        torch.save(best_model_state_dicts[i],
                   os.getcwd() + '/save_model/BCNNEW256_0.1_2_8_linear10/' + str(k) + '/k=' + str(k) + 'acc=' + str(
                       bestacc) + 'categories_l='
                   + str(categories_l) + 'causal_l=' + str(causal_l) + 'linear' + str(i) + '.pth')

    return bestacc
