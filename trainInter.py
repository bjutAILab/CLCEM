import torch
import torch.backends.cudnn as cudnn
import numpy as np
import os
import scipy.io as scio
import random
import layers
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
import copy
import test
import matplotlib.pyplot as plt

num_roi = 30
roi_idx = test.ROIALL(num_roi)
data_name = "data.mat"
# roi_idx = [4, 5, 7, 9, 11, 14, 16, 18, 19, 27, 32, 33, 35, 40, 41, 44, 48, 49, 50, 52, 58, 59, 61, 67, 69, 71, 73, 77, 78, 79]

class GoldMSI_LSD_Dataset(torch.utils.data.Dataset):

    def __init__(self, seed, mode="train", k=-1, validIndex=0):
        self.mode = mode
        if os.path.exists(data_name):#这个是直接定好的
            data = scio.loadmat(data_name)
            x, y = data['x'], data['y']
        else:
            (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = fc(seed)
            x = np.vstack((x_train, x_valid, x_test))
            y = np.vstack((y_train, y_valid, y_test))
            scio.savemat(data_name, {'x':x, 'y':y})
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
            x = x_train
            y = y_train
        y = y.reshape((-1,))
        self.X = torch.FloatTensor(x.astype(np.float32))
        self.Y = torch.FloatTensor(y.astype(np.float32))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        sample = [self.X[idx], self.Y[idx]]
        return sample


def fc(seed):
    filePath = os.getcwd()+"/abide1_samples_aal90_1035/"
    # filePath = "../clkf/abide1_samples_aal90_1035/"
    names = os.listdir(filePath)
    names.sort()
    random.seed(seed)
    random.shuffle(names)
    x = np.empty((len(names), 1, num_roi, num_roi))
    y = np.empty((len(names), 1))
    i = 0
    # map = np.zeros((90,1))
    # for id in roi_idx:
    #     map[id, 0] = 1
    # map = np.matmul(map, np.transpose(map, axes=[1, 0]))

    for name in names:
        load_fn = filePath + name
        load_data = scio.loadmat(load_fn)
        x_data = load_data['bold'][:, roi_idx]
        y_data = load_data['label']
        r_data = np.corrcoef(x_data, rowvar=False)  # 计算每一列之间的相关系数（-1，1）
        for a in range(num_roi):
            for b in range(num_roi):
                x[i, 0, a, b] = r_data[a, b]
        y[i, 0] = y_data
        i += 1

    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = \
        (x[:828, :, :, :], y[:828, :]), (x[828:932, :, :, :], y[828:932, :]), (x[932:, :, :, :], y[932:, :])
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


def trainOral(net, trainloader, optimizer, criterion):
    net.train()
    running_loss = 0.0
    categories_loss = 0.0
    causal_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)

        loss = criterion(outputs, targets.long())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / (batch_idx+1)


def testOral(net, testloader, criterion):
    net.eval()
    running_loss = 0.0
    categories_loss = 0.0
    causal_loss = 0.0
    preds = []
    ytrue = []
    for batch_idx, (inputs, targets) in enumerate(testloader):
        with torch.no_grad():
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets.long())
            running_loss += loss.item()
            preds.append(outputs.cpu().numpy())
            ytrue.append(targets.cpu().numpy())

    return np.vstack(preds), np.hstack(ytrue), running_loss / (batch_idx+1)


def run(nbepochs, lr, wd=0, seed=0, k=-1, validIndex=None):
    torch.cuda.empty_cache()
    # Training
    trainset = GoldMSI_LSD_Dataset(mode="train", seed=seed, k=k, validIndex=validIndex)
    net = layers.CNN(trainset.X)
    net = net.cuda(0)
    net = torch.nn.DataParallel(net, device_ids=[0])
    cudnn.benchmark = True

    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.999),
                                  weight_decay=wd)  # weight_decay Decay for L2 regularization
    trainset = GoldMSI_LSD_Dataset(mode="train", seed=seed, k=k, validIndex=validIndex)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=0)
    validset = GoldMSI_LSD_Dataset(mode="valid", seed=seed, k=k, validIndex=validIndex)
    validloader = torch.utils.data.DataLoader(validset, batch_size=256, shuffle=True, num_workers=0)
    testset = GoldMSI_LSD_Dataset(mode="test", seed=seed, k=k, validIndex=validIndex)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=True, num_workers=0)

    minloss = 1e15
    bestacc = 0
    besttn = 0
    bestfp = 0
    bestfn = 0
    besttp = 0
    bestpreds = 0
    besttrue = 0
    model_state_dict = None
    optimizer_state_dict = None
    C0, C1 = [], []

    for epoch in range(nbepochs):
        # seed = random.randint(0, 100)
        loss = trainOral(net, trainloader, optimizer, criterion)
        _, _, loss_valid = testOral(net, validloader, criterion)
        """
        print("Epoch %d" % epoch)
        print("Train Set : loss :" + str(loss) + " categories_loss:")
        print("Validation Set : loss :" + str(loss_valid))
        C0.append(loss)
        C1.append(loss_valid)
        plt.figure()
        plt.plot(C0, label='train')
        plt.plot(C1, label='test')
        plt.title("categories")
        plt.legend()
        plt.savefig("oral.jpg")
        plt.close()
        """
        preds, y_true, loss_test = testOral(net, testloader, criterion)
        preds_ = preds.argmax(1)
        cm = confusion_matrix(y_true.reshape((-1,)), preds_)
        tn = cm[0, 0]
        fp = cm[0, 1]
        fn = cm[1, 0]
        tp = cm[1, 1]
        acc = (tn + tp) / (tn + fp + fn + tp)
        if minloss > loss_valid:
            minloss = loss_valid
            bestacc = acc
            besttn = tn
            bestfp = fp
            bestfn = fn
            besttp = tp
            bestpreds = preds
            besttrue = y_true
            model_state_dict = copy.deepcopy(net.state_dict())
            optimizer_state_dict = copy.deepcopy(optimizer.state_dict())
    fpr, tpr, thresholds = roc_curve(besttrue, bestpreds[:, 1])
    auc = roc_auc_score(besttrue, bestpreds[:, 1])
    print(str(besttn) + "\t" + str(bestfp) + "\t" + str(bestfn) + "\t" + str(besttp) + "\t" + str(bestacc) + "\t" + str(
        auc))
    # net.load_state_dict(model_state_dict)
    #torch.save(net, os.getcwd()+'/'+str(bestacc)+'.pth')
    return bestacc
