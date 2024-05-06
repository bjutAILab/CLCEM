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
import matplotlib.pyplot as plt


data_name = "data.mat"
class GoldMSI_LSD_Dataset(torch.utils.data.Dataset):

    def __init__(self, seed, mode="train", k=-1, validIndex=0):
        self.mode = mode
        if os.path.exists(data_name):
            data = scio.loadmat(data_name)
            x, y = data['x'], data['y']
        else:
            (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = fc(seed)
            x = np.vstack((x_train, x_valid, x_test))
            y = np.vstack((y_train, y_valid, y_test))
            scio.savemat(data_name, {'x': x, 'y': y})
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


def trainCausal(Score, lamba_score, net, trainloader1, optimizer, criterion1, criterion2=None, trainloader2=None, validloader2=None):
    running_loss = 0.0
    categories_loss = 0.0
    causal_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(trainloader1):
        inputs, targets = inputs.cuda(), targets.cuda()
        net.train()
        optimizer.zero_grad()
        outputs, concept = net(inputs)
        loss1 = criterion1(outputs, targets.long())
        if criterion2 is not None:
            Score.setParameters(inputs, concept)
            yerr = Score.calScore()
            #loss2 = criterion2(concept, yerr)

            loss2 = 0
            for i in range(90):
                tempt = 0
                for j in range(90):
                    if i == j:
                        pass
                    else:
                        tempt += criterion2(concept[:, i], yerr[:, i, j])
                tempt = criterion2(concept[:, i], yerr[:, i, i])/tempt
                loss2 += tempt

            loss = loss1 + lamba_score*loss2
            causal_loss += loss2.item()
            loss.backward()
            optimizer.step()
        else:
            loss = loss1
            causal_loss = 0
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        categories_loss += loss1.item()

    if criterion2 is not None:
        net.eval()
        #Score.update(trainloader2, net)
        Score.train(10, trainloader2, validloader2, net)
        net.train()

    return running_loss / (batch_idx+1), categories_loss / (batch_idx+1), causal_loss / (batch_idx+1)


def testCausal(Score, lamba_score, net, testloader, criterion1, criterion2=None):
    net.eval()
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
            if criterion2 is not None:
                Score.setParameters(inputs, concept)
                yerr = Score.calScore()
                #loss2 = criterion2(concept, yerr)

                loss2 = 0
                for i in range(90):
                    tempt = 0
                    for j in range(90):
                        if i == j:
                            pass
                        else:
                            tempt += criterion2(concept[:, i], yerr[:, i, j])
                    tempt = criterion2(concept[:, i], yerr[:, i, i]) / tempt
                    loss2 += tempt

                causal_loss += loss2.item()
            else:
                loss2 = 0
                causal_loss = 0
            loss = loss1 + lamba_score * loss2
            running_loss += loss.item()
            categories_loss += loss1.item()

            preds.append(outputs.cpu().numpy())
            ytrue.append(targets.cpu().numpy())

    return np.vstack(preds), np.hstack(ytrue), running_loss / (batch_idx+1), categories_loss / (batch_idx+1), causal_loss / (batch_idx+1)


def trainOral(net, trainloader, optimizer, criterion):
    net.train()
    running_loss = 0.0
    categories_loss = 0.0
    causal_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs, _ = net(inputs)#_就是7056-》256的特征信息

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
            outputs, _ = net(inputs)
            loss = criterion(outputs, targets.long())
            running_loss += loss.item()
            preds.append(outputs.cpu().numpy())
            ytrue.append(targets.cpu().numpy())

    return np.vstack(preds), np.hstack(ytrue), running_loss / (batch_idx+1)


def runOral(nbepochs, lr, wd=0, seed=0, k=-1, validIndex=None):
    torch.cuda.empty_cache()
    # Training
    trainset = GoldMSI_LSD_Dataset(mode="train", seed=seed, k=k, validIndex=validIndex)#分割数据集
    net = layers.CNNScore(trainset.X)
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
    net.load_state_dict(model_state_dict)
    #torch.save(net, os.getcwd()+'/'+str(bestacc)+'.pth')
    return bestacc


def runCausal(Score, nbepochs1, nbepochs2, lr, wd=0, lamba_score=0.0001, seed=0, k=-1, validIndex=None):
    torch.cuda.empty_cache()
    # Training

    trainset = GoldMSI_LSD_Dataset(mode="train", seed=seed, k=k, validIndex=validIndex)
    net = layers.BrainNetCNNScore(trainset.X)
    net = net.cuda(0)
    net = torch.nn.DataParallel(net, device_ids=[0])
    cudnn.benchmark = True

    # class torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    # criterion = torch.nn.MSELoss()
    criterion1 = torch.nn.NLLLoss()
    criterion2 = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.999),
                                  weight_decay=wd)  # weight_decay Decay for L2 regularization

    trainset1 = GoldMSI_LSD_Dataset(mode="train", seed=seed, k=k, validIndex=validIndex)
    trainset2 = GoldMSI_LSD_Dataset(mode="train", seed=seed, k=k, validIndex=validIndex)
    trainloader1 = torch.utils.data.DataLoader(trainset1, batch_size=256, shuffle=True, num_workers=0)
    trainloader2 = torch.utils.data.DataLoader(trainset2, batch_size=256, shuffle=True, num_workers=0)
    validset1 = GoldMSI_LSD_Dataset(mode="valid", seed=seed, k=k, validIndex=validIndex)
    validset2 = GoldMSI_LSD_Dataset(mode="valid", seed=seed, k=k, validIndex=validIndex)
    validloader1 = torch.utils.data.DataLoader(validset1, batch_size=256, shuffle=True, num_workers=0)
    validloader2 = torch.utils.data.DataLoader(validset2, batch_size=256, shuffle=True, num_workers=0)
    testset = GoldMSI_LSD_Dataset(mode="test", seed=seed, k=k, validIndex=validIndex)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=True, num_workers=0)
    """
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
    index = -1
    for epoch in range(nbepochs1): 
        trainCausal(Score, lamba_score, net, trainloader1, optimizer, criterion1)
        _, _, loss_valid, categories_loss_valid, causal_loss_valid = testCausal(Score, lamba_score, net, validloader1,
                                                                                criterion1)
        preds, y_true, loss_test, ategories_loss_test, causal_loss_test = testCausal(Score, lamba_score, net, testloader,
                                                                                     criterion1)
        preds_ = preds.argmax(1)
        cm = confusion_matrix(y_true.reshape((-1,)), preds_)
        tn = cm[0, 0]
        fp = cm[0, 1]
        fn = cm[1, 0]
        tp = cm[1, 1]
        acc = (tn + tp) / (tn + fp + fn + tp)

        if minloss > loss_valid:
            index = epoch
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
    net.load_state_dict(model_state_dict)
    optimizer.load_state_dict(optimizer_state_dict)
    auc = roc_auc_score(besttrue, bestpreds[:, 1])
    print(str(index) + '\t' + str(besttn) + "\t" + str(bestfp) + "\t" + str(bestfn) + "\t" + str(besttp) + "\t"
         + str(bestacc) + "\t" + str(auc))
    """
    net = torch.load(os.getcwd()+'/0.7211538461538461.pth')
    net.eval()
    Score.train(300, trainloader2, validloader2, net)
    #Score.optimizer = torch.optim.AdamW(Score.model.parameters(), lr=lr, weight_decay=wd)

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
    for epoch in range(nbepochs2):
        loss, categories_loss, causal_loss = trainCausal(Score, lamba_score, net, trainloader1, optimizer, criterion1,
                                                         criterion2, trainloader2, validloader2)
        _, _, loss_valid, categories_loss_valid, causal_loss_valid = testCausal(Score, lamba_score, net, validloader1,
                                                                               criterion1, criterion2)

        print("Epoch %d" % epoch)
        print("Train Set : loss :" + str(loss) + " categories_loss:" + str(categories_loss) + " causal_loss:" + str(
            causal_loss))
        print("Validation Set : loss :" + str(loss_valid) + " categories_loss:" + str(
            categories_loss_valid) + " causal_loss:" + str(causal_loss_valid))

        preds, y_true, loss_test, ategories_loss_test, causal_loss_test = testCausal(Score, lamba_score, net,
                                                                                     testloader, criterion1, criterion2)
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
            optimizer_state_dict = copy.deepcopy(optimizer.state_dict())
    auc = roc_auc_score(besttrue, bestpreds[:, 1])
    print(str(besttn) + "\t" + str(bestfp) + "\t" + str(bestfn) + "\t" + str(besttp) + "\t" + str(bestacc) + "\t" + str(
        auc))
    print(str(index) + '\t' + str(categories_l) + '\t' + str(causal_l))
    torch.save(net, os.getcwd() + '/last_10.pth')
    net.load_state_dict(model_state_dict)
    optimizer.load_state_dict(optimizer_state_dict)
    torch.save(net, os.getcwd() + '/best_10.pth')
    return bestacc


def runSplit(Score, lamba_score, nbepochs, lr, wd=0, seed=0):
    # lr = 7e-4, wd = 3e-5
    print("==================="+str(seed)+"===================")
    ACC = 0.0
    for k in range(10):
        bestValid = 0
        bestAcc = 0
        for validIndex in range(9):
            torch.cuda.empty_cache()
            # Training
            trainset = GoldMSI_LSD_Dataset(mode="train", seed=seed, k=k, validIndex=validIndex)
            net = layers.BrainNetCNNEWScore(trainset.X)
            net = net.cuda(0)
            net = torch.nn.DataParallel(net, device_ids=[0])
            cudnn.benchmark = True
            criterion = torch.nn.NLLLoss()
            optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.999),
                                          weight_decay=wd)  # weight_decay Decay for L2 regularization
            trainset = GoldMSI_LSD_Dataset(mode="train", seed=seed, k=k, validIndex=validIndex)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True, num_workers=1)
            validset = GoldMSI_LSD_Dataset(mode="valid", seed=seed, k=k, validIndex=validIndex)
            validloader = torch.utils.data.DataLoader(validset, batch_size=512, shuffle=True, num_workers=1)
            testset = GoldMSI_LSD_Dataset(mode="test", seed=seed, k=k, validIndex=validIndex)
            testloader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=True, num_workers=1)

            minloss = 10000
            bestacc = 0
            for epoch in range(nbepochs):
                trainCausal(Score, lamba_score, net, trainloader, optimizer, criterion)
                _, _, loss_valid, categories_loss_valid, causal_loss_valid = testCausal(Score, lamba_score, net, validloader,
                                                                                  criterion)
                preds, y_true, loss_test, ategories_loss_test, causal_loss_test = testCausal(Score, lamba_score, net,
                                                                                       testloader, criterion)
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
            if bestacc > bestAcc:
                bestAcc = bestacc
                bestValid = validIndex
        print("k="+str(k)+"\tbestValid="+str(bestValid)+"\tbestAcc="+str(bestAcc))
        ACC += bestAcc
    print("Average acc="+str(ACC))


def runSaveC(Score, nbepochs1, nbepochs2, lr, wd=0, lamba_score=0.0001, seed=0, k=-1, validIndex=None):
    torch.cuda.empty_cache()
    PATH = os.getcwd() + "/save_model/BCNNEWS_"
    # Training

    trainset = GoldMSI_LSD_Dataset(mode="train", seed=seed, k=k, validIndex=validIndex)
    net = layers.BrainNetCNNScore(trainset.X)
    net = net.cuda(0)
    net = torch.nn.DataParallel(net, device_ids=[0])
    cudnn.benchmark = True

    # class torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    # criterion = torch.nn.MSELoss()
    criterion1 = torch.nn.NLLLoss()
    criterion2 = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.999),
                                  weight_decay=wd)  # weight_decay Decay for L2 regularization

    trainset = GoldMSI_LSD_Dataset(mode="train", seed=seed, k=k, validIndex=validIndex)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=1)
    validset = GoldMSI_LSD_Dataset(mode="valid", seed=seed, k=k, validIndex=validIndex)
    validloader = torch.utils.data.DataLoader(validset, batch_size=256, shuffle=True, num_workers=1)
    testset = GoldMSI_LSD_Dataset(mode="test", seed=seed, k=k, validIndex=validIndex)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=True, num_workers=1)

    minloss = 1e15
    model_state_dict = None
    optimizer_state_dict = None
    index = -1
    for epoch in range(nbepochs1):
        trainCausal(Score, lamba_score, net, trainloader, optimizer, criterion1)
        _, _, loss_valid, categories_loss_valid, causal_loss_valid = testCausal(Score, lamba_score, net, validloader,
                                                                                criterion1)
        preds, y_true, loss_test, ategories_loss_test, causal_loss_test = testCausal(Score, lamba_score, net,
                                                                                     testloader,
                                                                                     criterion1, criterion2)

        if minloss > loss_valid:
            minloss = loss_valid
            model_state_dict = copy.deepcopy(net.state_dict())
            optimizer_state_dict = copy.deepcopy(optimizer.state_dict())
    net.load_state_dict(model_state_dict)
    optimizer.load_state_dict(optimizer_state_dict)

    minloss = 1e15
    bestacc = 0
    mincloss = 1e15
    for epoch in range(nbepochs2):
        loss, categories_loss, causal_loss = trainCausal(Score, lamba_score, net, trainloader, optimizer, criterion1,
                                                         criterion2)
        _, _, loss_valid, categories_loss_valid, causal_loss_valid = testCausal(Score, lamba_score, net, validloader,
                                                                            criterion1, criterion2)
        preds, y_true, loss_test, ategories_loss_test, causal_loss_test = testCausal(Score, lamba_score, net, testloader,
                                                                                     criterion1, criterion2)
        preds_ = preds.argmax(1)
        cm = confusion_matrix(y_true.reshape((-1,)), preds_)
        tn = cm[0, 0]
        fp = cm[0, 1]
        fn = cm[1, 0]
        tp = cm[1, 1]
        acc = (tn + tp) / (tn + fp + fn + tp)
        if minloss > loss_valid:
            minloss = loss_valid
            mincloss = causal_loss
            model_state_dict = copy.deepcopy(net.state_dict())
            optimizer_state_dict = copy.deepcopy(optimizer.state_dict())
            bestepoch = epoch
            bestacc=acc
    torch.save({
        'epoch': bestepoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict,
        'loss': minloss}, PATH+str(k)+'_'+str(bestacc)+'_'+str(mincloss)+'.pth')

