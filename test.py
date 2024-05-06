import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import torch
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from trainAN import GoldMSI_LSD_Dataset, NolinerSEM
import scipy.io as scio
import torch.nn.functional as F
import copy
import heapq


def performance(path, k):
    net = torch.load(path)
    net.eval()
    testset = GoldMSI_LSD_Dataset(mode="test", seed=8, k=k, validIndex=3)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=0)
    preds = []
    ytrue = []
    for batch_idx, (inputs, targets) in enumerate(testloader):
        with torch.no_grad():
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs, concept = net(inputs)
            preds.append(outputs.cpu().numpy())
            ytrue.append(targets.cpu().numpy())
    preds, ytrue = np.vstack(preds), np.hstack(ytrue)
    preds_ = preds.argmax(1)
    cm = confusion_matrix(ytrue.reshape((-1,)), preds_)
    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tp = cm[1, 1]
    acc = (tn + tp) / (tn + fp + fn + tp)
    auc = roc_auc_score(ytrue, preds[:, 1])
    print(str(tn) + "\t" + str(fp) + "\t" + str(fn) + "\t" + str(tp) + "\t" + str(acc) + "\t" + str(auc))
    return acc


def Performance():
    k=6
    filePath = os.getcwd() + "/save_model/CNN256_0.1_2_8_linear10/0/"
    names = os.listdir(filePath)
    names.sort()
    l = len(names) // 257
    # l = len(names)
    for i in range(l):
        load_fn = filePath + names[i*257]
        # load_fn = filePath + names[i]
        print(names[i*257])
        # print(names[i])
        performance(load_fn, k)


def ROIscore():
    criterion2 = torch.nn.MSELoss()
    seed = 8
    k = 0
    validIndex = 3
    trainset = GoldMSI_LSD_Dataset(mode="all", seed=seed, k=k, validIndex=validIndex)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=False, num_workers=1)
    filePath = os.getcwd() + "/save_model/BCNNEW256_0.1_2_8_linear10/0/"
    filePath1 = os.getcwd() + "/save_model/CNN256_0.1_2_8_linear10/0/"

    names = os.listdir(filePath)
    names.sort()
    l = len(names)//257
    # l = len(names)

    net1 = NolinerSEM().cuda(0)
    net1 = torch.nn.DataParallel(net1, device_ids=[0])
    for i in range(l):
        # net0 = torch.load(filePath1 + names[i*257])
        net0 = torch.load(filePath1 + names[i * 257][:-5] + '.pth')

        net0.eval()

        score = np.zeros((256, 90))
        yerr = []
        cs = []
        ytrue = []
        for batch_idx, (input, targets) in enumerate(trainloader):
            ytrue.append(targets.cpu().numpy())
            with torch.no_grad():
                input = input.cuda()

                _, concept = net0(input)
                cs.append(concept)
                RSS_ls = []
                for j in range(256):
                    net1.load_state_dict(torch.load(filePath + names[i*257][:-5]+str(j)+'.pth'))
                    # net1.load_state_dict(torch.load(filePath + names[i]))
                    net1.eval()
                    pred = net1(input)
                    RSS_ls.append(torch.reshape(pred.clone().detach(), (-1, 1, 90)))
                yerr.append(torch.cat(RSS_ls, 1))
        yerr = torch.cat(yerr, 0)
        cs = torch.cat(cs, 0)
        for j in range(256):
            for k in range(90):
                # score[j, k] = 1035*np.log(criterion2(cs[:, j], yerr[:, j, k]).item()+1)
                score[j, k] = 1035 * np.log(criterion2(cs[:, j], yerr[:, j, k]).item() + 1)


        ytrue = np.hstack(ytrue)
        pred = torch.zeros((ytrue.shape[0], 2)).cuda(0)
        # weight = torch.transpose(net0.state_dict()['module.dense2.weight'], 0, 1)  # (2,256)^T
        weight = torch.transpose(net0.state_dict()['module.fc2.weight'], 0, 1)
        # bias = net0.state_dict()['module.dense2.bias']  # (2)
        # contribution1 = torch.mm(cs, weight)+bias

        contribution = np.zeros((ytrue.shape[0], 2, 256))
        for j in range(pred.shape[0]):
            # pred[j, 0] = torch.mm(cs[j, :].reshape((1, -1)), weight[:, 0].reshape((-1, 1)))[0, 0]
            # pred[j, 1] = torch.mm(cs[j, :].reshape((1, -1)), weight[:, 1].reshape((-1, 1)))[0, 0]
            contribution[j, 0, :] = (cs[j, :].reshape((-1,)) * weight[:, 0].reshape((-1,))).cpu().numpy()
            contribution[j, 1, :] = (cs[j, :].reshape((-1,)) * weight[:, 1].reshape((-1,))).cpu().numpy()

        # pred[:, 0] += bias[0]
        # pred[:, 1] += bias[1]
        # pred = F.log_softmax(pred, dim=1).cpu().numpy()
        # pred_ = pred.argmax(1)
        # cm = confusion_matrix(ytrue.reshape((-1,)), pred_)
        # tn = cm[0, 0]
        # fp = cm[0, 1]
        # fn = cm[1, 0]
        # tp = cm[1, 1]
        # acc = (tn + tp) / (tn + fp + fn + tp)
        # auc = roc_auc_score(ytrue, pred[:, 1])
        # print(acc)
        # print(auc)
        # print(names[i*257][:-4])

        scio.savemat(os.getcwd() + '/save_score/'+names[i*257][:-5]+'.mat', {'score': score, 'contribution': contribution})


def ROI30(n):
    # data = scio.loadmat(os.getcwd() + '/save_score/'+
    #                     'k=0acc=0.6634615384615384categories_l=0.41769926249980927causal_l=0.5351269245147705linear.mat')
    data = scio.loadmat(os.getcwd() + '/save_score/'+'k=0acc=0.7categories_l=0.8618468880653382causal_l=0.8236697912216187linear.mat')

    score = data['score']
    contri = data['contribution']
    ROI_score = np.zeros((90,))
    for i in range(256):
        t = score[i, :]
        total = np.sum(t)
        roi_idx = np.argmin(t)
        t = np.sort(t)
        significance = t[0]/t[1]
        ROI_score[roi_idx] += np.sum(np.abs(contri[:, 0, i] - contri[:, 1, i]))
    #print(ROI_score)

    return np.argsort(ROI_score)[-1*n:]


def ROIALL(n):
    filePath = os.getcwd() + '/save_score/'
    names = os.listdir(filePath)
    total = 0.0
    for name in names:
        total += (float(name[7:11])-0.6)
    SCORE = np.zeros((90,))
    for name in names:
        data = scio.loadmat(filePath+name)
        score = data['score']
        contri = data['contribution']
        ROI_score = np.zeros((90,))
        for i in range(256):
            t = score[i, :]
            total = np.sum(t)
            roi_idx = np.argmin(t)
            t = np.sort(t)
            significance = t[0]/t[1]
            ROI_score[roi_idx] += np.sum(np.abs(contri[:, 0, i] - contri[:, 1, i]))
        ROI_score = (ROI_score - np.min(ROI_score))/(np.max(ROI_score) - np.min(ROI_score))
        # SCORE += ROI_score * ((float(name[7:11])-0.6)/total)
        SCORE += ROI_score
    SCORE = (SCORE - np.min(SCORE)) / (np.max(SCORE) - np.min(SCORE))
    for i in range(90):
        print(SCORE[i])

    return np.argsort(SCORE)[-1*n:]


if __name__ == '__main__':
    Performance()
    # caculateBICwithAN()
    ROIscore()
    # score = scio.loadmat(os.getcwd() + '/save_score/k=0acc=0.6730769230769231categories_l=0.38114482164382935causal_l=0.5737479776144028linear.mat')['data']
    score = scio.loadmat(os.getcwd() + '/save_score/' + 'k=0acc=0.7categories_l=0.8618468880653382causal_l=0.8236697912216187linear.mat')['score']
    l=[]
    for i in range(256):
        t = score[i, :]
        t = np.sort(t)
        l.append(t[0]/t[1])
        # print(t[0]/t[1])
    print(ROIALL(30))



