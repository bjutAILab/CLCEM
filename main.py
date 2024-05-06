import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import trainAN
import trainInter
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import trainNet
import causalScore
import numpy as np
import torch


if __name__ == '__main__':

    seeds = [1, 2, 4, 6, 8, 9, 11, 16]
    validIndexs = [[6, 6, 0, 4, 4, 1, 7, 8, 4, 2], [1, 2, 4, 3, 8, 2, 0, 0, 6, 0], [3, 0, 8, 4, 5, 1, 8, 1, 1, 8],
                   [5, 4, 1, 3, 8, 2, 7, 1, 4, 3], [3, 3, 0, 6, 5, 7, 0, 2, 6, 0], [4, 1, 1, 2, 1, 6, 1, 3, 3, 6],
                   [6, 0, 8, 6, 1, 5, 6, 7, 5, 1], [7, 6, 5, 3, 5, 2, 1, 1, 5, 4]]
    wds = [9e-4, 8e-4, 7e-4, 6e-4, 5e-4, 4e-4, 3e-4, 2e-4, 1e-4, 9e-5, 8e-5, 7e-5, 6e-5, 5e-5, 4e-5, 3e-5, 2e-5, 1e-5]
    lrs = [9e-4, 8e-4, 7e-4, 6e-4, 5e-4, 4e-4, 3e-4, 2e-4, 1e-4, 9e-5, 8e-5, 7e-5, 6e-5, 5e-5, 4e-5, 3e-5, 2e-5, 1e-5]
    print("==========100=============")
    for lr in [0.0009]:
        for wd in [3e-05]:
            # Score = causalScore.BICN(lr=lr, wd=wd)
            AVE = 0
            min = 1
            max = 0
            for k in range(0, 10):

                for _ in range(10):
                    # acc = trainInter.run(nbepochs=150, lr=lr, wd=wd, seed=seeds[4], k=k, validIndex=validIndexs[4][k])
                    # acc = trainNet.runOral(nbepochs=150, lr=lr, wd=wd, seed=seeds[4], k=k, validIndex=validIndexs[4][k])
                    acc = trainAN.runCausal(nbepochs=20, lr=lr, wd=wd, seed=seeds[4], k=k, validIndex=validIndexs[4][k],
                                 lamba1=0.1)  # 10 100
                AVE += acc
                if min > acc:
                    min = acc
                if max < acc:
                    max = acc
            print('lr={0}\twd={1}\taverage acc={2}\tmin={3}\tmax={4}'.format(str(lr), str(wd), str(AVE), str(min),
                                                                             str(max)))

