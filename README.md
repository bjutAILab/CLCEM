## Concept-Level Causal Explanation Method for Brain Function Network Classification

This is the official implementation for Concept-Level Causal Explanation Method for Brain Function Network Classification

## Running example
Step 1. train and save model parameters, run the following statement in ```main.py```
```acc = trainAN.runCausal(nbepochs=20, lr=lr, wd=wd, seed=seeds[4], k=k, validIndex=validIndexs[4][k], lamba1=0.1)```

Step 2. selecting the most important brain regions based on the trained model parameters
run ```test.py```

Step 3. train other models based on the obtained important brain region features，run the following statement in ```main.py```
```acc = trainInter.run(nbepochs=150, lr=lr, wd=wd, seed=seeds[4], k=k, validIndex=validIndexs[4][k])```

If you want to compare with the model trained without the proposed brain region features, run the following statement in ```main.py```
```acc = trainNet.runOral(nbepochs=150, lr=lr, wd=wd, seed=seeds[4], k=k, validIndex=validIndexs[4][k])```

## Environment
The code is developed on one NVIDIA RTX 3090 GPU with 24 GB memory and tested in Python 3.9.0 and PyTorch 1.10.2.
