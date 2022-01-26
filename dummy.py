import timeit
from datetime import datetime
import socket
import os
import glob
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
#from parallel import DataParallelModel, DataParallelCriterion

from dummy_model import DUMMY
from dataloader import dummydata

""" Git Test
    Git Test 2
    Git Test 3
    Git Test 4
    Git Test 5
"""

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

if torch.cuda.device_count() > 1:
    print("{} GPUs is on use".format(torch.cuda.device_count()))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

def train_model(resume_nEpoch=0, nEpoch=100, lr=1e-3):

    resume_epoch = resume_nEpoch
    num_epoch = nEpoch

    net = DUMMY()
    net = torch.nn.DataParallel(net, device_ids=[0, 1, 2, 3])
    train_params = net.parameters()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(train_params, lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    print("Total params: {:.2f} M".format(sum(p.numel() for p in net.parameters()) / 1000000.0))

    net.to(device)
    criterion.to(device)

    train_dataset = DataLoader(dummydata(), batch_size=1600, shuffle=True, num_workers=4)
    train_size = len(train_dataset.dataset)

    for epoch in range(resume_epoch, num_epoch):
        
        start_time = timeit.default_timer()

        net.train()

        for inputs, labels in tqdm(train_dataset):
            
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)

            probs = nn.Softmax(dim=1)(outputs)
            preds = torch.max(probs, 1)[1]
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

        scheduler.step()



if __name__ == '__main__':
    train_model()