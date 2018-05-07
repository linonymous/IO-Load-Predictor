from __future__ import print_function
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt


class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(6, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, input, future = 0):
        outputs = []
        h_t = Variable(torch.zeros(input.size(0), 51).double(), requires_grad=False)
        c_t = Variable(torch.zeros(input.size(0), 51).double(), requires_grad=False)
        h_t2 = Variable(torch.zeros(input.size(0), 51).double(), requires_grad=False)
        c_t2 = Variable(torch.zeros(input.size(0), 51).double(), requires_grad=False)
        h_t, c_t = self.lstm1(input, (h_t, c_t))
        h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
        output = self.linear(h_t2)
        return output


if __name__ == '__main__':
    # set random seed to 0
    temp = Variable(torch.DoubleTensor([[19, 1, 2018, 23, 39, 1]]))
    np.random.seed(0)
    torch.manual_seed(0)

    # load data and make training set
    df = pd.read_csv("sar.csv")
    data = df.as_matrix()
    input = Variable(torch.from_numpy(data[:-100, :-1]), requires_grad=False)
    target = Variable(torch.from_numpy(data[:-100, -1:]), requires_grad=False)
    test_input = Variable(torch.from_numpy(data[-100:, :-1]), requires_grad=False)
    test_target = Variable(torch.from_numpy(data[-100:, -1:]), requires_grad=False)
    seq = Sequence()
    seq.double()
    criterion = nn.MSELoss()
    # LBFGS optimizer
    optimizer = optim.LBFGS(seq.parameters(), lr=0.8)
    # training
    for i in range(10):
        print('STEP: ', i)
        def closure():
            optimizer.zero_grad()
            out = seq(input)
            loss = criterion(out, target)
            print('loss:', loss.data.numpy()[0])
            loss.backward()
            return loss
        optimizer.step(closure)

    pred = seq(temp)
    print(pred.data)