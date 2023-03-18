import sys
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import HTML
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

#print(sys.path)
np.random.seed(2023)
torch.manual_seed(2023)

plt.rcParams["animation.html"] = "jshtml"
plt.rcParams['figure.dpi'] = 150  
plt.ioff()

def sigmoid(X):
    # ===============
    '''完成logistic函数'''
    #todo
    # ===============

class LogisticRegression(nn.Module):
    def __init__(self, num_inputs):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(num_inputs, 1, dtype=torch.float64)

        nn.init.normal_(self.linear.weight, mean=0, std=0.01)
        nn.init.constant_(self.linear.bias, val=0.01)


    def forward(self, x):
        y = self.linear(x.view(x.shape[0], -1))
        y = sigmoid(y)
        return y

    def automatic_update(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def manual_update(self, X, y, y_hat):
        '''
        x is the input feature;
        y is the ground truth label;
        y_hat is the predicted label.
        Please update self.linear.weight and self.linear.bias
        '''
        with torch.no_grad():
            # ===============
            '''将automatic_update设为False，不调用torch的自动更新，手动完成梯度下降法优化'''
            #todo
            # ===============

def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

def train_model(train_set, test_set, automatic_update=False):
    train_dataloader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=10000, shuffle=False)

    #print('train_iter', len(train))
    model = LogisticRegression(2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_func = nn.BCELoss()

    num_epochs = 10
    animation_fram = []
    for epoch in range(1, num_epochs + 1):
        train_l_sum, train_acc_sum, n = 0., 0., 0
        for Xy in train_dataloader:
            X, y = Xy[:, :-1], Xy[:, -1]
            y_hat = model(X).squeeze(1)
            loss = loss_func(y_hat, y).sum()

            if automatic_update:
                model.automatic_update(loss, optimizer)
            else:
                model.manual_update(X, y, y_hat)

            train_l_sum += loss.item()
            train_acc_sum += (torch.ge(y_hat, torch.ones_like(y_hat) * 0.5).float() == y).sum().item()
            n += y.shape[0]
            animation_fram.append((model.linear.weight.detach().numpy()[0, 0], \
                                   model.linear.weight.detach().numpy()[0, 1], \
                                   model.linear.bias.detach().numpy(), loss.detach().numpy()))

        print('epoch %d, loss %.4f, train acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n))
        # test_acc = evaluate_accuracy(test_iter, model)
        # print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
        #       % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

    Xy = next(iter(test_dataloader))
    X, y = Xy[:, :-1], Xy[:, -1]
    y_hat = model(X).squeeze(1)
    test_acc_sum = (torch.ge(y_hat, torch.ones_like(y_hat) * 0.5).float() == y).sum().item()
    n = y.shape[0]
    print(test_acc_sum, n)
    print('test acc %.3f' %(test_acc_sum / n))

    return animation_fram

if __name__ == '__main__':
    # generate data using normal distribution
    dot_num = 100
    x_p = np.random.normal(3., 1, dot_num)
    y_p = np.random.normal(6., 1, dot_num)
    y = np.ones(dot_num)
    C1 = np.array([x_p, y_p, y]).T

    x_n = np.random.normal(6., 1, dot_num)
    y_n = np.random.normal(3., 1, dot_num)
    y = np.zeros(dot_num)
    C2 = np.array([x_n, y_n, y]).T

    plt.scatter(C1[:, 0], C1[:, 1], c='b', marker='+')
    plt.scatter(C2[:, 0], C2[:, 1], c='g', marker='o')

    data_set = np.concatenate((C1, C2), axis=0)
    train_num = int(0.8*data_set.shape[0])
    np.random.shuffle(data_set)
    train_set, test_set = data_set[:train_num], data_set[train_num:]

    # train model using data set and output animation frame
    animation_fram = train_model(train_set, test_set)

    # generate animation
    f, ax = plt.subplots(figsize=(6, 4))
    f.suptitle('Logistic Regression Example', fontsize=15)
    plt.ylabel('Y')
    plt.xlabel('X')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    line_d, = ax.plot([], [], label='fit_line')
    C1_dots, = ax.plot([], [], '+', c='b', label='actual_dots')
    C2_dots, = ax.plot([], [], 'o', c='g', label='actual_dots')

    frame_text = ax.text(0.02, 0.95, '', horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)


    # ax.legend()

    def init():
        line_d.set_data([], [])
        C1_dots.set_data([], [])
        C2_dots.set_data([], [])
        return (line_d,) + (C1_dots,) + (C2_dots,)


    def animate(i):
        xx = np.arange(10, step=0.1)
        a = animation_fram[i][0]
        b = animation_fram[i][1]
        c = animation_fram[i][2]
        yy = a / -b * xx + c / -b
        line_d.set_data(xx, yy)

        C1_dots.set_data(C1[:, 0], C1[:, 1])
        C2_dots.set_data(C2[:, 0], C2[:, 1])

        frame_text.set_text('Timestep = %.1d/%.1d\nLoss = %.3f' % (i, len(animation_fram), animation_fram[i][3]))

        return (line_d,) + (C1_dots,) + (C2_dots,)


    anim = animation.FuncAnimation(f, animate, init_func=init,
                                   frames=len(animation_fram), interval=100, blit=True)

    plt.show()