import numpy as np
import qconfig1 as qconfig
import time
import torch
from torch.nn.modules import Module, LSTM
from tf_dataset_hw import HandWritingDatasetConditionalTF as dataset
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.optim as optim
import cv2
import PIL
from matplotlib import pyplot as mp
import torch.nn.functional as F
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import torchvision.transforms as T
from torch.nn.utils import clip_grad_norm
import random

transform = T.Compose([T.ToTensor()])
normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

batch_size = 400
class mdn(torch.nn.Module):
    """Gaussian mixture module as in Graves Section 4.1"""

    def __init__(self, n_inputs, n_mixture_components, eps=1e-6):
        # n_inputs = dimension N of h^n_t in Graves eq. (17)
        # n_mixture_components = number M of mixture components in Graves eq. (16)
        super().__init__()
        self.n_mixture_components = n_mixture_components
        self.eps = eps
        # linear layer as in Graves eq. (17)
        # ((proportions, m_x, m_y, s_x, s_y, rho)*n_mixture_components, pen)
        self.linear = torch.nn.Linear(n_inputs, n_mixture_components * 6 + 1)

    def compute_parameters(self, h, bias):
        # h: batch x n_inputs
        # The output of the input layer is batch x ...
        y_hat = self.linear(h)
        if y_hat.requires_grad:
            y_hat.register_hook(lambda x: x.clamp(min=-100, max=100))

        M = self.n_mixture_components

        # note that our ordering within y_hat is different to Graves eq (17)
        # we also incorporate the bias b given in Graves eq (61) and (62)
        # we have a regularisation self.eps that Graves does not have in the paper

        pi = torch.nn.functional.softmax(y_hat[:, :M] * (1 + bias), 1)  # Graves eq (19)
        mean_x = y_hat[:, M:M * 2]  # Graves eq (20)
        mean_y = y_hat[:, M * 2:M * 3]
        std_x = torch.exp(y_hat[:, M * 3:M * 4] - bias) + self.eps  # Graves eq (21)
        std_y = torch.exp(y_hat[:, M * 4:M * 5] - bias) + self.eps
        rho = torch.tanh(y_hat[:, M * 5:M * 6])  # Graves eq (22)
        rho = rho / (1 + self.eps)
        bernoulli = torch.sigmoid(y_hat[:, -1])  # Graves eq (18)
        bernoulli = (bernoulli + self.eps) / (1 + 2 * self.eps)
        # bernoulli = 1/(1+torch.exp(-(1+bias)*((torch.log(bernoulli)-torch.log(1-bernoulli))+3*(1-torch.exp(-bias))))) # this is NOT covered by Graves: Bias in the Bernoulli
        return pi, mean_x, mean_y, std_x, std_y, rho, bernoulli

    def predict(self, h, bias=0.0):
        pi, mean_x, mean_y, std_x, std_y, rho, bernoulli = self.compute_parameters(h, bias)
        mode = torch.multinomial(pi.data, 1)  # choose one mixture component
        m_x = mean_x.gather(1, mode).squeeze(1)  # data for the chosen mixture component
        m_y = mean_y.gather(1, mode).squeeze(1)
        s_x = std_x.gather(1, mode).squeeze(1)
        s_y = std_y.gather(1, mode).squeeze(1)
        r = rho.gather(1, mode).squeeze(1)

        normal = rho.new().resize_((h.size(0), 2)).normal_()
        x = normal[:, 0]
        y = normal[:, 1]

        x_n = (m_x + s_x * x).unsqueeze(-1)
        y_n = (m_y + s_y * (x * r + y * (1. - r ** 2) ** 0.5)).unsqueeze(-1)

        uniform = bernoulli.data.new(h.size(0)).uniform_()
        # print(bernoulli.data)
        pen = torch.autograd.Variable((bernoulli.data > uniform).float().unsqueeze(-1))
        # print(pen)
        return torch.cat([x_n, y_n, pen], dim=1)

    def forward(self, h_seq, tg_seq, hidden_dict=None):
        # h_seq: (seq, batch, features),  mask_seq: (seq, batch), tg_seq: (seq, batch, features=3)
        batch_size = h_seq.size(0)
        h_seq = h_seq.view(-1, h_seq.size(-1))
        tg_seq = tg_seq.view(-1, tg_seq.size(-1))

        atensor = next(self.parameters())
        bias = torch.zeros((), device=atensor.get_device(), dtype=atensor.dtype)
        pi, mean_x, mean_y, std_x, std_y, rho, bernoulli = self.compute_parameters(h_seq, bias)

        tg_x = tg_seq[:, 0:1]
        tg_y = tg_seq[:, 1:2]
        tg_pen = tg_seq[:, 2]

        tg_x_s = (tg_x - mean_x) / std_x
        tg_y_s = (tg_y - mean_y) / std_y

        z = tg_x_s ** 2 + tg_y_s ** 2 - 2 * rho * tg_x_s * tg_y_s  # Graves eq (25)

        tmp = 1 - rho ** 2
        # tmp is ln (pi N(x, mu, sigma, rho)) with N as in Graves eq (24) (this is later used for eq (26))
        mixture_part_loglikelihood = (
                    -z / (2 * tmp) - np.log(2 * np.pi) - torch.log(std_x) - torch.log(std_y) - 0.5 * torch.log(
                tmp) + torch.log(pi))

        # logsumexp over the mixture components
        # mixture_log_likelihood the log in the first part of Graves eq (26)
        mpl_max, _ = mixture_part_loglikelihood.max(1, keepdim=True)
        mixture_log_likelihood = (mixture_part_loglikelihood - mpl_max).exp().sum(1).log() + mpl_max.squeeze(1)

        # these are the summands in Graves eq (26)
        loss_per_timestep = (
                    -mixture_log_likelihood - tg_pen * torch.log(bernoulli) - (1 - tg_pen) * torch.log(1 - bernoulli))

        # loss as in Graves eq (26)
        loss = torch.sum(loss_per_timestep) / batch_size
        return loss


class HandwritingGenerator(Module):
    def __init__(self, hidden_size, num_mixture_components):
        super(HandwritingGenerator, self).__init__()
        # First LSTM layer, takes as input a tuple (x, y, eol)
        self.lstm1_layer = LSTM(input_size=3,
                                hidden_size=hidden_size,
                                batch_first=True)
        # Gaussian Window layer

        self.mdn = mdn(n_inputs=hidden_size,
                       n_mixture_components=num_mixture_components)
        # Hidden State Variables
        self.hidden_size = hidden_size
        self.hidden1 = None

        # Initiliaze parameters
        self.reset_parameters()

    def forward(self, input_, output_, bias=None):
        # First LSTM Layer
        # temp = torch.zeros([1,1,3],dtype=torch.float32).to("cuda")
        output2 = torch.zeros([batch_size,input_.size(1),self.hidden_size],dtype=torch.float32).to("cuda")
        # print(input_.size(1))
        for i in range(input_.size(1)):
            # temp[0,0,0] = input_[0,i,:][0]
            # temp[0,0,1] = input_[0,i,:][1]
            # print(i)
            output2[:, i:i+1,:], self.hidden1 = self.lstm1_layer(input_[:, i:i+1, :], self.hidden1)
        loss = self.mdn(output2, output_)
        # loss =
        return loss

    def predict(self, input_, bias):
        output1, self.hidden1 = self.lstm1_layer(input_, self.hidden1)

        point = self.mdn.predict(output1.view(-1, output1.size(-1)),bias)
        # loss =
        return point

    def reset_parameters(self):
        self.hidden1 = None

def main(config):
    torch.manual_seed(config['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_stroke = HandwritingGenerator(900, 20).to(device)
    print("Checking device:",device)

    # print(training_dataset.alphabet)
    # alphabet is from 11 to 63
    # for i in range(11,63):
    #     print(training_dataset.alphabet[i])

    flag=False
    model_stroke.to(device)
    # model_stroke.load_state_dict(torch.load('model_stroke.pt'))
    optimizer = torch.optim.RMSprop(model_stroke.parameters(), lr=1e-3, eps=1e-4, alpha=0.95, momentum=0.9, centered=True)
    print(config['learning_rate'])
    min = 999.0
    supercount=0
    file = open("mine.txt", "r")
    data = file.read()
    temp = data.split("\n")
    stroke = []
    for x in temp:
        try:
            temp = []
            # print(x)
            x = x.split(" ")
            temp.append(float(x[0]))
            temp.append(float(x[1]))
            temp.append(float(x[2]))
            stroke.append(temp)
            # print(temp)
        except:
            pass

    supercount=0
    print(len(stroke))
    n_train = len(stroke)
    # print(n_train)
    stroke = np.array(stroke)
    # # for i in range()
    x_data = torch.from_numpy(stroke[:, 0]).to(device)
    y_data = torch.from_numpy(stroke[:, 1]).to(device)
    p_data = torch.from_numpy(stroke[:, 2]).to(device)
    input = torch.zeros([batch_size, n_train - 1, 3], dtype=torch.float).to(device)
    # temp = x_data[:-1]
    for i in range(batch_size):
        input[i, :, 0] = x_data[:-1]
        input[i, :, 1] = y_data[:-1]
        input[i, :, 2] = p_data[:-1]
    output = torch.zeros([batch_size, n_train - 1, 3], dtype=torch.float).to(device)
    for i in range(batch_size):
        output[i, :, 0] = x_data[1:]
        output[i, :, 1] = y_data[1:]
        output[i, :, 2] = p_data[1:]

    for epoch in range(40000):
        # print("Epoch {}:".format(epoch))
        optimizer.zero_grad()
        loss = None
        model_stroke.reset_parameters()
        loss = model_stroke(input, output)
        loss.backward()
        torch.nn.utils.clip_grad_norm(model_stroke.parameters(), 5)
        optimizer.step()
        if (epoch+1)%200==0:
            print("Epoch: {}, Loss: {} ".format(epoch, loss.item()))

        if (epoch+1)%1000!=0:
            continue
        print("Testing")
        plt.cla()
        model_stroke.eval()
        model_stroke.to("cpu")
        model_stroke.reset_parameters()
        # test_model = test_model.load_state_dict(model.state_dict())
        test_input = torch.zeros([1, 1, 3], dtype=torch.float).to("cpu")
        # pass
        test_input[0, 0, 0] = x_data[0]
        test_input[0, 0, 1] = y_data[0]
        test_input[0, 0, 2] = p_data[0]
        xx = 0
        yy = 0
        plt.cla()
        draw_x = []
        draw_y = []
        for i in range(n_train):
            s_output = model_stroke.predict(test_input, bias=100)
            test_input[0, 0, 0] = s_output[0, 0]
            test_input[0, 0, 1] = s_output[0, 1]
            test_input[0, 0, 2] = s_output[0, 2]
            xx += s_output[0, 0].item()
            yy += s_output[0, 1].item()
            if int(s_output[0, 2].item()) == 0:
                draw_x.append(xx)
                draw_y.append(yy)
            else:
                plt.plot(draw_x,draw_y,'r')
                draw_x = []
                draw_y = []
            # testing the model
        # plt.show()
        plt.plot(draw_x, draw_y,'r')
        plt.gca().invert_yaxis()
        plt.savefig(str(epoch) + ".jpg")
        plt.close()
        model_stroke.to("cuda")
        model_stroke.train()
        print(">>> done <<<")

if __name__ == '__main__':
    config_dict = qconfig.main()
    main(config_dict)
    print(">>> kho mau <<<")

