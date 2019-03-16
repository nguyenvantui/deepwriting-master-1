import torch.utils.data
import numpy as np
from torch.autograd import Variable
from torch.nn.modules import Module, Linear
from functools import reduce
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from ipdb import set_trace as bp

device  = "cuda"
n_train = 50
batch_size = 32

def generate_data(n_train):
   epsilon = np.random.normal(size=(n_train))
   x = np.empty([],dtype=float)
   for i in range(n_train):
       x= np.append(x,i*0.1)
   y = 7 * np.sin(0.75 * x) + 0.5 * x
   x_data = np.zeros([n_train], dtype=float)
   y_data = np.zeros([n_train], dtype=float)
   for i in range(1, n_train):
       x_data[i]=x[i]-x[i-1]
       y_data[i]=y[i]-y[i-1]

   return x_data, y_data

x_data, y_data = generate_data(n_train)
x_data = torch.from_numpy(x_data).to(device)
y_data = torch.from_numpy(y_data).to(device)
input = torch.zeros([1,n_train-1,3],dtype=torch.float).to(device)
temp =x_data[:-1]
# bp()
input[0,:,0] = x_data[:-1]
# bp()
input[0,:,1] = y_data[:-1]
output = torch.zeros([1,n_train-1,3],dtype=torch.float).to(device)
output[0,:,0] = x_data[1:]
output[0,:,1] = y_data[1:]
# bp()
# plt.figure(figsize=(8, 8))
# xx=0
# yy=0
# for i in range(n_train):
#     xx+=x_data[i]
#     yy+=y_data[i]
#     plt.plot(xx,yy,'bo')
# plt.scatter(x_data, y_data)
# plt.show()
# bp()

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
        mixture_part_loglikelihood = (-z / (2 * tmp)- np.log(2 * np.pi) - torch.log(std_x) - torch.log(std_y) - 0.5 * torch.log(tmp) + torch.log(pi))

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

print(">>> Training on: ",device)
model = mdn(3,20).to(device)
test_model = mdn(3,20).to("cpu")
num_epoch = 10
opt = torch.optim.RMSprop(model.parameters(), lr=1e-4, eps=1e-4, alpha=0.95, momentum=0.9, centered=True)
for epoch in range(num_epoch):
    # print()
    model.to(device)
    opt.zero_grad()
    loss = model(input,output)
    loss.backward()
    opt.step()
    print("<>",epoch," ",loss.item())
    if (epoch+1)%4:
        print("Test")
        pass
        # plt.cla()
        model.to("cpu")
        # test_model = test_model.load_state_dict(model.state_dict())
        test_input = torch.zeros([1,3],dtype=torch.float).to("cpu")
        # pass
        test_input[0,0]=x_data[1]
        test_input[0,1]=y_data[1]
        xx=0
        yy=0
        for i in range(n_train):
            s_output = model.predict(test_input,bias=10)
            test_input = s_output
            xx+=s_output[0,0].item()
            yy+=s_output[0,1].item()
            plt.plot(xx,yy,'bo')
        # testing the model
        # plt.show()

print(">> helloworld <<")
print("toi la quan")
print("hehe toi la vu duc quan")
