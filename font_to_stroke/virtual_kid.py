import time
import numpy as np
import torch
from torch.nn.modules import Module, LSTM, Linear
from tf_dataset_hw import HandWritingDatasetConditionalTF as dataset
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import torch.nn as nn
import torch.utils.data
import torchtext
import torch.utils.model_zoo as model_zoo
import torch.optim as optim
import cv2
import pickle
import PIL
from matplotlib import pyplot as mp
import torch.nn.functional as F

batch_size = 50
image_size = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hello world this challenge is easy
# i did it my way
class GaussianWindow(Module):
    def __init__(self, input_size, num_components):
        super(GaussianWindow, self).__init__()
        self.input_size = input_size
        self.num_components = num_components
        self.parameter_layer = Linear(in_features=input_size, out_features=3*num_components)

    def forward(self, input_, onehot, prev_kappa=None):
        abk_hats = self.parameter_layer(input_)
        abk = torch.exp(abk_hats).unsqueeze(3)
        alpha, beta, kappa = abk.chunk(3, dim=2)

        if prev_kappa is not None:
            kappa = kappa + prev_kappa
        u = torch.autograd.Variable(torch.arange(0, onehot.size(1)))
        u = u.float().to(device)
        phi = torch.sum(alpha * torch.exp(-beta * ((kappa- u) ** 2)), dim=2)
        # phi = torch.sum(alpha * torch.exp(-beta*(kappa-u)**2), dim=-1)
        window = torch.matmul(phi, onehot)
        return window, kappa, phi

    def __repr__(self):
        s = '{name}(input_size={input_size}, num_components={num_components})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

class Attention(torch.nn.Module):
    def __init__(self, n_inputs, n_mixture_components):
        super().__init__()

        self.n_mixture_components = n_mixture_components
        # linear layer from Graves eq (28)
        self.linear = torch.nn.Linear(n_inputs, n_mixture_components*3)
    def forward(self, h_1, kappa_prev, c_seq_len):
        # h_1: batch x n_inputs, kappa_prev batch x n_mixture_components
        K = self.n_mixture_components
        params = torch.exp(self.linear(h_1)) # exp of Graves eq (48)
        alpha = params[:,:K]                 # Graves eq (49)
        beta  = params[:,K:2*K]              # Graves eq (50)
        kappa = kappa_prev + 0.1*params[:,2*K:]  # Graves eq (51)
        u = torch.arange(0,c_seq_len, out=kappa.new()).view(-1,1,1)
        phi = torch.sum(alpha * torch.exp(-beta*(kappa-u)**2), dim=-1)
        return phi, kappa

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
        # pr(bernoulli.data)

        pen = torch.autograd.Variable((bernoulli.data > uniform).float().unsqueeze(-1))

        # pr(pen)
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
        mixture_part_loglikelihood = (-z / (2 * tmp) - np.log(2 * np.pi) - torch.log(std_x) - torch.log(std_y) - 0.5 * torch.log(tmp) + torch.log(pi))

        # logsumexp over the mixture components
        # mixture_log_likelihood the log in the first part of Graves eq (26)
        mpl_max, _ = mixture_part_loglikelihood.max(1, keepdim=True)
        mixture_log_likelihood = (mixture_part_loglikelihood - mpl_max).exp().sum(1).log() + mpl_max.squeeze(1)

        # these are the summands in Graves eq (26)
        loss_per_timestep = (-mixture_log_likelihood - tg_pen * torch.log(bernoulli) - (1 - tg_pen) * torch.log(1 - bernoulli))

        # loss as in Graves eq (26)
        loss = torch.sum(loss_per_timestep) / batch_size
        return loss

class HandwritingGenerator(Module):
    def __init__(self, hidden_size, num_window, num_mixture_components):
        super(HandwritingGenerator, self).__init__()
        # First LSTM layer, takes as input a tuple (x, y, eol)

        self.lstm1_layer = LSTM(input_size=3,
                                hidden_size=hidden_size,
                                batch_first=True)

        # self.window_layer = Attention(n_inputs=hidden_size,
        #                                    n_mixture_components=num_window)

        self.window_layer = GaussianWindow(input_size=hidden_size,
                                           num_components=num_mixture_components)

        self.mdn = mdn(n_inputs=hidden_size,
                       n_mixture_components=num_mixture_components)

        # Hidden State Variables

        self.hidden_size = hidden_size

        self.hidden1 = None
        self.prev_kappa = None
        # Initiliaze parameters
        self.reset_parameters()

    def forward(self,input_image, input_, output_, bias=None):
        # First LSTM Layer
        # output1, self.hidden1 = self.lstm1_layer(input_, self.hidden1)
        # pr(self.hidden1)

        device = "cuda"
        temp = torch.zeros([1, 1, 3], dtype=torch.float32).to(device)
        # temp = torch.autograd.Variable(temp)
        output2 = torch.zeros([1, input_.size(1), self.hidden_size], dtype=torch.float32).to(device)
        # output2 = torch.autograd.Variable(output2)

        for i in range(input_.size(1)):
            # # pr(i)
            # temp[0, 0, 0] = input_[0, i, :][0]
            # temp[0, 0, 1] = input_[0, i, :][1]

            output2[0, i, :], self.hidden1 = self.lstm1_layer((input_[0, i, :].unsqueeze(0)).unsqueeze(0), self.hidden1)
            temp = (output2[0,i,:].unsqueeze(0)).unsqueeze(0)
            window, self.prev_kappa, phi = self.window_layer(temp, image, self.prev_kappa)
            # x = F.sigmoid(self.linear1(x))
            # output2[0, i, :] = x
            # print(self.hidden1)
        # out
        # output2, self.hidden1 = self.lstm1_layer(input_, self.hidden1)
        loss = self.mdn(output2, output_)
        # b()
        return loss
    def predict(self, input_, bias=None):
        # seperately predict
        output1, self.hidden1 = self.lstm1_layer(input_, self.hidden1)
        # output1 = F.sigmoid(self.linear(output1))
        point = self.mdn.predict(output1.view(-1, output1.size(-1)))
        return point

    def reset_parameters(self):
        self.hidden1 = None
        self.prev_kappa = None

class ds(torch.utils.data.Dataset):
    def __init__(self, rawdata):
        self.n = len(rawdata.char_labels)
        self.image = []
        self.strokes = []
        self.mask = []
        self.id = 0
        maxmax = 0
        # images=[]
        # strokes=[]

        for x in range(len(rawdata.char_labels)):
            # pass

            read_data = training_dataset.char_labels[x]
            all_stroke = training_dataset.undo_normalization(training_dataset.samples[x])
            countt = -1
            stroke = []
            print("{}/{}".format(x,len(rawdata.char_labels)))
            for char in training_dataset.char_labels[x]:
                # if (read_data[id]<=63) and (11<=read_data[id]):
                # print(char)
                countt += 1
                if (training_dataset.char_encoder.classes_[char] == "A"):
                    temp = all_stroke[countt]
                    stroke.append(temp)
                else:
                    if len(stroke)>4:
                        # if maxmax<len(stroke):
                        #     maxmax=len(stroke)
                        self.id += 1
                        stroke = np.array(stroke)
                        # # for i in range()
                        x_data = stroke[:, 0] * 1000
                        y_data = stroke[:, 1] * 1000
                        p_data = stroke[:, 2]
                        xx = 0
                        yy = 0
                        draw_xx = []
                        draw_yy = []
                        plt.cla()
                        fig = plt.figure()
                        for i in range(len(stroke)):
                            xx += x_data[i]
                            yy += y_data[i]
                            if int(p_data[i]) == 0:
                                draw_xx.append(xx)
                                draw_yy.append(yy)
                            else:
                                plt.plot(draw_xx, draw_yy,'k')
                                draw_xx = []
                                draw_yy = []

                        plt.plot(draw_xx, draw_yy,'k')
                        plt.gca().invert_yaxis()
                        plt.axis('off')
                        fig.canvas.draw()
                        # image = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
                        data = np.fromstring(fig.canvas.tostring_rgb() , dtype='uint8')
                        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                        data = cv2.resize(data,dsize=(image_size,image_size))
                        data = (255 - data[:, :, 0]) / 255
                        data = (data!=0)
                        data = data.astype(float)
                        # data = data.astype(floảt)
                        # for kk in data:
                        #     print(kk)
                        plt.cla()
                        plt.clf()

                        # plt.imshow(data)
                        # plt.show()
                        # plt.savefig(str(self.id) + ".jpg")

                        plt.close()
                        self.strokes.append(stroke)
                        self.image.append(data)
                    stroke = []

        # temp = torch.zeros([1,maxmax],dtype=torch.float32)
        # temp = strokes[0]
        # print(temp)
        # for i in range(len(self.image)):
        #     print(i)

            # pass

    def __len__(self):
        return len(self.image)

    def __getitem__(self,i):
        return (self.image[i],self.strokes[i])

print(">>> qei <<<")


pickle_in = open("quan.data","rb")
dataset = pickle.load(pickle_in)
iter = np.arange(0,len(dataset))

model = HandwritingGenerator(900,100,20).to(device)
# print(len(dataset))
for i, (image,stroke) in zip(iter,dataset):
    x_data = stroke[:, 0]
    y_data = stroke[:, 1]
    p_data = stroke[:, 2]
    n_train = len(x_data)
    x_data = torch.from_numpy(x_data).to(device)
    y_data = torch.from_numpy(y_data).to(device)
    p_data = torch.from_numpy(p_data).to(device)
    input = torch.zeros([1, n_train - 1, 3], dtype=torch.float).to(device)
    # temp = x_data[:-1]
    input[0, :, 0] = x_data[:-1]
    input[0, :, 1] = y_data[:-1]
    input[0, :, 2] = p_data[:-1]
    output = torch.zeros([1, n_train - 1, 3], dtype=torch.float).to(device)
    output[0, :, 0] = x_data[1:]
    output[0, :, 1] = y_data[1:]
    output[0, :, 2] = p_data[1:]
    # print(stroke)
    loss = model(image,input,output)
#     print(i)
#     print(image)
#     print(stroke)
# print(len(dataset))
# training_dataset = dataset("./data/deepwriting_training.npz")
# dataset = ds(training_dataset)
# # train_it = torchtext.data.BucketIterator.splits(dataset, batch_size=batch_size, repeat=False)

# training_dataset = dataset("./data/deepwriting_training.npz")
# dataset = ds(training_dataset)
# filehandler = open('quan.data', 'wb')
# pickle.dump(dataset,filehandler)

print(">>> wei <<<")
