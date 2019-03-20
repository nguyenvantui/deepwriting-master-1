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

__all__ = ['vgg19']


model_urls = {
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}

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
        output1, self.hidden1 = self.lstm1_layer(input_, self.hidden1)
        # temp = torch.zeros([1,1,3],dtype=torch.float32).to("cuda")
        output2 = torch.zeros([1,input_.size(1),self.hidden_size],dtype=torch.float32).to("cuda")
        for i in range(input_.size(1)):
            # temp[0,0,0] = input_[0,i,:][0]
            # temp[0,0,1] = input_[0,i,:][1]
            output2[0,i,:], self.hidden1 = self.lstm1_layer((input_[0, i, :].unsqueeze(0)).unsqueeze(0), self.hidden1)
        loss = self.mdn(output2, output_)
        # loss =
        return loss

    def predict(self, input_, bias):
        output1, self.hidden1 = self.lstm1_layer(input_, self.hidden1)

        temp = output1.view(-1, output1.size(-1))
        point = self.mdn.predict(temp,bias)
        # loss =
        return point

    def reset_parameters(self):
        self.hidden1 = None

class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def forward1(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def vgg19(pretrained=True, **kwargs):
    """VGG 19-layer model (configuration "E")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
    return model

def main(config):
    torch.manual_seed(config['seed'])
    print(">>> star loading data <<<")
    id=1
    font = ImageFont.truetype("AppleGothic.ttf",180)
    training_dataset = dataset(config['training_data'])
    # print("")
    # training_dataset = dataset(config['validation_data'])

    print(">>> star loading model <<<")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model_stroke = HandwritingGenerator(config['hidden_size'], config['num_mixture_components']).to(device)
    model_stroke = HandwritingGenerator(400, 5).to(device)
    print("Checking device:",device)

    # print(training_dataset.alphabet)
    # alphabet is from 11 to 63
    # for i in range(11,63):
    #     print(training_dataset.alphabet[i])

    flag=False
    print(training_dataset.alphabet[36])
    if flag == False:
        # model_vgg = vgg19(pretrained=True).eval()
        # model_vgg = vgg19(pretrained=True).eval()
        # model_vgg.to(device)
        # print(len(training_dataset.char_labels))
        model_stroke.to(device)
        # model_stroke.load_state_dict(torch.load('model_stroke.pt'))
        optimizer = torch.optim.RMSprop(model_stroke.parameters(), lr=1e-4, eps=1e-4, alpha=0.95, momentum=0.9, centered=True)
        print(config['learning_rate'])
        min = 999.0
        supercount=0
        for epoch in range(config['num_epochs']):
            print("Epoch {}:".format(epoch))
            count = 1
            total_loss = 0
            # print(len(training_dataset.char_labels))
            # time.sleep(1000000)
            for x in range(len(training_dataset.char_labels)):
                read_data = training_dataset.char_labels[x]
                all_stroke = training_dataset.undo_normalization(training_dataset.samples[x])
                # xxx = np.array([0.0, 0.0, 0.0],dtype=float)
                stroke = []
                # stroke.append(xxx)
                # for id in range(len(read_data)-1):
                countt=-1
                for char in training_dataset.char_labels[x]:
                    # if (read_data[id]<=63) and (11<=read_data[id]):
                    countt+=1
                    # print(char)
                    if (training_dataset.char_encoder.classes_[char]=="z"):
                        # print(x)
                        # print(all_stroke[id])
                        # char_ = training_dataset.alphabet[read_data[id]]
                        # all_stroke[id].append(1.0);
                        temp = all_stroke[countt]
                        stroke.append(temp)
                    else:
                        if len(stroke)>4:
                            # print(len(stroke))
                            # print(char_)
                            # for idx in range(1, len(stroke)):
                            #     print(stroke[idx])
                            # time.sleep(1000)

                            count+=1
                            n_train = len(stroke)
                            # print(n_train)
                            stroke = np.array(stroke)
                            # # for i in range()
                            x_data = stroke[:,0]*100
                            y_data = stroke[:,1]*100
                            p_data = stroke[:,2]
                            xx = 0
                            yy = 0
                            draw_xx=[]
                            draw_yy=[]
                            plt.cla()
                            for i in range(n_train):
                                xx += x_data[i]
                                yy += y_data[i]
                                if int(p_data[i]) == 0:
                                    draw_xx.append(xx)
                                    draw_yy.append(yy)
                                else:
                                    plt.plot(draw_xx, draw_yy, 'b')
                                    draw_xx = []
                                    draw_yy = []
                                # draw_xx.append(xx)
                                # draw_yy.append(yy)
                            # testing the model
                            # plt.show()
                            plt.plot(draw_xx, draw_yy)
                            plt.gca().invert_yaxis()
                            plt.savefig(str(count) + ".jpg")
                            plt.close()
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
                            optimizer.zero_grad()
                            loss = None
                            model_stroke.reset_parameters()
                            loss = model_stroke(input, output)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm(model_stroke.parameters(), 2)
                            optimizer.step()
                            total_loss+=loss.item()
                            if count%config['draw']==0:
                                # print("Epoch: {}, Percent:{:.2f}, Loss: {} ".format(epoch,(count/6204)*100, total_loss/config['draw']))
                                print("Epoch: {}, Percent:{:.2f}, Loss: {} ".format(epoch,(count/100)*100, total_loss/config['draw']))
                                # total_loss=0
                                # torch.save(model_stroke.state_dict(), 'model_stroke.pt')
                            # supercount += 1
                            # if (supercount + 1) % 100 == 0:
                            #     print("Testing")
                            #     plt.cla()
                            #     model_stroke.eval()
                            #     model_stroke.to("cpu")
                            #     model_stroke.reset_parameters()
                            #     # test_model = test_model.load_state_dict(model.state_dict())
                            #     test_input = torch.zeros([1, 1, 3], dtype=torch.float).to("cpu")
                            #     # pass
                            #     test_input[0, 0, 0] = x_data[0]
                            #     test_input[0, 0, 1] = y_data[0]
                            #     # print(x_data[0])
                            #     # time.sleep(10000000)
                            #     # bp()
                            #     xx = 0
                            #     yy = 0
                            #     plt.cla()
                            #     draw_x = []
                            #     draw_y = []
                            #     for i in range(n_train - 5):
                            #         s_output = model_stroke.predict(test_input, bias=200)
                            #         test_input[0, 0, 0] = s_output[0, 0]
                            #         test_input[0, 0, 1] = s_output[0, 1]
                            #         test_input[0, 0, 2] = s_output[0, 2]
                            #         xx += s_output[0, 0].item()
                            #         yy += s_output[0, 1].item()
                            #         draw_x.append(xx)
                            #         draw_y.append(yy)
                            #         # testing the model
                            #     # plt.show()
                            #     plt.plot(draw_x, draw_y)
                            #     plt.plot(draw_x[n_train - 6],draw_y[n_train - 6],'ro')
                            #     plt.gca().invert_yaxis()
                            #     plt.savefig(str(epoch) + ".jpg")
                            #     plt.close()
                            #     model_stroke.to("cuda")
                            #     model_stroke.train()
                        stroke = []
                        # xxx = np.array([0.0, 0.0, 0.0], dtype=float)
                        # stroke.append(xxx)
    # else:
    #     char_= "z"
    #     device = "cpu"
    #     model_vgg = vgg19(pretrained=True).eval()
    #     model_vgg.to(device)
    #     # print(len(training_dataset.char_labels))
    #     model_stroke.to(device)
    #     model_stroke.load_state_dict(torch.load('model_stroke.pt'))
    #     img = Image.new("RGB", (224, 224), (255, 255, 255))
    #     draw = ImageDraw.Draw(img)
    #     draw.text((55, 10), char_, (0, 0, 0), font=font)
    #     img.show()
    #     img = normalize(transform(img)).unsqueeze(0).to(device)
    #     # img = normalize(img)
    #     output_cnn = model_vgg.forward1(img)
    #     output_cnn = output_cnn.unsqueeze(0).to(device)
    #     input_strokes = torch.zeros([1,1,2], dtype=torch.float32).to(device)
    #     input_strokes[0,0,0] = 0.0
    #     input_strokes[0,0,1] = 0.0
    #     all_input=[]
    #     abs_x=0
    #     abs_y=0
    #     for i in range(50):
    #         output = model_stroke(input_strokes, output_cnn)
    #         x=output[0,0,0].item()
    #         y=output[0,0,1].item()
    #         abs_x += x
    #         abs_y += y
    #         plt.plot(abs_x, abs_y, 'bo')
    #         print(x," ",y)
    #         input_strokes[0, 0, 0] = x
    #         input_strokes[0, 0, 1] = y
    #         # all_input.append(input_)
    #     # print(count)
    #     plt.gca().invert_yaxis()
    #     plt.show()
if __name__ == '__main__':
    config_dict = qconfig.main()
    main(config_dict)
    print(">>> kho mau <<<")

