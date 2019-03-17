import torch
import numpy as np
import torch.utils.data
import matplotlib.pyplot as plt

n_train = 1000
class DS(torch.utils.data.Dataset):
    def __init__(self, n):
        self.n = n
        self.y = torch.rand(n)*21-10.5
        self.x = torch.sin(0.75*self.y)*7.0+self.y*0.5+torch.randn(n)
    def __len__(self):
        return self.n
    def __getitem__(self,i):
        return (self.x[i],self.y[i])

train_ds = DS(n_train)
plt.scatter(train_ds.x.numpy(),train_ds.y.numpy(), s=2)
plt.show()
# train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size)

class HandwritingGenerator(Module):
    def __init__(self, alphabet_size, hidden_size, num_window_components, num_mixture_components):
        super(HandwritingGenerator, self).__init__()
        self.alphabet_size = alphabet_size + 1
        # First LSTM layer, takes as input a tuple (x, y, eol)
        self.lstm1_layer = LSTM(input_size=3,
                                hidden_size=hidden_size,
                                batch_first=True)

        self.lstm2_layer = LSTM(input_size=3 + hidden_size + alphabet_size + 1,
                                hidden_size=hidden_size,
                                batch_first=True)

        # Third LSTM layer, takes as input the concatenation of the output of the first LSTM layer,
        # the output of the second LSTM layer
        # and the output of the Window layer

        self.lstm3_layer = LSTM(input_size=hidden_size,
                                hidden_size=hidden_size,
                                batch_first=True)

        self.output_layer = mdn(input_size=hidden_size,
                                num_mixtures=num_mixture_components)

        self.hidden1 = None
        self.hidden2 = None
        self.hidden3 = None

        # Initiliaze parameters
        self.reset_parameters()

    def forward(self, strokes, onehot):
        # First LSTM Layer
        input_ = strokes
        output1, self.hidden1 = self.lstm1_layer(input_, self.hidden1)
        # Gaussian Window Layer

        eos, pi, mu1, mu2, sigma1, sigma2, rho = self.output_layer(output1, bias)
        return (eos, pi, mu1, mu2, sigma1, sigma2, rho)