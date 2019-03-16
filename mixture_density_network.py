from matplotlib import pyplot
import torch
import torch.utils.data
import numpy
from torch.autograd import Variable
import itertools
import seaborn

n_train = 1000
batch_size = 32
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
pyplot.scatter(train_ds.x.numpy(),train_ds.y.numpy(), s=2)
pyplot.show()
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size)

class GaussianMixture1d(torch.nn.Module):
    def __init__(self, n_in, n_mixtures, eps=0):
        super(GaussianMixture1d, self).__init__()
        self.n_in = n_in
        self.eps = eps
        self.n_mixtures = n_mixtures
        self.lin = torch.nn.Linear(n_in, 3*n_mixtures)
        self.log_2pi = numpy.log(2*numpy.pi)

    def params(self, inp, pi_bias=0, std_bias=0):
        # inp = batch x input
        p = self.lin(inp)
        pi = torch.nn.functional.softmax(p[:,:self.n_mixtures]*(1+pi_bias)) # mixture weights (probability weights)
        mu = p[:,self.n_mixtures:2*self.n_mixtures] # means of the 1d gaussians
        sigma = (p[:,2*self.n_mixtures:]-std_bias).exp() # stdevs of the 1d gaussians
        sigma = sigma+self.eps
        return pi,mu,sigma

    def forward(self, inp, x):
        # x = batch x 3 (=movement x,movement y,end of stroke)
        # loss, negative log likelihood
        pi,mu,sigma = self.params(inp)
        log_normal_likelihoods =  -0.5*((x.unsqueeze(1)-mu) / sigma)**2-0.5*self.log_2pi-torch.log(sigma) # batch x n_mixtures
        log_weighted_normal_likelihoods = log_normal_likelihoods+pi.log() # batch x n_mixtures
        maxes,_ = log_weighted_normal_likelihoods.max(1)
        mixture_log_likelihood = (log_weighted_normal_likelihoods-maxes.unsqueeze(1)).exp().sum(1).log()+maxes # log-sum-exp with stabilisation
        neg_log_lik = -mixture_log_likelihood
        return neg_log_lik

    def predict(self, inp, pi_bias=0, std_bias=0):
        # inp = batch x n_in
        pi,mu,sigma = self.params(inp, pi_bias=pi_bias, std_bias=std_bias)
        x = inp.data.new(inp.size(0)).normal_()
        mixture = pi.data.multinomial(1)       # batch x 1 , index to the mixture component
        sel_mu = mu.data.gather(1, mixture).squeeze(1)
        sel_sigma = sigma.data.gather(1, mixture).squeeze(1)
        x = x*sel_sigma+sel_mu
        return Variable(x)

class Model(torch.nn.Module):
    def __init__(self, n_inp = 1, n_hid = 24, n_mixtures = 24):
        super(Model, self).__init__()
        self.lin = torch.nn.Linear(n_inp, n_hid)
        self.mix = GaussianMixture1d(n_hid, n_mixtures)
    def forward(self, inp, x):
        h = torch.tanh(self.lin(inp))
        l = self.mix(h, x)
        return l.mean()
    def predict(self, inp, pi_bias=0, std_bias=0):
        h = torch.tanh(self.lin(inp))
        return self.mix.predict(h, std_bias=std_bias, pi_bias=pi_bias)

m = Model(1, 32, 20)
opt = torch.optim.Adam(m.parameters(), 0.001)
m.cuda()
losses = []

for epoch in range(2000):
    thisloss  = 0
    for i,(x,y) in enumerate(train_dl):
        x = Variable(x.float().unsqueeze(1).cuda())
        y = Variable(y.float().cuda())
        opt.zero_grad()
        loss = m(x, y)
        loss.backward()
        thisloss += loss.data[0]/len(train_dl)
        opt.step()
    losses.append(thisloss)
    if epoch % 10 == 0:
        print(epoch, loss.data[0])

x = Variable(torch.rand(1000,1).cuda()*30-15)
y = m.predict(x)
y2 = m.predict(x, std_bias=10)
pyplot.subplot(1,2,1)
pyplot.scatter(train_ds.x.numpy(),train_ds.y.numpy(), s=2)
pyplot.scatter(x.data.cpu().squeeze(1).numpy(), y.data.cpu().numpy(),facecolor='r', s=3)
pyplot.scatter(x.data.cpu().squeeze(1).numpy(), y2.data.cpu().numpy(),facecolor='g', s=3)
pyplot.subplot(1,2,2)
pyplot.title("loss")
pyplot.plot(losses)
pyplot.show()

