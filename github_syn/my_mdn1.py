import torch
from torch.nn.modules import Module, LSTM
import numpy as np

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
        loss_per_timestep = ( -mixture_log_likelihood - tg_pen * torch.log(bernoulli) - (1 - tg_pen) * torch.log(1 - bernoulli))

        # loss as in Graves eq (26)
        loss = torch.sum(loss_per_timestep) / batch_size
        return loss

class RNNCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = torch.nn.LSTMCell(input_size, hidden_size, bias)
    def forward(self, inp, hidden = None):
        if hidden is None:
            batch_size = inp.size(0)
            hx = inp.new(batch_size, self.hidden_size).zero_()
            cx = inp.new(batch_size, self.hidden_size).zero_()
            hidden = (hx, cx)
        return self.cell(inp, hidden)
    def get_hidden(self, hidden):
        return hidden[0]


class HandwritingModel(torch.nn.Module):
    def __init__(self, n_hidden, n_chars, n_attention_components, n_gaussians, grad_clipping=10):
        super(HandwritingModel, self).__init__()
        self.n_hidden = n_hidden
        self.n_chars = n_chars
        self.n_attention_components = n_attention_components
        self.n_gaussians = n_gaussians

        self.attention = Attention(n_hidden, n_attention_components)
        self.rnn_cell = RNNCell(3 + self.n_chars, n_hidden)
        self.grad_clipping = grad_clipping
        self.mixture = MixtureGaussians2DandPen(n_hidden + self.n_chars, n_gaussians)

    def rnn_step(self, inputs, h_state_pre, k_pre, w_pre, c, c_mask, mask=None, hidden_dict=None):
        # inputs: (batch_size, n_in + n_in_c)
        inputs = torch.cat([inputs, w_pre], dim=1)

        # h: (batch_size, n_hidden)
        h_state = self.rnn_cell(inputs, h_state_pre)
        h = self.rnn_cell.get_hidden(h_state)
        if h.requires_grad:
            h.register_hook(lambda x: x.clamp(min=-self.grad_clipping, max=self.grad_clipping))

        # update attention
        phi, k = self.attention(h, k_pre, c.size(0))
        phi = phi * c_mask
        # w: (batch_size, n_chars)
        w = torch.sum(phi.unsqueeze(-1) * c, dim=0)
        if mask is not None:
            k = mask.unsqueeze(1) * k + (1 - mask.unsqueeze(1)) * k_pre
            w = mask.unsqueeze(1) * w + (1 - mask.unsqueeze(1)) * w_pre
        if w.requires_grad:
            w.register_hook(lambda x: x.clamp(min=-100, max=100))
        return h_state, k, phi, w

    def forward(self, seq_pt, seq_mask, seq_pt_target, c, c_mask,
                h_ini=None, k_ini=None, w_ini=None, hidden_dict=None):
        batch_size = seq_pt.size(1)
        atensor = next(m.parameters())

        # if h_ini is None:
        #    h_ini = self.mixture.linear.weight.data.new(batch_size, self.n_hidden).zero_()
        if k_ini is None:
            k_ini = atensor.new(batch_size, self.n_attention_components).zero_()
        if w_ini is None:
            w_ini = atensor.new(batch_size, self.n_chars).zero_()

        # Convert the integers representing chars into one-hot encodings
        # seq_str will have shape (seq_length, batch_size, n_chars)

        c_idx = c
        c = c.data.new(c.size(0), c.size(1), self.n_chars).float().zero_()
        c.scatter_(2, c_idx.view(c.size(0), c.size(1), 1), 1)

        seq_h = []
        seq_k = []
        seq_w = []
        seq_phi = []
        h_state, k, w = h_ini, k_ini, w_ini
        for inputs, mask in zip(seq_pt, seq_mask):
            h_state, k, phi, w = self.rnn_step(inputs, h_state, k, w, c, c_mask, mask=mask, hidden_dict=hidden_dict)
            h = self.rnn_cell.get_hidden(h_state)
            seq_h.append(h)
            seq_k.append(k)
            seq_w.append(w)
            if hidden_dict is not None:
                seq_phi.append(phi)
        seq_h = torch.stack(seq_h, 0)
        seq_k = torch.stack(seq_k, 0)
        seq_w = torch.stack(seq_w, 0)
        if hidden_dict is not None:
            hidden_dict['seq_h'].append(seq_h.data.cpu())
            hidden_dict['seq_k'].append(seq_k.data.cpu())
            hidden_dict['seq_w'].append(seq_w.data.cpu())
            hidden_dict['seq_phi'].append(torch.stack(seq_phi, 0).data.cpu())
        seq_hw = torch.cat([seq_h, seq_w], dim=-1)

        loss = self.mixture(seq_hw, seq_mask, seq_pt_target, hidden_dict=hidden_dict)
        return loss

    def predict(self, pt_ini, seq_str, seq_str_mask,
                h_ini=None, k_ini=None, w_ini=None, bias=.0, n_steps=10000, hidden_dict=None):
        # pt_ini: (batch_size, 3), seq_str: (length_str_seq, batch_size), seq_str_mask: (length_str_seq, batch_size)
        # h_ini: (batch_size, n_hidden), k_ini: (batch_size, n_mixture_attention), w_ini: (batch_size, n_chars)
        # bias: float    The bias that controls the variance of the generation
        # n_steps: int   The maximal number of generation steps.
        atensor = next(m.parameters())
        bias = bias * torch.ones((), device=atensor.get_device(), dtype=atensor.dtype)
        batch_size = pt_ini.size(0)
        if k_ini is None:
            k_ini = atensor.new(batch_size, self.n_attention_components).zero_()
        if w_ini is None:
            w_ini = atensor.new(batch_size, self.n_chars).zero_()

        # Convert the integers representing chars into one-hot encodings
        # seq_str will have shape (seq_length, batch_size, n_chars)

        input_seq_str = seq_str
        seq_str = pt_ini.data.new(input_seq_str.size(0), input_seq_str.size(1), self.n_chars).float().zero_()
        seq_str.scatter_(2, input_seq_str.data.view(seq_str.size(0), seq_str.size(1), 1), 1)
        seq_str = torch.autograd.Variable(seq_str)

        mask = torch.autograd.Variable(self.mixture.linear.weight.data.new(batch_size).fill_(1))
        seq_pt = [pt_ini]
        seq_mask = [mask]

        last_char = seq_str_mask.long().sum(0) - 1

        pt, h_state, k, w = pt_ini, h_ini, k_ini, w_ini
        for i in range(n_steps):
            h_state, k, phi, w = self.rnn_step(pt, h_state, k, w, seq_str, seq_str_mask, mask=mask,
                                               hidden_dict=hidden_dict)
            h = self.rnn_cell.get_hidden(h_state)
            hw = torch.cat([h, w], dim=-1)
            pt = self.mixture.predict(hw, bias)
            seq_pt.append(pt)

            last_phi = torch.gather(phi, 0, last_char.unsqueeze(0)).squeeze(0)
            max_phi, _ = phi.max(0)
            mask = mask * (1 - (last_phi >= 0.95 * max_phi).float())
            seq_mask.append(mask)
            if mask.data.sum() == 0:
                break
        seq_pt = torch.stack(seq_pt, 0)
        seq_mask = torch.stack(seq_mask, 0)
        return seq_pt, seq_mask