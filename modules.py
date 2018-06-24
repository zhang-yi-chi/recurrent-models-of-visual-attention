import torch
import torch.nn as nn
from torch.nn import functional as F


class GlimpseNetwork(nn.Module):
    """Glimpse netowrk

    Glimpse network described in the paper
    http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf

    Args:
        input_size: The number of expected features in the input x
        location_size: The number of expected features in the input location
        internal_size: The number of expected features in h_g and h_l described
            in the paper
        output_size: The number of expected features in the output

    Inputs: x, location
        - x of shape (batch_size, input_size): tensor containing features of 
          the input batched images
        - location of shape (batch_size, location_size): tensor containing
          features of input batched locations

    Outputs: output
        - output of shape (batch_size, output_size): tensor containing features
          of glimpse g described in the paper
    """

    def __init__(self, input_size, location_size, internal_size, output_size):
        super(GlimpseNetwork, self).__init__()

        self.fc_g = nn.Linear(input_size, internal_size)
        self.fc_l = nn.Linear(location_size, internal_size)
        self.fc_gg = nn.Linear(internal_size, output_size)
        self.fc_lg = nn.Linear(internal_size, output_size)

    def forward(self, x, location):
        hg = F.relu(self.fc_g(x))
        hl = F.relu(self.fc_l(location))
        output = F.relu(self.fc_gg(hg) + self.fc_lg(hl))
        return output


class CoreNetwork(nn.Module):
    """Core network

    Core network described in the paper
    http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf

    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of expected features in hidden states of RNN

    Inputs: g, prev_h
        - g of shape (batch_size, input_size): tensor containing glimpse features
          computed by GlimpseNetwork
        - prev_h of shape (batch_size, hidden_size): tensor containing the previous
          hidden state

    Outputs: h
        - h of shape (batch_size, hidden_size): tensor containing the current
          hidden state
    """

    def __init__(self, input_size, hidden_size):
        super(CoreNetwork, self).__init__()

        self.hidden_size = hidden_size
        self.rnn_cell = nn.RNNCell(
            input_size, hidden_size, nonlinearity='relu')

    def forward(self, g, prev_h):
        h = self.rnn_cell(g, prev_h)
        return h

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)


class LocationNetwork(nn.Module):
    """Location network

    Location network described in the paper
    http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf

    Args:
        input_size: The number of expected features in the input x
        output_size: The number of expected features in the output

    Inputs: x, location
        - x of shape (batch_size, input_size): tensor containing the output of
          CoreNetwork

    Outputs: output, log_p
        - output of shape (batch_size, output_size): tensor containing features
          of the next location l described in the paper
        - log_p of shape (batch_size, 1): tensor containing the log probability
          of the location
    """

    def __init__(self, input_size, output_size, std=1e-3):
        super(LocationNetwork, self).__init__()

        self.std = std
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        mu = F.tanh(self.fc(x.detach()))
        # mu = torch.clamp(self.fc(x), -1.0, 1.0)
        if self.training:
            # distribution = torch.distributions.Normal(mu, self.std)
            # noise = torch.normal(torch.zeros_like(mu), self.std)
            # output = (mu + noise).detach()
            # log_p = distribution.log_prob(output)
            # log_p = torch.sum(log_p, dim=1)

            distribution = torch.distributions.Normal(mu, self.std)
            output = torch.clamp(distribution.sample(), -1.0, 1.0)
            log_p = distribution.log_prob(output)
            log_p = torch.sum(log_p, dim=1)
        else:
            # output = F.tanh(mu)
            output = mu
            log_p = torch.ones(output.size(0))
        # distribution = torch.distributions.Normal(mu, self.std)
        # output = torch.clamp(distribution.sample().detach(), -1.0, 1.0)
        # log_p = distribution.log_prob(output)
        # log_p = torch.sum(log_p, dim=1)
        return output, log_p


class ActionNetwork(nn.Module):
    """Action network

    Action network described in the paper
    http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf

    Args:
        input_size: The number of expected features in the input x
        output_size: The number of expected features in the output

    Inputs: x
        - x of shape (batch_size, input_size): tensor containing the output of
          CoreNetwork

    Outputs: logit
        - logit of shape (batch_size, output_size): tensor containing the logit
          of the predicted actions.
    """

    def __init__(self, input_size, output_size):
        super(ActionNetwork, self).__init__()

        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        logit = self.fc(x)
        return logit


class BaselineNetwork(nn.Module):
    """Baseline network

    Baseline network described in the paper
    http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf
    Here I use the simple version which only contains a simple
    fully connected neural network with sigmoid function

    Args:
        input_size: The number of expected features in the input x
        output_size: The number of expected features in the output

    Inputs: x
        - x of shape (batch_size, input_size): tensor containing the output of
          CoreNetwork

    Outputs: output
        - output of shape (batch_size, output_size): tensor containing the
          predicted rewards
    """

    def __init__(self, input_size, output_size):
        super(BaselineNetwork, self).__init__()

        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        output = F.sigmoid(self.fc(x.detach()))
        return output
