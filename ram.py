import torch
import torch.nn as nn
from torch.nn import functional as F

from modules import GlimpseNetwork, LocationNetwork, CoreNetwork, ActionNetwork, BaselineNetwork
from utils import get_glimpse


class RAM(nn.Module):
    """Reccurrent Attention Model

    Reccurrent Attention Model described in the paper
    http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf

    Args:
        location_size: The number of expected features in location, 2 for MNIST dataset
        location_std: The standard deviation used by location network
        action_size: The number of expected features in action, 10 for MNIST dataset
        glimpse_size: The number of expected size of initial glimpse, 8 for MNIST dataset
            which gives a cropped image of 8X8 centered on the given location 
        num_glimpses: The number of expected glimpses
        num_scales: The number of scale of a glimpse
        feature_size: The number of expected features in h_g and h_l described
            in the paper
        glimpse_feature_size: The number of expected features in glimpse feature,
            which is g described in the paper
        hidden_size: The number of expected features in the hidden state of RNN

    Inputs: x
        - x of shape (batch_size, channel, heigth, width): tensor containing features
          of the input images, (batch_size, 1, 28, 28) for MNIST dataset

    Outputs: action_log_probs, locations, location_log_probs, baselines
        - action_log_probs of shape (batch_size, action_size): tensor containing the
          log probabilities of the predicted actions
        - locations of shape (batch_size, glimpse_size): tensor containing the locations
          for each glimpse
        - location_log_probs of shape (batch_size, location_size): tensor containing the
          log probabilities of the predicted locations
        - baselines of shape (batch_size, glimpse_size): tensor containing the
          the predicted rewards
    """

    def __init__(self, location_size, action_size, location_std,
                 glimpse_size, num_glimpses, num_scales, feature_size, glimpse_feature_size,
                 hidden_size):
        super(RAM, self).__init__()

        self.location_size = location_size
        self.glimpse_size = glimpse_size
        self.num_glimpses = num_glimpses
        self.num_scales = num_scales

        # compute input size after retina encoding
        self.input_size = glimpse_size * glimpse_size * num_scales

        self.glimpse_network = GlimpseNetwork(
            self.input_size, location_size, feature_size, glimpse_feature_size)
        self.location_network = LocationNetwork(hidden_size, location_size, location_std)
        self.action_network = ActionNetwork(hidden_size, action_size)
        self.core_network = CoreNetwork(glimpse_feature_size, hidden_size)
        self.baseline_network = BaselineNetwork(hidden_size, 1)

    def init_location(self, batch_size):
        return torch.zeros(batch_size, self.location_size)

    def forward(self, x):
        batch_size = x.size(0)
        hidden = self.core_network.init_hidden(batch_size)
        location = self.init_location(batch_size)

        location_log_probs = torch.empty(batch_size, self.num_glimpses)
        locations = torch.empty(batch_size, self.num_glimpses, self.location_size)
        baselines = torch.empty(batch_size, self.num_glimpses)

        for i in range(self.num_glimpses):
            locations[:, i] = location
            glimpse = get_glimpse(
                x, location.detach(), self.glimpse_size, self.num_scales)
            glimpse_feature = self.glimpse_network(glimpse, location)
            hidden = self.core_network(glimpse_feature, hidden)
            location, log_prob = self.location_network(hidden)
            baseline = self.baseline_network(hidden)
            location_log_probs[:, i] = log_prob
            baselines[:, i] = baseline.squeeze()
        action_logits = self.action_network(hidden)

        return action_logits, locations, location_log_probs, baselines
