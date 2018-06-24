from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from scipy.misc import imread
from utils import get_glimpse

# im = imread('5.png', mode='F')
# size = im.shape[0]
# im = im.reshape(1, 1, size, size) / 255.0
# print(im.shape)

# gsize = 600
# factor = gsize / size
# # gsize = int(size*factor)

# a = F.affine_grid(Variable(torch.tensor([[[0.8,0,0], [0,0.8,0.0]]])), torch.Size([1,1,gsize,gsize]))
# # a = Variable(torch.randn(1,5,5,2))
# print(a.size())
# x = Variable(torch.tensor(im))
# s = F.grid_sample(x, a)
# print(s.size())
# save_image(x, 'x.png')
# save_image(s, 's.png')
# exit()

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--test-t-e-s-t', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args.test_t_e_s_t)
torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)

location = torch.empty(args.batch_size, 2)
location[:, 0] = 0.5
location[:, 1] = -0.5

for batch_idx, (data, label) in enumerate(train_loader):
    data = data.to(device)
    output = get_glimpse(data, location, 10, 2)
    break
print(output.size())
save_image(data, 'data.png')
save_image(output.view(args.batch_size, 1, 20, 10), 'test.png')
print(label.size())