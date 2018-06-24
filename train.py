import argparse
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
import torch.utils.data
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

from ram import RAM
from utils import draw_locations

parser = argparse.ArgumentParser(description='RAM MNIST Example')
parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                    help='input batch size for training (default: 20)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='N',
                    help='init learning rate (default: 1e-3)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=500, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--input-size', type=int, default=28, metavar='N',
                    help='input image size for training (default: 28)')
parser.add_argument('--location-size', type=int, default=2, metavar='N',
                    help='input location size for training (default: 2)')
parser.add_argument('--location-std', type=float, default=0.15, metavar='N',
                    help='standard deviation used by location network (default: 0.15)')
parser.add_argument('--action-size', type=int, default=10, metavar='N',
                    help='input action size (number of classes) for training (default: 10)')
parser.add_argument('--glimpse-size', type=int, default=8, metavar='N',
                    help='glimpse image size for training (default: 8)')
parser.add_argument('--num-glimpses', type=int, default=7, metavar='N',
                    help='number of glimpses for training (default: 7)')
parser.add_argument('--num-scales', type=int, default=2, metavar='N',
                    help='number of scales (retina patch) for training (default: 2)')
parser.add_argument('--feature-size', type=int, default=128, metavar='N',
                    help='location and input glimpse feature size for training (default: 128)')
parser.add_argument('--glimpse-feature-size', type=int, default=256, metavar='N',
                    help='output glimpse feature size for training (default: 256)')
parser.add_argument('--hidden-size', type=int, default=256, metavar='N',
                    help='feature size for RNN (default: 256)')
args = parser.parse_args()
assert args.glimpse_size * 2**(args.num_scales - 1) <= args.input_size, \
    "glimpse_size * 2**(num_scales-1) should smaller than or equal to input-size"

torch.manual_seed(args.seed)

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# training set : validation set : test set = 50000 : 10000 : 10000
train_set = datasets.MNIST(
    'data', train=True, download=True, transform=transforms.ToTensor())
indices = list(range(len(train_set)))
valid_size = 10000
train_size = len(train_set) - valid_size
train_idx, valid_idx = indices[valid_size:], indices[:valid_size]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=args.batch_size, sampler=train_sampler, **kwargs)
valid_loader = torch.utils.data.DataLoader(
    train_set, batch_size=args.batch_size, sampler=valid_sampler, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)

model = RAM(location_size=args.location_size, location_std=args.location_std, action_size=args.action_size, glimpse_size=args.glimpse_size, num_glimpses=args.num_glimpses,
            num_scales=args.num_scales, feature_size=args.feature_size, glimpse_feature_size=args.glimpse_feature_size, hidden_size=args.hidden_size).to(device)
# Compute learning rate decay rate
lr_decay_rate = args.lr / args.epochs
# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
# scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, verbose=True, patience=5)

predtion_loss_fn = nn.CrossEntropyLoss()

def loss_function(labels, action_logits, location_log_probs, baselines):
    pred_loss = predtion_loss_fn(action_logits, labels.squeeze())

    predictions = torch.argmax(action_logits, dim=1, keepdim=True)
    num_repeats = baselines.size(-1)
    rewards = (labels == predictions.detach()).float().repeat(1, num_repeats)
    baseline_loss = F.mse_loss(rewards, baselines)

    b_rewards = rewards - baselines.detach()
    reinforce_loss = torch.mean(
        torch.sum(-location_log_probs * b_rewards, dim=1))
    return pred_loss + baseline_loss + reinforce_loss


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        action_logits, _, location_log_probs, baselines = model(data)
        labels = labels.unsqueeze(dim=1)
        loss = loss_function(labels, action_logits,
                             location_log_probs, baselines)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), train_size,
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / train_size))


def test(epoch, data_source, size):
    model.eval()
    total_correct = 0.0
    with torch.no_grad():
        for i, (data, labels) in enumerate(data_source):
            data = data.to(device)
            action_logits, _, _, _ = model(data)
            predictions = torch.argmax(action_logits, dim=1)
            total_correct += torch.sum((labels == predictions)).item()
    accuracy = total_correct / size
    image = data[0:1]
    _, locations, _, _ = model(image)
    draw_locations(image.numpy()[0][0], locations.detach().numpy()[0], epoch=epoch)
    return accuracy


best_valid_accuracy, test_accuracy = 0, 0
for epoch in range(1, args.epochs + 1):
    train(epoch)
    accuracy = test(epoch, valid_loader, valid_size)
    scheduler.step(accuracy)
    print('====> Validation set accuracy: {:.2%}'.format(accuracy))
    if accuracy > best_valid_accuracy:
        best_valid_accuracy = accuracy
        test_accuracy = test(epoch, test_loader, len(test_loader.dataset))
        # torch.save(model, 'save/best_model')
        print('====> Test set accuracy: {:.2%}'.format(test_accuracy))
print('====> Test set accuracy: {:.2%}'.format(test_accuracy))
