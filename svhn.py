#!/usr/bin/env python3
# adapted from https://github.com/pytorch/examples/tree/master/mnist
import argparse
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from collections import defaultdict
from temperature_scaling import Platt, PlattDiag, Temperature
        

class Net(nn.Module):
    def __init__(self, calibrated=False):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class CalibratedNet(Net):
    def __init__(self, scaling):
        super().__init__()
        if 'matrix' in scaling:
            self.calibration = Platt()
        elif 'diag' in scaling:
            self.calibration = PlattDiag()
        else:
            self.calibration = Temperature()

    def forward(self, x):
        x = self.logits(x)
        x = self.calibration(x)
        return F.log_softmax(x, dim=1)
    
    def logits(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def load_from_net(self, m:Net):
        self.conv1.load_state_dict(m.conv1.state_dict())
        self.conv2.load_state_dict(m.conv2.state_dict())
        self.fc1.load_state_dict(m.fc1.state_dict())
        self.fc2.load_state_dict(m.fc2.state_dict())

        
def build_conv_digit_net(model_path, calibrated=False):
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    if calibrated:
        nnet = CalibratedNet([method for method in ['matrix', 'diag', 'temp'] if method in model_path][0])
    else:
        nnet = Net()

    nnet.load_state_dict(state_dict)
    return nnet

    
def train(args, model, device, train_loader, optimizer, epoch,test_loader = None):
    model.train()
    if args.save_epoch_progress:
        progress_list = []

    for batch_idx, (data, target) in tqdm(enumerate(train_loader), ' Train Epoch: {}'.format(epoch)):
        # 937 batches
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0 and batch_idx != 0:
            print('\tLoss: {:.6f}'.format(loss.item()))
            if args.save_epoch_progress:
                progress = {}
                test_loss = test(args, model, device, test_loader)
                progress['epoch'] = epoch
                progress['subepoch'] = batch_idx
                progress['test_loss'] = test_loss
                progress_list.append(progress)
    if args.save_epoch_progress:
        return progress_list


def calibration_post_process(args, model: CalibratedNet, device, valid_loader):
    model.eval()
    model.calibration.train()
    data = {'nll':list(), 'ece':list()}
    for i in range(args.itercali):
        nll, ece, anll, aece = platt_Scaling(valid_loader, model, args.tlr)
        data['nll'].append(nll)
        data['ece'].append(ece)
    
    data['nll'].append(anll)
    data['ece'].append(aece)
    pd.DataFrame(data).to_csv('{}.csv'.format(args.scaling))
    

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    #print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #test_loss, correct, len(test_loader.dataset),
    #100. * correct / len(test_loader.dataset)))
    return test_loss, correct / len(test_loader.dataset)


def save(model, args, training_progress, epoch):
    fname = "mnist_e{}_lr{}.pt".format(epoch,args.lr)
    if args.trained_model:
        fname = args.trained_model

    
    
    if (args.save_model):
         torch.save(model.state_dict(), fname)
         print(fname)
    if args.save_epoch_progress:
        
        dd = defaultdict(list)
        for d in training_progress:
            for key, value in d.items():
                dd[key].append(value)
        df = pd.DataFrame.from_dict(dd)
        df['lr'] = args.lr
        df.to_csv("mnist_LearningCurve.csv",mode='a',
            header=False, index=False)

    return fname


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the resulting Model')
    parser.add_argument('--calibrated', action='store_true', default=False,
                        help='enables calibration post processing step')
    parser.add_argument('--tlr',type=float, default=0.01,
                        help='Learning rate for the temperature scaling step')
    parser.add_argument('--itercali', type=int, default=10, help='iterations of calibration')
    parser.add_argument('--trained_model', type=str, help='path to an pretrained model to calibrate', default='')
    parser.add_argument('--save-every', action='store_true', help='enable this to save model at each epoch', default=True)

    parser.add_argument('--save-epoch-progress', action='store_true', default=True, 
                        help='For running validation set at subepochs')

    parser.add_argument('--scaling', type=str, default='matrix', help='platt scaling technique (`matrix`, `diag`, `temp`)')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    data_transform = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])

    train_dataset = datasets.SVHN('data', train=True, download=True,
                       transform=data_transform)

    split = int(np.floor_divide(len(train_dataset), 1/0.8)) # 80% train, 20% valid
    indices = list(range(len(train_dataset)))

    train_sampler = SubsetRandomSampler(indices[:split])
    valid_sampler = SubsetRandomSampler(indices[split:])

    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.batch_size, sampler=train_sampler, **kwargs)
    
    valid_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.batch_size, sampler=valid_sampler, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    model =  Net().to(device) 
    if args.trained_model:
        model = build_conv_digit_net(args.trained_model)
    
    c_model = CalibratedNet(args.scaling)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    training_progress = []

    # skip training for pretrained clf
    if not args.trained_model :
        for epoch in range(1, args.epochs + 1):
            _progress  = train(args, model, device, train_loader, optimizer, epoch,test_loader)
            if args.save_epoch_progress:
                training_progress = training_progress + _progress
            if args.save_every:
                fname = save(model, args, training_progress, epoch)


    if args.calibrated :
        print('calibrating')
        c_model.load_from_net(model)
        calibration_post_process(args, c_model, device, valid_loader)
        fname = args.trained_model if args.trained_model else fname
        fname = "plattcalibrated-{}_{}_tl{}.pt".format(args.scaling, fname.split('.')[0], args.tlr)
        torch.save(c_model.state_dict(), fname)
        
    # test set evaluation
    test(args, c_model if args.calibrated else model, device, test_loader)
        

   

        
if __name__ == '__main__':
	main()
