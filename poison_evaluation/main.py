'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import numpy as np

from models import *
from PIL import Image
import cv2
        
class CIFAR_load(torch.utils.data.Dataset):
    def __init__(self, root, baseset, dummy_root='./data', split='train', download=False, **kwargs):

        self.baseset = baseset
        self.transform = self.baseset.transform
        self.samples = os.listdir(os.path.join(root, 'data'))
        self.root = root
        self.poison_index = []
        for item in self.samples:
            self.poison_index.append(int(item.split('.')[0]))
            
        self.poison_images, self.labels, self.true_images = [], [], []
        
        for idx in range(50000):
            if idx in self.poison_index:
                poison_img = Image.open(os.path.join(self.root, 'data', str(idx)+'.png'))
                poison_img = self.transform(poison_img)
                true_img, label = self.baseset[idx]
            else:
                true_img, label = self.baseset[idx]
                poison_img = true_img
            self.poison_images.append(poison_img)
            self.labels.append(label)
            self.true_images.append(true_img)
        

    def __len__(self):
        return len(self.baseset)

    def __getitem__(self, idx):
        
        poison_img, label, true_img = self.poison_images[idx], self.labels[idx], self.true_images[idx]

        return poison_img, label, true_img

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--load_path', type=str)
parser.add_argument('--runs', type=int, default=5)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

print(args)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0,0,0), (1,1,1)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0,0,0), (1,1,1)),
])

baseset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)

trainset = CIFAR_load(root=args.load_path, baseset=baseset)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

accs = []

for run in range(args.runs):
    # Model
    print('==> Building model..')
    # net = VGG('VGG19')
    net = ResNet18()
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    # net = ShuffleNetV2(1)
    # net = EfficientNetB0()
    # net = RegNetX_200MF()
    #net = SimpleDLA()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    # Training
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets, clean_inputs) in enumerate(trainloader):
            inputs, targets, clean_inputs = inputs.to(device), targets.to(device), clean_inputs.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    def test(epoch):
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            print(f'loss: {test_loss/(batch_idx + 1)}, acc: {100. * correct/total}')

        # Save checkpoint.
        acc = 100.*correct/total
        return acc

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,75,90], gamma=0.1)
    for epoch in range(start_epoch, start_epoch+args.epochs):
        train(epoch)
        acc = test(epoch)
        
        checkpoint_path = args.load_path + '/checkpoint'
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path, exist_ok=True)
        checkpoint_path = checkpoint_path + '/ckpt.pth'
        checkpoint = {}
        checkpoint['net'] = net.state_dict()
        checkpoint['acc'] = acc
        torch.save(checkpoint, checkpoint_path)
        
        scheduler.step()
        if epoch == start_epoch + args.epochs - 1:
            accs.append(acc)

print(accs)
print(f'Mean accuracy: {np.mean(np.array(accs))}, \
            Std_error: {np.std(np.array(accs))/np.sqrt(args.runs)}')
