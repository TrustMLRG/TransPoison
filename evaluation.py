# This file is only used to load some code snippet
import argparse
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms, datasets

from util import set_optimizer, AverageMeter
from util import adjust_learning_rate, accuracy, reduce_mean
from resnet import LinearClassifier
from datasets import NOIDXTPCIFAR10, NOIDXTPCIFAR100

from util import log
print_green = lambda text: log(text, color='green')
print = lambda text: log(text, color='white')

def parse_option():
    parser = argparse.ArgumentParser('argument for linear evaluation')

    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'tpcifar10', 'tpcifar100'], help='dataset')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')

    opt, _ = parser.parse_known_args()

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    if opt.dataset == 'cifar10' or opt.dataset == 'tpcifar10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100' or opt.dataset == 'tpcifar100':
        opt.n_cls = 100
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    return opt

def set_linear_loader(opt, model, whole_model):
    # construct data loader for linear probing
    if opt.dataset == 'cifar10' or opt.dataset == 'tpcifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100' or opt.dataset == 'tpcifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder, transform=train_transform, download=True)
        val_dataset = datasets.CIFAR10(root=opt.data_folder, train=False, transform=val_transform)
    elif opt.dataset == 'tpcifar10':
        train_dataset = NOIDXTPCIFAR10(root=opt.data_folder, transform=train_transform)
        val_dataset = datasets.CIFAR10(root=opt.data_folder, train=False, transform=val_transform)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder, transform=train_transform, download=True)
        val_dataset = datasets.CIFAR100(root=opt.data_folder, train=False, transform=val_transform)
    elif opt.dataset == 'tpcifar100':
        train_dataset = NOIDXTPCIFAR100(root=opt.data_folder, transform=train_transform)
        val_dataset = datasets.CIFAR100(root=opt.data_folder, train=False, transform=val_transform)
    else:
        raise ValueError(opt.dataset)

    print(f"train size: {train_dataset.__len__()}\tval size: {val_dataset.__len__()}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=int(opt.batch_size),
        num_workers=opt.num_workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=int(opt.batch_size),
        num_workers=opt.num_workers, pin_memory=True)

    return train_loader, val_loader

def train_linear(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    # training linear classifier
    model.eval()
    classifier.train()

    losses = AverageMeter()
    top1 = AverageMeter()

    for idx, (images, labels) in enumerate(train_loader):

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # compute loss
        with torch.no_grad():
            features = model.encoder(images)
        output = classifier(features.detach())
        loss = criterion(output, labels)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))

        reduce_loss = loss
        reduce_acc1 = acc1

        # update metric
        losses.update(reduce_loss.item(), bsz)
        top1.update(reduce_acc1[0], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses.avg, top1.avg

def validate_linear(val_loader, model, classifier, criterion, opt):
    # validating linear classifier
    model.eval()
    classifier.eval()

    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = classifier(model.encoder(images))
            loss = criterion(output, labels)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))

            reduce_loss = loss
            reduce_acc1 = acc1

            # update metric
            losses.update(reduce_loss.item(), bsz)
            top1.update(reduce_acc1[0], bsz)

    return losses.avg, top1.avg

def train_val_linear(model, whole_model, opt):
    # linear probing
    best_acc = 0
    linear_opt = parse_option()
    linear_opt.dataset = opt.dataset
    linear_opt.batch_size = opt.batch_size
    linear_opt.size = opt.size
    linear_opt.arch = opt.arch
    linear_opt.data_folder = opt.data_folder
    linear_opt.epochs = 100

    print(linear_opt)
    linear_train_loader, linear_val_loader = set_linear_loader(linear_opt, model, whole_model)

    classifier = LinearClassifier(arch=linear_opt.arch, num_classes=linear_opt.n_cls)
    linear_criterion = torch.nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        classifier = classifier.to('cuda')
        linear_criterion = linear_criterion.to('cuda')
    linear_optimizer = set_optimizer(linear_opt, classifier)

    # training routine
    for epoch in range(1, linear_opt.epochs + 1):
        adjust_learning_rate(linear_opt, linear_optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, acc = train_linear(linear_train_loader, model, classifier, linear_criterion,
                                 linear_optimizer, epoch, linear_opt)

        # eval for one epoch
        val_loss, val_acc = validate_linear(linear_val_loader, model, classifier, linear_criterion, linear_opt)

        time2 = time.time()

        print('Train/Val epoch {}, total time {:.2f}, train loss {:.2f}, train acc {:.2f}, *val acc {:.2f}'.format(
            epoch, time2 - time1, loss, acc, val_acc)
        )

        if val_acc > best_acc:
            best_acc = val_acc

    return best_acc

def linear_eval(model, logger, epoch, opt):
    print_green(f"================== Epoch [{epoch}]: LINEAR EVAL ==================")
    if opt.cl_alg == 'SimCLR':
        eval_model = model.backbone
    elif opt.cl_alg == 'SupCL':
        eval_model = model.backbone
    elif opt.cl_alg == 'SimSiam':
        eval_model = model.backbone.backbone        
    else:
        eval_model = model.backbone.encoder_q
    acc = train_val_linear(eval_model, model, opt)
    print_green(f"Epoch {epoch} | ***best linear_acc {acc:.2f}\n")
    logger.add_scalar('val/linear_acc', acc, epoch)