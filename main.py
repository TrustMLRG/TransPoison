import os
import argparse
import time
import math
import logging

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from model import AttackModel
from resnet import resnet18_normal
from datasets import I_CIFAR10, I_CIFAR100, TPCIFAR10, TPCIFAR100, P_CIFAR10_TwoCropTransform, P_CIFAR100_TwoCropTransform, DatasetPoisoning
from util import TwoCropTransform, AverageMeter, save_model, set_seed
from util import set_model_backbone_grad, convert_classwise_to_samplewise
from util import adjust_learning_rate, warmup_learning_rate, reduce_mean
from losses import SimCLRLoss, MoCoLoss, SymNegCosineSimilarityLoss, SupConLoss, SimSiamLoss
from evaluation import linear_eval

from util import log, RandomTransform
print_yellow = lambda text: log(text, color='yellow')
print_cyan = lambda text: log(text, color='cyan')
print_green = lambda text: log(text, color='green')
print = lambda text: log(text, color='white')

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=100,
                        help='save frequency')
    parser.add_argument('--eval_freq', type=int, default=100,
                        help='evaluate frequency')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')
    parser.add_argument('--seed', type=int, default=1112,
                        help='seed')
    parser.add_argument('--folder_name', type=str, default='',
                        help='folder name')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.5,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--cosine', action='store_true', default=True,
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true', default=True,
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    # arch / dataset
    parser.add_argument('--arch', type=str, default='resnet18',
                        help='backbone architecture')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'path', 'tpcifar10', 'tpcifar100'], help='dataset to use')
    parser.add_argument('--mean', type=str,
                        help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str,
                        help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None,
                        help='path to custom dataset')
    parser.add_argument('--dataset_size', type=int, default=50000,
                        help='dataset size (train split)')
    parser.add_argument('--size', type=int, default=32,
                        help='parameter for RandomResizedCrop')

    # contrastive learning algorithms
    parser.add_argument('--cl_alg', type=str, default='SimCLR',
                        choices=['SimCLR', 'MoCov2', 'SupCL', 'SimSiam'], help='contrastive learning algorithms to attack')
    parser.add_argument('--temp', type=float, default=0.5,
                        help='temperature for CL loss function')

    # moco related arguments
    parser.add_argument('--moco-dim', default=128, type=int,
                        help='feature dimension')
    parser.add_argument('--moco-k', default=4096, type=int,
                        help='queue size; number of negative keys')
    parser.add_argument('--moco-m', default=0.99, type=float,
                        help='moco momentum of updating key encoder')

    # different training schemes
    parser.add_argument('--baseline', action='store_true', default=False,
                        help='run baseline CL models')
    parser.add_argument('--samplewise', action='store_true', default=False,
                        help='choose samplewise contrastive poisoning')

    # contrastive poisoning(CP)-related parameters
    parser.add_argument('--num_steps', default=5, type=int,
                        help='number of steps to perturb in PGD')
    parser.add_argument('--step_size', default=0.1, type=float,
                        help='perturb step size in PGD')
    parser.add_argument('--model_step', default=1000, type=int,
                        help='number of model train steps (a large value (e.g., 1000) means training the whole dataset)')
    parser.add_argument('--noise_step', default=1000, type=int,
                        help='number of noise optimization steps (a large value (e.g., 1000) means training the whole dataset)')
    parser.add_argument('--allow_mmt_grad', action='store_true', default=False,
                        help='allow gradients to flow through the momentum encoder to update delta (for MoCov2 and BYOL)')

    # model resuming
    parser.add_argument('--resume', type=str, default='',
                        help='path to the checkpoint for resuming training')

    opt = parser.parse_args()

    if opt.cl_alg == 'MoCov2':
        opt.temp = 0.2
        opt.learning_rate = 0.3

    if not (opt.baseline):
        opt.syncBN = False

    if opt.dataset == 'cifar10' or opt.dataset == 'tpcifar10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100' or opt.dataset == 'tpcifar100':
        opt.n_cls = 100

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './data'
    opt.model_path = './results/contrastive_learning/{}_attack_models/{}'.format(opt.dataset, opt.folder_name)
    opt.tb_path = './results/contrastive_learning/{}_attack_tensorboard/{}'.format(opt.dataset, opt.folder_name)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_bsz_{}_temp_{}_trial_{}_ep_{}_seed_{}'.\
        format(opt.cl_alg, opt.dataset, opt.arch, opt.learning_rate,
               opt.batch_size, opt.temp, opt.trial, opt.epochs, opt.seed)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    # folder naming
    if opt.baseline:
        opt.model_name = f"{opt.model_name}"
    else:
        opt.model_name = f"{opt.model_name}" \
                         f"_delta_wt_{opt.delta_weight:.4f}" \
                         f"{('_samplewise' if opt.samplewise else '_classwise') + ('_Mstep_' + str(opt.model_step) + '_Nstep_' + str(opt.noise_step)) + ('_pgd_' + str(opt.num_steps) + '_' + str(opt.step_size))}" \
                         f"{('_mmt_grad') if opt.allow_mmt_grad else ''}"

    if len(opt.resume):
        opt.model_name = opt.resume.split('/')[-2]

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder, exist_ok=True)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder, exist_ok=True)

    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(os.path.join(opt.save_folder, 'training.log')),
            logging.StreamHandler()
        ])

    print(f'Options: {opt}')
    print(f'Folder name: {opt.folder_name}')
    print(f'Experiment name: {opt.model_name}')

    return opt

def set_loader(opt, model):
    # construct data loader
    if opt.dataset == 'cifar10' or opt.dataset == 'tpcifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100' or opt.dataset == 'tpcifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    if opt.baseline:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])

        if opt.dataset == 'cifar10':
            train_dataset = I_CIFAR10(root=opt.data_folder,
                                      transform=TwoCropTransform(train_transform),
                                      download=True)
        elif opt.dataset == 'cifar100':
            train_dataset = I_CIFAR100(root=opt.data_folder,
                                       transform=TwoCropTransform(train_transform),
                                       download=True)
        elif opt.dataset == 'tpcifar10':
            train_dataset = TPCIFAR10(root=opt.data_folder,
                                       transform=TwoCropTransform(train_transform))
        elif opt.dataset == 'tpcifar100':
            train_dataset = TPCIFAR100(root=opt.data_folder,
                                       transform=TwoCropTransform(train_transform)
            )
        else:
            raise ValueError(opt.dataset)

    opt.dataset_size = train_dataset.__len__()

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=int(opt.batch_size),
        num_workers=opt.num_workers, pin_memory=True, drop_last=True)

    return train_loader

def set_model(opt):
    model = AttackModel(arch=opt.arch, dataset=opt.dataset, opt=opt)

    if opt.cl_alg == 'SimCLR':
        criterion = SimCLRLoss(temperature=opt.temp)
    elif opt.cl_alg == 'SupCL':
        criterion = SupConLoss(temperature=opt.temp)
    elif opt.cl_alg.startswith('MoCo'):
        criterion = MoCoLoss(temperature=opt.temp)
    elif opt.cl_alg == 'SimSiam':
        criterion = SimSiamLoss()
    else:
        raise ValueError(opt.cl_alg)

    if opt.cl_alg.startswith('MoCo'):
        for param in model.backbone.encoder_k.parameters():
            param.requires_grad = False

    model = model.to('cuda')
    cudnn.benchmark = True

    return model, criterion

def set_optimizer(opt, model):
    optimizer = optim.SGD(model.backbone.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    optim_dict = {'optimizer': optimizer}

    return optim_dict

def resume_training(opt, model, optimizer, delta_optimizer):
    ckpt_state = torch.load(opt.resume, map_location='cpu')
    
    print_yellow(f"Checkpoint {opt.resume} loaded!")
    
    try:
        model.load_state_dict(ckpt_state['model'])
    except:
        model.module.load_state_dict(ckpt_state['model'])
    optimizer.load_state_dict(ckpt_state['optimizer'])

    return ckpt_state['epoch']

def train_cl_baseline(train_loader, model, criterion, optimizer, epoch, opt):

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (images, labels, indexes) in enumerate(train_loader):
        data_time.update(time.time() - end)
        if opt.baseline:
            images = torch.cat([images[0], images[1]], dim=0)

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            indexes = indexes.cuda(non_blocking=True)

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        output = model(images, indexes, labels=labels)

        if opt.cl_alg == 'SimCLR':
            features = output['features']
        elif opt.cl_alg == 'SupCL':
            features = output['features']
        elif opt.cl_alg == 'SimSiam':
            (z1, z2, p1, p2) = output['output']
        else:
            moco_logits = output['moco_logits']

        bsz = labels.shape[0]

        # compute loss
        if opt.cl_alg == 'SimCLR':
            con_loss = criterion(features)
        elif opt.cl_alg == 'SupCL':
            con_loss = criterion(features, labels)
        elif opt.cl_alg == 'SimSiam':
            con_loss = criterion(z1, z2, p1, p2)
        else:
            con_loss = criterion(moco_logits)

        reduce_con_loss = con_loss
        losses.update(reduce_con_loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        con_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print_yellow('Train: [{0}][{1}/{2}]\t'
                         'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Contrastive Loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))

    return losses.avg

def main_worker(opt):
    torch.autograd.set_detect_anomaly(True)

    cudnn.benchmark = True

    model, criterion = set_model(opt)

    # build optimizer
    optim_dict = set_optimizer(opt, model)
    optimizer = optim_dict['optimizer']

    # tensorboard
    logger = SummaryWriter(log_dir=opt.tb_folder, flush_secs=2)

    start_epoch = 1
    # resume training
    if len(opt.resume):
        start_epoch = resume_training(opt, model, optimizer, delta_optimizer)
        print_yellow(f"<=== Epoch [{start_epoch}] Resumed from {opt.resume}!")
        if start_epoch % opt.eval_freq == 0:
            linear_eval(model, logger, start_epoch, opt)
        start_epoch += 1

    # build data loader
    train_loader = set_loader(opt, model)
    train_iterator = iter(train_loader)

    # training routine
    for epoch in range(start_epoch, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()

        loss = train_cl_baseline(train_loader, model, criterion, optimizer, epoch, opt)

        time2 = time.time()

        print_yellow('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.add_scalar('train/loss', loss, epoch)
        logger.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

        if epoch % 50 == 0:
            save_model(model, optimizer, opt, epoch, os.path.join(opt.save_folder, 'curr_last.pth'))

        # online linear probing every eval_freq epochs
        if epoch % opt.eval_freq == 0:
            linear_eval(model, logger, epoch, opt)

    save_file = os.path.join(opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)

def main():
    opt = parse_option()
    set_seed(opt.seed)
    main_worker(opt)

if __name__ == '__main__':
    main()
