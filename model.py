import torch
import torch.nn as nn
import kornia.augmentation as K

from cl_algs import MoCo, SimSiam
from resnet import ResNetWithHead, ResNetNoHead, model_dict

from util import log
print_yellow = lambda text: log(text, color='yellow')

class AttackModel(nn.Module):
    def __init__(self, arch, dataset, opt):
        super(AttackModel, self).__init__()
        self.arch = arch
        self.dataset = dataset
        self.opt = opt
        if opt.cl_alg == 'SimCLR':
            self.backbone = ResNetWithHead(arch=arch)
        elif opt.cl_alg == 'SupCL':
            self.backbone = ResNetWithHead(arch=arch)
        elif opt.cl_alg == 'SimSiam':
            self.backbone = SimSiam(
                ResNetNoHead(arch=arch), num_ftrs=model_dict[arch][1], out_dim=512, allow_mmt_grad=opt.allow_mmt_grad
            )
        elif opt.cl_alg.startswith('MoCo'):
            self.backbone = MoCo(
                ResNetWithHead, arch=arch, dim=opt.moco_dim, K=opt.moco_k, m=opt.moco_m, T=opt.temp, mlp=True, allow_mmt_grad=opt.allow_mmt_grad
            )
        else:
            raise ValueError(opt.cl_alg)

        mean = (0., 0., 0.)
        std = (1., 1., 1.)

        normalize = K.Normalize(mean=mean, std=std)

        self.transform = nn.Sequential(
            K.RandomResizedCrop(size=(opt.size, opt.size), scale=(0.2, 1.)),
            K.RandomHorizontalFlip(),
            K.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            K.RandomGrayscale(p=0.2),
            normalize
        )

    def forward(self, img, index, labels=None, poison_delta=None, generate_flag=False):
        bsz = img.shape[0]
        
        if self.opt.baseline:
            mixed_img = img
            bsz = img.shape[0] // 2
        else:
            poison_delta = poison_delta.to('cuda')

            if generate_flag == False:
                mixed_img = torch.clamp(
                    img + torch.clamp(poison_delta[index], min=-1., max=1.), min=0., max=1.
                )
            else:
                mixed_img = torch.clamp(
                    img + torch.clamp(poison_delta, min=-1., max=1.), min=0., max=1.
                )
        # data augmentation
        if self.opt.baseline:
            aug1, aug2 = torch.split(mixed_img, [bsz, bsz], dim=0)
        else:
            aug1, aug2 = self.transform(mixed_img), self.transform(mixed_img)
        aug = torch.cat([aug1, aug2], dim=0)

        out_dict = {}
        if self.opt.cl_alg == 'SimCLR':
            features = self.backbone(aug)
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            out_dict['features'] = features
        elif self.opt.cl_alg == 'SupCL':
            features = self.backbone(aug)
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            out_dict['features'] = features
        elif self.opt.cl_alg == 'SimSiam':
            z1, z2, p1, p2 = self.backbone(aug1, aug2)
            out_dict['output'] = (z1, z2, p1, p2)
        elif self.opt.cl_alg.startswith('MoCo'):
            moco_logits = self.backbone(
                im_q=aug1, im_k=aug2.detach() if not self.opt.allow_mmt_grad else aug2
            )
            out_dict['moco_logits'] = moco_logits
        else:
            raise ValueError(self.opt.cl_alg)

        return out_dict