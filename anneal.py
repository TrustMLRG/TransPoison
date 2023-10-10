"""General interface script to launch poisoning jobs."""

import torch

import datetime
import time

import village

from model import AttackModel
from losses import SimCLRLoss, MoCoLoss, SymNegCosineSimilarityLoss

torch.backends.cudnn.benchmark = village.consts.BENCHMARK
torch.multiprocessing.set_sharing_strategy(village.consts.SHARING_STRATEGY)

# Parse input arguments
args = village.options().parse_args()
# 100% reproducibility?
if args.deterministic:
    village.utils.set_deterministic()
    
def set_model(opt):
    model_contra = AttackModel(arch=opt.arch, dataset=opt.dataset, opt=opt)

    if opt.cl_alg == 'SimCLR':
        criterion_contra = SimCLRLoss(temperature=opt.temp)
    elif opt.cl_alg.startswith('MoCo'):
        criterion_contra = MoCoLoss(temperature=opt.temp)
    else:
        raise ValueError(opt.cl_alg)
    

    if opt.cl_alg.startswith('MoCo'):
        for param in model_contra.backbone.encoder_k.parameters():
            param.requires_grad = False

    model_contra = model_contra.to('cuda')
    
    return model_contra, criterion_contra


if __name__ == "__main__":

    setup = village.utils.system_startup(args)

    model = village.Client(args, setup=setup)
    materials = village.Furnace(args, model.defs.batch_size, model.defs.augmentations, setup=setup)
    forgemaster = village.Forgemaster(args, setup=setup)

    start_time = time.time()
    if args.pretrained:
        print('Loading pretrained model...')
        stats_clean = None
    else:
        stats_clean = model.train(materials, max_epoch=args.max_epoch)
        
    model_contra, criterion_contra = set_model(args)
    
    train_time = time.time()

    poison_delta = forgemaster.forge(model, materials, model_contra, criterion_contra)
    forge_time = time.time()

    # Export
    if args.save is not None:
        materials.export_poison(poison_delta, path=args.poison_path, mode=args.save)

    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    print('---------------------------------------------------')
    print(f'Finished computations with train time: {str(datetime.timedelta(seconds=train_time - start_time))}')
    print(f'--------------------------- forge time: {str(datetime.timedelta(seconds=forge_time - train_time))}')
    print('-------------Job finished.-------------------------')
