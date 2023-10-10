# TransPoison

Code for "Transferable Availability Poisoning Attacks"

## :surfer: Poison generation

```
python anneal.py --net ResNet18 --dataset CIFAR10 --recipe targeted --eps 8 --budget 1.0 --save poison_dataset --poison_path ./results/resnet18_cifar10_tp --attackoptim PGD --cl_alg SimCLR --allow_mmt_grad --restarts 1
```

## :snowboarder: Evaluation

- Eval with supervised learning
```
python poison_evaluation/main.py --load_path ./results/resnet18_cifar10_tp/ --runs 1
```

- Eval with supervised contrastive learning
```
python main.py --dataset tpcifar10 --arch resnet18 --cl_alg SupCL --folder_name baseline_tp --baseline --epochs 1000 --eval_freq 100
```

- Eval with semi-supervised learning (FixMatch)

```
python FixMatch/Train_fixmatch.py --seed 1 --arch resnet18 --dataset tpcifar10 --n_classes 10
```

- Eval with SimSiam
```
python main.py --dataset tpcifar10 --arch resnet18 --cl_alg SimSiam --folder_name baseline_tp --baseline --epochs 1000 --eval_freq 100
```

- Eval with SimCLR
```
python main.py --dataset tpcifar10 --arch resnet18 --cl_alg SimCLR --folder_name baseline_tp --baseline --epochs 1000 --eval_freq 100
```