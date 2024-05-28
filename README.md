## Candidate Label Set Pruning: A Data-centric Perspective for Deep Partial-label Learning (ICLR2024)

## Requirements

python=3.9.12

torch=1.12.1

protobuf=3.20.1

[lavis](https://github.com/salesforce/LAVIS)

[faiss-gpu=1.7.2](https://github.com/facebookresearch/faiss)

betaincder=0.1.1

scipy

## Datasets

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) 

[CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)

[Tiny-ImageNet](http://cs231n.stanford.edu/tiny-imagenet-200.zip)

[PASCAL_VOC](https://drive.google.com/file/d/1OxZWambUGPcPttBFg9oo--Vd3j3Xu9tS/view)

## Deep partial-label learning methods

[CC](https://lfeng-ntu.github.io/Code/RCCC.zip)

[PRODEN](https://github.com/lvjiaqi77/PRODEN)

[CAVL](https://github.com/Ferenas/CAVL)

[LWS](https://github.com/hongwei-wen/LW-loss-for-partial-label)

[PiCO](https://github.com/SZU-AdvTech-2022/132-PiCO-Contrastive-Label-Disambiguation-for-Partial-Label-Learning?tab=readme-ov-file)

[CRDPLL](https://github.com/wu-dd/PLCR)

[ABLE](https://github.com/AlphaXia/ABLE)

[IDGP](https://github.com/palm-ml/idgp)

[SoLa](https://github.com/hbzju/SoLar)

[RECORDS](https://github.com/MediaBrain-SJTU/RECORDS-LTPLL)

[POP](https://github.com/palm-ml/POP)

Note that we use the same training schedule (e.g, same models, optimizers, and hyperparameters) of these methods before and after pruning. 

## Feature extractors

ResNet-S: trained by supervised learning.

ResNet-SSL: trained by the self-supervised learning method [SimCLR](https://github.com/sthalles/SimCLR)

ResNet-I: using the model weight pre-trained on ImageNet-1K

These model weights can be found in https://drive.google.com/file/d/129BPiup5Aq0_QW0-YH4q4SewQdP9NI3Q/view?usp=sharing.

[CLIP](https://github.com/salesforce/LAVIS)

[ALBEF](https://github.com/salesforce/LAVIS)

[BLIP-2](https://github.com/salesforce/LAVIS)

## Run the pruning algorithm

Dataset path: {dataset_root}

Please carefully select parameters used in run.sh. 

sh run.sh {gpu_device} {dataset} {partial_rate} {imb_rate} {tau} {k}

For examples ({model_name}='blip2' {model_type}='pretrain'):

Uniform:

sh run.sh 0 cifar10 0.4 0.0 0.6 150

sh run.sh 0 cifar10 0.6 0.0 0.6 150

sh run.sh 0 cifar100 0.01 0.0 0.6 150

sh run.sh 0 cifar100 0.05 0.0 0.6 150

sh run.sh 0 tiny-imagenet 0.01 0.0 0.4 150

sh run.sh 0 tiny-imagenet 0.05 0.0 0.4 150

Instance-dependent (LD):

sh run.sh 0 cifar10 0.0 0.0 0.6 50

sh run.sh 0 cifar100 0.5 0.0 0.6 150

Label-dependent (ID):

sh run.sh 0 cifar10 -1.0 0.0 0.2 5

sh run.sh 0 cifar100 -1.0 0.0 0.2 5

sh run.sh 0 tiny-imagenet -1.0 0.0 0.2 50

Long-tailed (LT):

sh run.sh 0 cifar10 0.3 0.01 0.2 50

sh run.sh 0 cifar10 0.3 0.02 0.2 50

sh run.sh 0 cifar10 0.5 0.01 0.2 50

sh run.sh 0 cifar10 0.5 0.02 0.2 50

sh run.sh 0 cifar100 0.01 0.01 0.2 50

sh run.sh 0 cifar100 0.01 0.02 0.2 50

sh run.sh 0 cifar100 0.05 0.01 0.2 50

sh run.sh 0 cifar100 0.05 0.02 0.2 50

PASCAL_VOC:

sh run.sh 0 voc 0.0 0.0 0.1 5


## Original and pruned datasets

Original and pruned datasets used in the experiment can be found in https://drive.google.com/file/d/129BPiup5Aq0_QW0-YH4q4SewQdP9NI3Q/view?usp=sharing.

## Upper bound in Figure 1

Using cal_er.py to reproduce the Figure 1. 

## Contact us

If you have any further questions, please feel free to send an e-mail to: shuohe123@gmail.com.
