import torch
import time
import shutil
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from loss.contrastive import BalSCL
from loss.logitadjust import LogitAdjust
from supcon import SupConLoss
#from loss.focalloss import FocalLoss
import math
from tensorboardX import SummaryWriter
from dataset.inat import INaturalist
from dataset.imagenet import ImageNetLT
from models import resnext,resnet_cifar
import numpy as np
from prettytable import PrettyTable

import warnings
import torch.backends.cudnn as cudnn
import random
from dataset.cifar import IMBALANCECIFAR100
from dataset.cifar import IMBALANCECIFAR10
from randaugment import rand_augment_transform, AutoAugmentOp
import torchvision
from utils import GaussianBlur, shot_acc
# from torch.models.tensorboard import SummaryWriter
import argparse
import os
import json
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Cifar', choices=['inat','Cifar','imagenet'])
parser.add_argument('--cifaropt', default='100', choices=['100','10'])
parser.add_argument('--data', default='/home/kin/桌面/cifar100/', metavar='DIR')
parser.add_argument('--arch', default='resnet50', choices=['resnet50', 'resnext50','resnet32'])
parser.add_argument('--workers', default=0, type=int)
parser.add_argument('--epochs', default=300, type=int)
parser.add_argument('--temp', default=0.07, type=float, help='scalar temperature for contrastive learning')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[160, 180], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=20, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--alpha', default=0.5, type=float, help='cross entropy loss weight')
parser.add_argument('--beta', default=0.5, type=float, help='supervised contrastive loss weight')
parser.add_argument('--randaug', default=True, type=bool, help='use RandAugmentation for classification branch')
parser.add_argument('--cl_views', default='sim-sim', type=str, choices=['sim-sim', 'sim-rand', 'rand-rand','com-com'],
                    help='Augmentation strategy for contrastive learning views')
parser.add_argument('--feat_dim', default=100, type=int, choices=[100, 10, 1000],help='feature dimension of mlp head')
parser.add_argument('--warmup_epochs', default=0, type=int,
                    help='warmup epochs')
parser.add_argument('--root_log', type=str, default='log')
parser.add_argument('--cos', default=True, type=bool,
                    help='lr decays by cosine scheduler. ')
parser.add_argument('--use_norm', default=True, type=bool,
                    help='cosine classifier.')
parser.add_argument('--randaug_m', default=100, type=int, help='randaug-m')
parser.add_argument('--randaug_n', default=2, type=int, help='randaug-n')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training')
parser.add_argument('--reload', default=False, type=bool, help='load supervised model')


class CustomCallback:
    def __init__(self, filepath, save_every=100):
        self.epoch_acc = []
        self.epoch_weights = []
        self.filepath = filepath
        self.save_every = save_every


    def on_epoch_end(self, model, epoch, accuracy):
        self.epoch_acc.append(accuracy)
        if epoch % self.save_every == 0:
            weights = [param.detach().cpu().numpy().flatten() for param in model.parameters() if len(param.size()) > 1][:1]  # 仅保存前两层的权重
            self.epoch_weights.append(weights)

    def save_to_file(self):
        data = {
            'accuracy': self.epoch_acc
        }
        with open(self.filepath, 'w') as f:
            json.dump(data, f)


def main():
    args = parser.parse_args()
    args.store_name = '_'.join(
        [args.dataset, args.arch, 'batchsize', str(args.batch_size), 'epochs', str(args.epochs), 'temp', str(args.temp),
         'lr', str(args.lr), args.cl_views])

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    if args.cifaropt == '100':
        num_classes = 100
    elif args.cifaropt == '10':
        num_classes = 10

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch == 'resnet50':
        model = resnext.BCLModel(name='resnet50', num_classes=num_classes, feat_dim=args.feat_dim,
                                 use_norm=args.use_norm)
    elif args.arch == 'resnext50':
        model = resnext.BCLModel(name='resnext50', num_classes=num_classes, feat_dim=args.feat_dim,
                                 use_norm=args.use_norm)
    elif args.arch == 'resnet32':
        model = resnet_cifar.BCLModel(name='resnet32', num_classes=num_classes, feat_dim=args.feat_dim,
                                 use_norm=args.use_norm)
    else:
        raise NotImplementedError('This model is not supported')
    print(model)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    gradients = {name: [] for name, param in model.named_parameters() if param.requires_grad}

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:0')
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    txt_train = f'dataset/ImageNet_LT/ImageNet_LT_train.txt' if args.dataset == 'imagenet' \
        else f'dataset/iNaturalist18/iNaturalist18_train.txt'
    txt_val = f'dataset/ImageNet_LT/ImageNet_LT_val.txt' if args.dataset == 'imagenet' \
        else f'dataset/iNaturalist18/iNaturalist18_val.txt'

    normalize = transforms.Normalize((0.466, 0.471, 0.380), (0.195, 0.194, 0.192)) if args.dataset == 'inat' \
        else transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    Normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    rgb_mean = (0.5, 0.5, 0.5)
    ra_params = dict(translate_const=int(32 * 0.45), img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]), )
    augmentation_randncls = [
        transforms.RandomResizedCrop(32, scale=(0.08, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
        ], p=1.0),
        rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(args.randaug_n, args.randaug_m), ra_params),
        transforms.ToTensor(),
        Normalize,
    ]
    augmentation_randnclsstack = [
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(args.randaug_n, args.randaug_m), ra_params),
        transforms.ToTensor(),
        Normalize,
    ]
    augmentation_sim = [
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        Normalize
    ]
    basic_augmentations = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomResizedCrop(32),
        transforms.ToTensor(),
        Normalize
    ]
    augmentation_auto = [
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        AutoAugmentOp(name='Rotate',prob=0.7,magnitude=5),
        transforms.ToTensor(),
        Normalize
    ]

    if args.cl_views == 'sim-sim':
        transform_train = [transforms.Compose(basic_augmentations), transforms.Compose(augmentation_sim),
                           transforms.Compose(augmentation_sim), ]
    elif args.cl_views == 'sim-rand':
        transform_train = [transforms.Compose(basic_augmentations), transforms.Compose(augmentation_randnclsstack),
                           transforms.Compose(augmentation_sim), ]
    elif args.cl_views == 'rand-rand':
        transform_train = [transforms.Compose(basic_augmentations), transforms.Compose(augmentation_randnclsstack),
                           transforms.Compose(augmentation_randnclsstack), ]
    elif args.cl_views == 'auto-auto':
        transform_train = [transforms.Compose(basic_augmentations), transforms.Compose(augmentation_auto),
                           transforms.Compose(augmentation_auto), ]
    elif args.cl_views == 'bas-bas':
        transform_train = [transforms.Compose(basic_augmentations), transforms.Compose(basic_augmentations),
                           transforms.Compose(basic_augmentations), ]
    else:
        raise NotImplementedError("This augmentations strategy is not available for contrastive learning branch!")

    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # 归一化
    ])
    if args.cifaropt == '100':
        train_dataset = IMBALANCECIFAR100(root=args.data, imb_type='exp', imb_factor=0.01, train=True, transform=transform_train,
                                         download=True)
        val_dataset = IMBALANCECIFAR100(root=args.data, imb_type='exp', imb_factor=0.01, train=False, transform=transform,
                                       download=True)
    elif args.cifaropt == '10':
        train_dataset = IMBALANCECIFAR10(root=args.data, imb_type='exp', imb_factor=0.01, train=True,
                                          transform=transform_train,
                                          download=True)
        val_dataset = IMBALANCECIFAR10(root=args.data, imb_type='exp', imb_factor=0.01, train=False,
                                        transform=transform,
                                        download=True)
    cls_num_list = train_dataset.get_cls_num_list()
    args.cls_num = len(cls_num_list)

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    criterion_ce = LogitAdjust(cls_num_list).cuda(args.gpu)
    criterion_scl = BalSCL(cls_num_list, args.temp).cuda(args.gpu)

    callback = CustomCallback(filepath='./log_1.json')
    #criterion_scl = SupConLoss().cuda(args.gpu)

    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))
    #confusion = ConfusionMatrix(num_classes=args.feat_dim, labels=train_dataset.targets)

    best_acc1 = 0.0
    best_many, best_med, best_few = 0.0, 0.0, 0.0

    '''if args.reload:
        txt_test = f'dataset/ImageNet_LT/ImageNet_LT_test.txt' if args.dataset == 'imagenet' \
            else f'dataset/iNaturalist18/iNaturalist18_val.txt'
        test_dataset = INaturalist(
            root=args.data,
            txt=txt_test,
            transform=val_transform, train=False
        ) if args.dataset == 'inat' else ImageNetLT(
            root=args.data,
            txt=txt_test,
            transform=val_transform, train=False)

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        acc1, acc1_avg, many, med, few, pre, recall, f1 = validate(train_loader, val_loader, model, criterion_ce , args, tf_writer)
        print('Prec@1: {:.3f}, Acc_avg@1:{:.3f}, Best Prec@1: {:.3f}, Many Prec@1: {:.3f}, Med Prec@1: {:.3f}, Few Prec@1: {:.3f}'.format(
                acc1, acc1_avg, best_acc1,
                best_many,
                best_med,
                best_few))
        return'''

    for epoch in range(args.start_epoch, args.epochs):
        adjust_lr(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion_ce, criterion_scl, optimizer, epoch, args, tf_writer)

        # evaluate on validation set
        acc1_avg, many, med, few = validate(train_loader, val_loader, model, criterion_ce, args,
                                                         tf_writer)

        # remember best acc@1 and save checkpoint
        is_best = acc1_avg > best_acc1
        best_acc1 = max(acc1_avg, best_acc1)

        callback.on_epoch_end(model, epoch, acc1_avg)
        if is_best:
            best_many = many
            best_med = med
            best_few = few
        print(' Acc_avg@1:{:.3f}, Best Prec@1: {:.3f}, Many Prec@1: {:.3f}, Med Prec@1: {:.3f}, Few Prec@1: {:.3f},'.format(acc1_avg, best_acc1,
                                                                                                                                        best_many,
                                                                                                                                        best_med,
                                                                                                                                        best_few,))
        if not os.path.isdir("./weights"):
            os.mkdir("./weights")
        save_checkpoint(args, {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best)
        torch.save(save_checkpoint, "./weights/bestModel.pth")

    '''plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(callback.epoch_acc, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy per Epoch')
    plt.legend()

    # 绘制每个epoch的权重变化（示例，仅绘制第一个权重）
    plt.subplot(1, 2, 2)
    first_weights = [weights[0][0] for weights in callback.epoch_weights]
    plt.plot(first_weights, label='Weights of First Layer')
    plt.xlabel('Epoch')
    plt.ylabel('Weight Value')
    plt.title('Weights of First Layer per 30 Epoch')
    plt.legend()
    plt.show()'''

    '''plt.subplot(1, 2, 2)
    for name, grad_list in gradients.items():
        plt.plot(range(args.start_epoch, args.epochs), grad_list, label=name)

    plt.xlabel('Epoch')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Norms over Epochs')
    #plt.legend()
    plt.grid(True)
    plt.show()'''



def train(train_loader, model, criterion_ce, criterion_scl, optimizer, epoch, args, tf_writer):
    batch_time = AverageMeter('Time', ':6.3f')
    ce_loss_all = AverageMeter('CE_Loss', ':.4e')
    scl_loss_all = AverageMeter('SCL_Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    model.train()
    end = time.time()
    for i, data in enumerate(train_loader):
        inputs, targets = data
        inputs = torch.cat([inputs[0],inputs[1],inputs[2]], dim=0)#以batchsize维度进行拼接
        inputs, targets = inputs.cuda(), targets.cuda()
        batch_size = targets.shape[0]
        feat_mlp, logits, centers = model(inputs)
        centers = centers[:args.cls_num]
        f1, f2 ,f3= torch.split(feat_mlp, [batch_size, batch_size, batch_size], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1), f3.unsqueeze(1)], dim=1)
        #output_feature = torch.cat([f1.unsqueeze(1),f2.unsqueeze(1), f3.unsqueeze(1)], dim=1)
        #logits, _, __ = torch.split(logits, [batch_size, batch_size, batch_size], dim=0)
        #logits = Local(output_feature)
        scl_loss = criterion_scl(centers, features, targets,epoch)
        #scl_loss = criterion_scl(features, targets)
        ce_loss = criterion_ce(logits, targets)
        loss = args.alpha * ce_loss + args.beta * scl_loss

        ce_loss_all.update(ce_loss.item(), batch_size)
        scl_loss_all.update(scl_loss.item(), batch_size)
        acc1 = accuracy(logits, targets, topk=(1,))
        top1.update(acc1[0].item(), batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}] \t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'CE_Loss {ce_loss.val:.4f} ({ce_loss.avg:.4f})\t'
                      'SCL_Loss {scl_loss.val:.4f} ({scl_loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                ce_loss=ce_loss_all, scl_loss=scl_loss_all, top1=top1 ))  #TODO

            print(output)

    '''for name, param in model.named_parameters():
        if param.requires_grad:
            gradients[name].append(param.grad.norm().item())'''

    tf_writer.add_scalar('CE loss/train', ce_loss_all.avg, epoch)
    tf_writer.add_scalar('SCL loss/train', scl_loss_all.avg, epoch)
    tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)


def validate(train_loader, val_loader, model, criterion_ce, args,tf_writer=None, flag='val'):
    model.eval()
    batch_time = AverageMeter('Time', ':6.3f')
    ce_loss_all = AverageMeter('CE_Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    total_logits = torch.empty((0, args.cls_num)).cuda()
    total_labels = torch.empty(0, dtype=torch.long).cuda()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            inputs, targets = data
            inputs = torch.cat([inputs, inputs, inputs], dim=0)
            inputs, targets = inputs.cuda(), targets.cuda()
            batch_size = targets.size(0)
            feat_mlp, logits, centers = model(inputs)
            ce_loss = criterion_ce(logits, targets)

            total_logits = torch.cat((total_logits, logits))
            total_labels = torch.cat((total_labels, targets))

            acc1 = accuracy(logits, targets, topk=(1,))
            ce_loss_all.update(ce_loss.item(), batch_size)
            top1.update(acc1[0].item(), batch_size)
            outputs = torch.softmax(logits, dim=1)
            outputs = torch.argmax(outputs, dim=1)
            #confusion.update(outputs.to("cpu").numpy(), targets.to("cpu").numpy())


            batch_time.update(time.time() - end)

        if i % args.print_freq == 0:
            output = ('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'CE_Loss {ce_loss.val:.4f} ({ce_loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, ce_loss=ce_loss_all, top1=top1, ))  # TODO
            print(output)
        '''TP, Precision, Recall, Specificity, Acc, F1 = confusion.summary()
        macro_Precision = sum(Precision) / len(Precision)
        macro_Recall = sum(Recall) / len(Recall)
        macro_acc = sum(Acc) / len(Acc)
        macro_F1 = sum(F1) / len(F1)
        tf_writer.add_scalar('CE loss/val', ce_loss_all.avg, epoch)
        tf_writer.add_scalar('acc/val_top1', top1.avg, epoch)
        tf_writer.add_scalar('Macro Prediction', macro_Precision, epoch)
        tf_writer.add_scalar('Macro Recall', macro_Recall, epoch)
        tf_writer.add_scalar('Macro F1', macro_F1, epoch)'''

        tf_writer.add_scalar('CE loss/val', ce_loss_all.avg)
        tf_writer.add_scalar('acc/val_top1', top1.avg)

        probs, preds = F.softmax(total_logits.detach(), dim=1).max(dim=1)
        many_acc_top1, median_acc_top1, low_acc_top1 = shot_acc(preds, total_labels, train_loader,
                                                                acc_per_cls=False)
        return top1.avg, many_acc_top1, median_acc_top1, low_acc_top1


def save_checkpoint(args, state, is_best):
    filename = os.path.join(args.root_log, args.store_name, 'bcl_ckpt.pth.tar')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


class TwoCropTransform:
    def __init__(self, transform1, transform2):
        self.transform1 = transform1
        self.transform2 = transform2

    def __call__(self, x):
        return [self.transform1(x), self.transform2(x), self.transform2(x)]


def adjust_lr(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if epoch < args.warmup_epochs:
        lr = lr / args.warmup_epochs * (epoch + 1)
    elif args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs + 1) / (args.epochs - args.warmup_epochs + 1)))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred)).contiguous()

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
