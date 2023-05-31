# python train.py --method baseline --epochs 200
# python train.py --method baseline --data_augmentation --epochs 200
# python train.py --method cutout --data_augmentation --epochs 200
# python train.py --method mixup --data_augmentation --epochs 200
# python train.py --method cutmix --data_augmentation --epochs 200  --cutmix_prob 1.0

import numpy as np
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
# from torchvision.utils import make_grid
from torchvision import datasets, transforms

from util.misc import CSVLogger
from model.resnet import ResNet18
from model.vgg import VGG


model_options = ['resnet18', 'vgg']
#dataset_options = ['cifar100']

parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--method', default='baseline')
parser.add_argument('--model', '-a', default='resnet18',
                    choices=model_options)
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=1,
                    help='number of epochs to train (default: 20)')
parser.add_argument('--learning_rate', type=float, default=0.1,
                    help='learning rate')
parser.add_argument('--data_augmentation', action='store_true', default=False,
                    help='augment data by flipping and cropping')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed (default: 1)')
parser.add_argument('--n_holes', type=int, default=1, 
                    help='number of holes to cut out from image')
parser.add_argument('--length', type=int, default=16, 
                    help='length of the holes')
parser.add_argument('--alpha', default=0.2, type=float, 
                    help='mixup interpolation coefficient (default: 1)')
parser.add_argument('--cutmix_prob', default=1.0, type=float, 
                    help='cutmix probability')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')


def train_cutout(epoch):
    print('\nEpoch: %d' % epoch)
    xentropy_loss_avg = 0.
    correct = 0.
    total = 0.

    progress_bar = tqdm(train_loader)
    for i, (images, labels) in enumerate(progress_bar):
        progress_bar.set_description('Epoch ' + str(epoch))

        images = images.cuda()
        labels = labels.cuda()

        model.zero_grad()
        pred = model(images)

        xentropy_loss = criterion(pred, labels)
        xentropy_loss.backward()
        optimizer.step()

        xentropy_loss_avg += xentropy_loss.item()

        # Calculate running average of accuracy
        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels.data).sum().item()
        accuracy = correct / total

        progress_bar.set_postfix(
            xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
            acc='%.3f' % accuracy)
    return accuracy

def train_mixup(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    progress_bar = tqdm(train_loader)
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        progress_bar.set_description('Epoch ' + str(epoch))
        
        if args.cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, args.alpha, args.cuda)
        inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
        model.zero_grad()
        outputs = model(inputs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                    + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar.set_postfix(
            xentropy='%.3f' % (train_loss / (batch_idx + 1)),
            acc='%.3f' % (100. * correct / total))
    return (correct / total).item()


def train_cutmix(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    progress_bar = tqdm(train_loader)
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        progress_bar.set_description('Epoch ' + str(epoch))
        
        if args.cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        r = np.random.rand(1)
        if args.alpha > 0 and r < args.cutmix_prob:
            # generate mixed sample
            lam = np.random.beta(args.alpha, args.alpha)
            rand_index = torch.randperm(inputs.size()[0]).cuda()
            target_a = targets
            target_b = targets[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
            inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
            # compute output
            output = model(inputs)
            loss = cutmix_criterion(criterion, output, target_a, target_b, lam)
        else:
            # compute output
            output = model(inputs)
            loss = criterion(output, targets)

        train_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar.set_postfix(
            xentropy='%.3f' % (train_loss / (batch_idx + 1)),
            acc='%.3f' % (100. * correct / total))
    return (correct / total).item()


def test():
    model.eval()  # model turns to 'eval' mode
    correct = 0.
    total = 0.
    for images, labels in test_loader:
        images = images.cuda()
        labels = labels.cuda()

        with torch.no_grad():
            pred = model(images)

        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    test_acc = correct / total
    model.train()
    return test_acc

# Data Augmentation
args = parser.parse_args()
if args.method == 'cutout':
    from util.cutout import Cutout
    train = train_cutout
elif args.method == 'mixup':
    from util.mixup import mixup_data, mixup_criterion
    train = train_mixup
elif args.method == 'cutmix':
    from util.cutmix import rand_bbox, cutmix_criterion
    train = train_cutmix
elif args.method == 'baseline':
    train = train_cutout
else:
    raise Exception('unknown method: {}'.format(args.method))

normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                 std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
train_transform = transforms.Compose([])
if args.data_augmentation:
    train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
    train_transform.transforms.append(transforms.RandomHorizontalFlip())
train_transform.transforms.append(transforms.ToTensor())
train_transform.transforms.append(normalize)
if args.method == 'cutout':
    train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))
test_transform = transforms.Compose([transforms.ToTensor(), normalize])

args = parser.parse_args()

# CPU or GPU
args.cuda = not args.no_cuda and torch.cuda.is_available()
cudnn.benchmark = True  # Should make training go faster for large models
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
print(args)


# Data Set
num_classes = 100
train_dataset = datasets.CIFAR100(root='./data/',
                                  train=True,
                                  transform=train_transform,
                                  download=True)
test_dataset = datasets.CIFAR100(root='./data/',
                                 train=False,
                                 transform=test_transform,
                                 download=True)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           pin_memory=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=False,
                                          pin_memory=True)

# CNN Model
if args.model == 'resnet18':
    model = ResNet18(num_classes=num_classes)
elif args.model == 'vgg':
    model = VGG(num_classes=num_classes)
else:
    raise Exception('unknown model: {}'.format(args.model))

# Training(learning Rate)
model = model.cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
                            momentum=0.9, nesterov=True, weight_decay=5e-4)
scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

# Tempt Results
test_id = args.model + '_' + args.method
if not args.data_augmentation:
    test_id += '_noaugment'
filename = './runs/' + test_id + '.csv'
csv_logger = CSVLogger(args=args, fieldnames=['epoch', 'train_acc', 'test_acc']
                       , filename=filename)

for epoch in range(1, args.epochs + 1):
    train_acc = train(epoch)
    test_acc = test()
    tqdm.write('test_acc: %.3f' % test_acc)
    scheduler.step()
    row = {'epoch': str(epoch), 'train_acc': str(train_acc), 'test_acc': str(test_acc)}
    csv_logger.writerow(row)

torch.save(model.state_dict(), './checkpoint/' + test_id + '.pt')
csv_logger.close()