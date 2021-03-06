from __future__ import print_function
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from datasets import PartDataset
from pointnet import PointNetDenseCls #只从模型中import接口函数就可以了
import torch.nn.functional as F



parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='seg',  help='output folder')
parser.add_argument('--model', type=str, default = '',  help='model path') #之前训练的模型


opt = parser.parse_args()
print (opt)

opt.manualSeed = random.randint(1, 10000) # 在训练开始时，参数的初始化是随机的，为了让每次的结果一致，我们需要设置随机种子(seed不变，random就没用)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', classification = False, class_choice = ['Chair'])
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=int(opt.workers))

test_dataset = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', classification = False, class_choice = ['Chair'], train = False)
testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))
num_classes = dataset.num_seg_classes
print('classes', num_classes)
try:
    os.makedirs(opt.outf) #创建 output的路径
except OSError:
    pass

blue = lambda x:'\033[94m' + x + '\033[0m'


classifier = PointNetDenseCls(k = num_classes) #模型实例化， 每个点都label一个class

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model)) #载入预训练模型的参数the_model = TheModelClass(*args, **kwargs)
                                                      #                 the_model.load_state_dict(torch.load(PATH))，opt.model是个path

optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
classifier.cuda()

num_batch = len(dataset)/opt.batchSize #把data分成多个batch

for epoch in range(opt.nepoch):
    for i, data in enumerate(dataloader, 0):  #enumerate(dataloader,start= 0) 中0可省略
        points, target = data
        points, target = Variable(points), Variable(target)
        points = points.transpose(2,1)           #??? shape=nx3x1?nx1x3
        points, target = points.cuda(), target.cuda()   
        optimizer.zero_grad()               # 初始化grad=0
        pred, _ = classifier(points)        #
        pred = pred.view(-1, num_classes)   #shape=num_classesx1?
        target = target.view(-1,1)[:,0] - 1
        #print(pred.size(), target.size())
        loss = F.nll_loss(pred, target)
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1] #极大似然
        correct = pred_choice.eq(target.data).cpu().sum()
        print('[%d: %d/%d] train loss: %f accuracy: %f' %(epoch, i, num_batch, loss.data[0], correct/float(opt.batchSize * 2500)))
        
        if i % 10 == 0:
            j, data = enumerate(testdataloader, 0).next()
            points, target = data
            points, target = Variable(points), Variable(target)
            points = points.transpose(2,1) 
            points, target = points.cuda(), target.cuda()   
            pred, _ = classifier(points)
            pred = pred.view(-1, num_classes)
            target = target.view(-1,1)[:,0] - 1

            loss = F.nll_loss(pred, target)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print('[%d: %d/%d] %s loss: %f accuracy: %f' %(epoch, i, num_batch, blue('test'), loss.data[0], correct/float(opt.batchSize * 2500)))
    
    torch.save(classifier.state_dict(), '%s/seg_model_%d.pth' % (opt.outf, epoch)) #每个epoch保存一次模型torch.save(the_model, PATH)
