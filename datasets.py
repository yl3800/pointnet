from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import json
import codecs
import numpy as np
import progressbar
import sys
import torchvision.transforms as transforms
import argparse
import json #JavaScript


class PartDataset(data.Dataset):
    def __init__(self, root, npoints = 2500, classification = False, class_choice = None, train = True):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')# 生成绝对路径
        self.cat = {}

        self.classification = classification

        with open(self.catfile, 'r') as f: #打开self.catfile，只读（f=open（‘self.catfile’，‘r’））
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]       # 字典生成方式
        #print(self.cat)                       #print出一个字典 self.cat={'Earphone': '03261776','Motorbike': '03790512,...}
        
        
        if not class_choice is  None:         # 输入一个 lass_choice
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}   #=》选定训练对象{'Mug': '03797390'，。。。}

        self.meta = {}
        for item in self.cat:  #对于每一种物品
            #print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item], 'points')  # points文件夹（每个chair一个file，存储chair的三维点云坐标）
            dir_seg = os.path.join(self.root, self.cat[item], 'points_label') #points_label文件夹（每个chair一个file，存储chair上每个的label）
            #print(dir_point, dir_seg)
            fns = sorted(os.listdir(dir_point))  # os.listdir 列出所有文件
            if train:
                fns = fns[:int(len(fns) * 0.9)] #前九个椅子作为training data
            else:
                fns = fns[int(len(fns) * 0.9):] #最后1个作为test data

            #print(os.path.basename(fns))
            for fn in fns:                     #对于每个椅子的pointcloud文件
                token = (os.path.splitext(os.path.basename(fn))[0]) #token 就是去掉路径，去掉.pts的文件名
                self.meta[item].append((os.path.join(dir_point, token + '.pts'), os.path.join(dir_seg, token + '.seg')))
                #self.meta['chair']=['/home/.../000001.pts','home/.../00001.seg']
             
        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1]))
                # self.datapath=['chair','/home/.../000001.pts','home/.../00001.seg','chair'.......]


        self.classes = dict(zip(self.cat, range(len(self.cat)))) #{'Mug': 0, 'Chair': 1}
        print(self.classes)
        self.num_seg_classes = 0
        if not self.classification: 
            for i in range(len(self.datapath)/50):
                l = len(np.unique(np.loadtxt(self.datapath[i][-1]).astype(np.uint8)))
                if l > self.num_seg_classes:
                    self.num_seg_classes = l
        #print(self.num_seg_classes)


    def __getitem__(self, index): # __getitem__,python四种特殊函数之一，例如d=PartDataset（。。。），d[index]，就会调用__getitem__
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]  # cls='chair'
        point_set = np.loadtxt(fn[1]).astype(np.float32)
        seg = np.loadtxt(fn[2]).astype(np.int64)
        #print(point_set.shape, seg.shape)

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        #resample
        point_set = point_set[choice, :]
        seg = seg[choice]
        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        if self.classification:
            return point_set, cls
        else:
            return point_set, seg

    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    print('test')
    d = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', class_choice = ['Chair'])
    print(len(d))
    ps, seg = d[0]
    print(ps.size(), ps.type(), seg.size(),seg.type())

    d = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', classification = True)
    print(len(d))
    ps, cls = d[0]
    print(ps.size(), ps.type(), cls.size(),cls.type())
