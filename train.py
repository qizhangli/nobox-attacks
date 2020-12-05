import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
import os
import torchvision.transforms as T
from model_autoencoder import *
import numpy as np
import sys
from torch.backends import cudnn
from PIL import Image
import torch.nn.functional as F
import time
from prep_dataset import OUR_dataset
import argparse
from utils import *
import csv

parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--n_imgs', type=int, default=20, help='number of all reference images')
parser.add_argument('--n_iters', type=int, default=15000)
parser.add_argument('--n_decoders', type=int, default=20)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--mode', type=str, default='prototypical')
parser.add_argument('--save_dir', type=str, default='./trained_ae')
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=250)

def initialize_model(decoder_num):
    model = autoencoder(input_nc=3, output_nc=3, n_blocks=3, decoder_num=decoder_num)
    model = nn.Sequential(
        Normalize(),
        model,
    )
    model.to(device)
    return model

def train_prototypical(model, img, n_imgs, n_decoders, n_iters, prototype_ind_csv_writer):
    if n_imgs == 1:
        tar_ind_ls = [0, 1]
    else:
        tar_ind_ls = mk_proto_ls(n_imgs)
    tar_ind_ls = tar_ind_ls[:n_decoders * 2]
    prototype_ind_csv_writer.writerow(tar_ind_ls.tolist())
    img_tar = img[tar_ind_ls]
    if n_decoders != 1:
        img_tar = F.interpolate(img_tar, (56, 56))
    since = time.time()
    for i in range(n_iters):
        rand_ind = torch.cat((torch.randint(0, n_imgs, size=(1,)), torch.randint(n_imgs, 2 * n_imgs, size=(1,))))
        img_input = img[rand_ind].clone()
        if do_aug:
            img_input = aug(img_input)
        assert img_input.shape[3] == 224
        outputs, _ = model(img_input)
        gen_img = torch.cat(outputs, dim=0)
        loss = nn.MSELoss()(gen_img, img_tar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 100 == 0:
            print(iter_ind + 1, i + 1, round(loss.item(), 5), '{} s'.format(int(time.time() - since)))
    return model
def train_unsup(model, img, n_iters):
    img_input = img
    img_tar = img.clone()
    since = time.time()
    for i in range(n_iters):
        for img_ind in range(img_input.shape[0]):
            if args.mode == 'rotate':
                img_input[img_ind:img_ind + 1] = rot(img_input[img_ind:img_ind + 1])
            elif args.mode == 'jigsaw':
                img_input[img_ind] = shuffle(img_input[img_ind], 1)
        outputs, _ = model(img_input)
        loss = nn.MSELoss()(outputs[0], img_tar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 100 == 0:
            print(iter_ind + 1, i + 1, round(loss.item(), 5), '{} s'.format(int(time.time() - since)))
    return model

if __name__ == '__main__':
    args = parser.parse_args()
    cudnn.benchmark = False
    cudnn.deterministic = True
    SEED = 0
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    print(args)


    mode = args.mode
    assert mode in ['prototypical', 'unsup_naive', 'jigsaw', 'rotate']
    save_dir = args.save_dir
    n_imgs = args.n_imgs // 2
    n_iters = args.n_iters
    lr = args.lr
    if mode != 'prototypical':
        n_decoders = 1
    else:
        n_decoders = args.n_decoders
    assert n_decoders <= n_imgs**2, 'Too many decoders.'

    os.makedirs(save_dir+'/models', exist_ok=True)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    batch_size = n_imgs*2
    do_aug = True
    data_root = './'



    trans = T.Compose([
        T.Resize((256,256)),
        T.CenterCrop(224),
        T.ToTensor()
    ])
    dataset = OUR_dataset(data_dir = 'data/ILSVRC2012_img_val',
                          data_csv_dir='data/selected_data.csv',
                          mode='train',
                          img_num = n_imgs,
                          transform = trans)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers = 1)


    if mode == 'prototypical':
        prototype_ind_csv = open(save_dir+'/prototype_ind.csv', 'a')
        prototype_ind_csv_writer = csv.writer(prototype_ind_csv)

    for iter_ind, (img, label_ind) in enumerate(data_loader):
        if not args.start <= iter_ind < args.end:
            continue
        model = initialize_model(n_decoders)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        img = img.to(device)

        if mode == 'prototypical':
            train_prototypical(model, img, n_imgs, n_decoders, n_iters, prototype_ind_csv_writer)
        else:
            train_unsup(model, img, n_iters)

        model.eval()
        torch.save(model.state_dict(), save_dir + '/models/{}.pth'.format(iter_ind))