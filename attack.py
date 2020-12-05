import os
import torch
import time
import torchvision.transforms as T
from PIL import Image
import torch.nn as nn
import torchvision
import random
import argparse
import torch.nn.functional as F
import numpy as np
from torch.backends import cudnn
from model_autoencoder import *
import sys
from prep_dataset import OUR_dataset
from utils import *
import csv

parser = argparse.ArgumentParser(description='Attack')
parser.add_argument('--epsilon', type=float, default=0.1)
parser.add_argument('--ila_niters', type=int, default=100)
parser.add_argument('--ce_niters', type=int, default=200)
parser.add_argument('--ce_epsilon', type=float, default=0.3)
parser.add_argument('--ce_alpha', type=float, default=1.0)
parser.add_argument('--n_imgs', type=int, default=20)
parser.add_argument('--n_decoders', type=int, default=20)
parser.add_argument('--ae_dir', type=str, default='./trained_ae')
parser.add_argument('--save_dir', type=str, default='./adv_images')
parser.add_argument('--mode', type=str, default='prototypical')
parser.add_argument('--ce_method', type=str, default='ifgsm')
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=2500)
args = parser.parse_args()




class ILA(torch.nn.Module):
    def __init__(self):
        super(ILA, self).__init__()
    def forward(self, ori_mid, tar_mid, att_mid):
        bs = ori_mid.shape[0]
        ori_mid = ori_mid.view(bs, -1)
        tar_mid = tar_mid.view(bs, -1)
        att_mid = att_mid.view(bs, -1)
        W = att_mid - ori_mid
        V = tar_mid - ori_mid
        V = V / V.norm(p=2,dim=1, keepdim=True)
        ILA = (W*V).sum() / bs
        return ILA



def save_attack_img(img, file_dir):
    T.ToPILImage()(img.data.cpu()).save(file_dir)

def initialize_model(decoder_num):
    model = autoencoder(input_nc=3, output_nc=3, n_blocks=3, decoder_num=decoder_num)
    model = nn.Sequential(
        Normalize(),
        model,
    )
    model.to(device)
    return model


def attack_ila(model, ori_img, tar_img, attack_niters, eps):
    model.eval()
    ori_img = ori_img.to(device)
    img = ori_img.clone()
    with torch.no_grad():
        _, tar_h_feats = model(tar_img)
        _, ori_h_feats = model(ori_img)
    for i in range(attack_niters):
        img.requires_grad_(True)
        _, att_h_feats = model(img)
        loss = ILA()(ori_h_feats.detach(), tar_h_feats.detach(), att_h_feats)
        if (i+1) % 50 == 0:
            print('\r ila attacking {}, {:0.4f}'.format(i+1, loss.item()),end=' ')
        loss.backward()
        input_grad = img.grad.data.sign()
        img = img.data + 1. / 255 * input_grad
        img = torch.where(img > ori_img + eps, ori_img + eps, img)
        img = torch.where(img < ori_img - eps, ori_img - eps, img)
        img = torch.clamp(img, min=0, max=1)
    print('')
    return img.data

def attack_ce_unsup(model, ori_img, attack_niters, eps, alpha, n_imgs, ce_method):
    model.eval()
    ori_img = ori_img.to(device)
    nChannels = 3
    tar_img = []
    for i in range(n_imgs):
        tar_img.append(ori_img[[i, n_imgs + i]])
    for i in range(n_imgs):
        tar_img.append(ori_img[[n_imgs+i, i]])
    tar_img = torch.cat(tar_img, dim=0)
    tar_img = tar_img.reshape(2*n_imgs,2,nChannels,224,224)
    img = ori_img.clone()
    for i in range(attack_niters):
        if ce_method == 'ifgsm':
            img_x = img
        # In our implementation of PGD, we incorporate randomness at each iteration to further enhance the transferability
        elif ce_method == 'pgd':
            img_x = img + img.new(img.size()).uniform_(-eps, eps)
        img_x.requires_grad_(True)
        outs, _ = model(img_x)
        outs = outs[0].unsqueeze(1).repeat(1, 2, 1, 1, 1)
        loss_mse_ = nn.MSELoss(reduction='none')(outs, tar_img).sum(dim = (2,3,4)) / (nChannels*224*224)
        loss_mse = - alpha * loss_mse_
        label = torch.tensor([0]*n_imgs*2).long().to(device)
        loss = nn.CrossEntropyLoss()(loss_mse,label)
        if (i+1) % 50 == 0 or i == 0:
            print('\r attacking {}, {:0.4f}'.format(i, loss.item()), end=' ')
        loss.backward()
        input_grad = img_x.grad.data.sign()
        img = img.data + 1. / 255 * input_grad
        img = torch.where(img > ori_img + eps, ori_img + eps, img)
        img = torch.where(img < ori_img - eps, ori_img - eps, img)
        img = torch.clamp(img, min=0, max=1)
    print('')
    return img.data

def attack_ce_proto(model, ori_img, attack_niters, eps, alpha, n_decoders, ce_method, n_imgs, prototype_inds):
    model.eval()
    ori_img = ori_img.to(device)
    tar_img = []
    for i in range(n_decoders):
        tar_img.append(ori_img[[prototype_inds[2*i],prototype_inds[2*i+1]]])
    tar_img = torch.cat(tar_img, dim = 0)
    nChannels = 3
    if n_decoders == 1:
        decoder_size = 224
    else:
        decoder_size = 56
        tar_img = F.interpolate(tar_img, size=(56,56))
    tar_img = tar_img.reshape(n_decoders,2,nChannels,decoder_size,decoder_size).unsqueeze(1)
    tar_img = tar_img.repeat(1,n_imgs*2,1,1,1,1).reshape(n_imgs*2*n_decoders,2,nChannels,decoder_size,decoder_size)
    img = ori_img.clone()
    for i in range(attack_niters):
        if ce_method == 'ifgsm':
            img_x = img
        elif ce_method == 'pgd':
            img_x = img + img.new(img.size()).uniform_(-eps, eps)
        img_x.requires_grad_(True)
        outs, _ = model(img_x)
        outs = torch.cat(outs, dim = 0).unsqueeze(1).repeat(1,2,1,1,1)
        loss_mse_ = nn.MSELoss(reduction='none')(outs,tar_img).sum(dim = (2,3,4))/(nChannels*decoder_size*decoder_size)
        loss_mse = - alpha * loss_mse_
        label = torch.tensor(([0]*n_imgs+[1]*n_imgs)*n_decoders).long().to(device)
        loss = nn.CrossEntropyLoss()(loss_mse,label)
        if (i+1) % 50 == 0 or i == 0:
            print('attacking {}, {:0.4f}'.format(i, loss.item()))
        loss.backward()
        input_grad = img_x.grad.data.sign()
        img = img.data + 1. / 255 * input_grad
        img = torch.where(img > ori_img + eps, ori_img + eps, img)
        img = torch.where(img < ori_img - eps, ori_img - eps, img)
        img = torch.clamp(img, min=0, max=1)
    print('')
    return img.data


if __name__ == '__main__':
    SEED = 0
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)

    print(args)

    mode = args.mode
    save_dir = args.save_dir
    n_imgs = args.n_imgs // 2
    if mode != 'prototypical':
        n_decoders = 1
    else:
        n_decoders = args.n_decoders
    assert n_decoders <= n_imgs ** 2, 'Too many decoders.'
    os.makedirs(save_dir, exist_ok=True)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    batch_size = n_imgs * 2
    epsilon = args.epsilon
    ce_epsilon = args.ce_epsilon
    ila_niters = args.ila_niters
    ce_niters = args.ce_niters
    ce_alpha = args.ce_alpha
    ae_dir = args.ae_dir
    ce_method = args.ce_method
    assert ce_method in ['ifgsm', 'pgd']
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')




    trans = T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop(224),
        T.ToTensor()
    ])
    dataset = OUR_dataset(data_dir='data/ILSVRC2012_img_val',
                          data_csv_dir='data/selected_data.csv',
                          mode='attack',
                          img_num=n_imgs,
                          transform=trans)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = False, num_workers = 1)





    for data_ind, (ori_img, _) in enumerate(dataloader):
        if not args.start <= data_ind < args.end:
            continue
        model = initialize_model(n_decoders)
        model.load_state_dict(torch.load('{}/models/{}.pth'.format(ae_dir, data_ind)))
        model.eval()
        ori_img = ori_img.to(device)
        if mode =='prototypical':
            prototype_ind_csv = open(ae_dir+'/prototype_ind.csv', 'r')
            prototype_ind_ls = list(csv.reader(prototype_ind_csv))
            old_att_img = attack_ce_proto(model, ori_img, attack_niters = ce_niters,
                                          eps = ce_epsilon, alpha=ce_alpha, n_decoders = n_decoders,
                                          ce_method = ce_method, n_imgs = n_imgs,
                                          prototype_inds = list(map(int,prototype_ind_ls[data_ind])))   #**
        else:
            old_att_img = attack_ce_unsup(model, ori_img, attack_niters = ce_niters,
                                          eps = ce_epsilon, alpha=ce_alpha, n_imgs=n_imgs,
                                          ce_method=ce_method)

        att_img = attack_ila(model, ori_img, old_att_img, ila_niters, eps=epsilon)
        for save_ind in range(batch_size):
            file_path, file_name = dataset.imgs[data_ind * 2*n_imgs + save_ind][0].split('/')[-2:]
            os.makedirs(save_dir + '/' + file_path, exist_ok=True)
            save_attack_img(img=att_img[save_ind],
                            file_dir=os.path.join(save_dir, file_path, file_name[:-5]) + '.png')
            print('\r', data_ind * batch_size + save_ind, 'images saved.', end=' ')
