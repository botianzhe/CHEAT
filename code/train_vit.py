import torch
import numpy as np
import cv2
import sys
import os
import shutil
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
import torch.distributed as dist

from dataset import FFDADataset,DFDataset,MultiFakeDataset
from torch import nn, optim
import net_conf as config
import datetime
from tqdm import tqdm
import imageio
from logger import Logger, Visualizer
import argparse
from xception import xception
from models.utils import Vgg19,get_logger
from logger import Logger, Visualizer
from models.mae import MAE22

from strong_transform import augmentation, trans
from prefetch_generator import BackgroundGenerator
from sklearn.metrics import roc_auc_score
from models.vision_transformer import SwinUnet as ViT_seg
from config import get_config
# from torchvision.models.swin_transformer import swin_v2_b
from models.swinvit import SwinTransformer
from models.sam import SAM
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        # self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_data = next(self.loader)
            print(self.next_data)
        except StopIteration:
            self.next_data = None
            return

    def next(self):
        # torch.cuda.current_stream().wait_stream(self.stream)
        if self.next_data is not None:
            data, label = self.next_data
            self.preload()
            return data, label
        else:
            return None, None

def load_model(gunet,path):
    ckpt = torch.load(path, map_location="cpu")
    # print(ckpt)
    start_epoch = ckpt.get("epoch", 0)
    best_acc = ckpt.get("acc1", 0.0)
    pretrained_dict = {k: v for k, v in ckpt["state_dict"].items() if 'idnet' not in k }
    # pretrained_dict = ckpt["state_dict"]
    gunet.load_state_dict(pretrained_dict)
    return gunet


def save_checkpoint(path, state_dict, epoch=0, arch="", acc1=0):
    filedir=os.path.dirname(path)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    def trans(state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[7:]
            if torch.is_tensor(v):
                v = v.cpu()
            new_state_dict[k] = v
        return new_state_dict
    gunet_state_dict=trans(state_dict)
    torch.save({
        "epoch": epoch,
        "arch": arch,
        "acc1": acc1,
        "state_dict": gunet_state_dict
    }, path)


def eval1(validate_dataset):

    torch.autograd.set_grad_enabled(False)
    gunet.eval()
    
    val_losses = []
    val_acc = []
    val_auc=[]
    t1sims=[]
    t2sims=[]
    preds=[]
    labels=[]
    with tqdm(validate_dataset, desc='Batch') as bar:
        for b, batch in enumerate(bar):
            anchorimg, label = batch
           
            anchorimg = anchorimg.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

            label = F.one_hot(label,2).float()

            
            pred,_,_=gunet(anchorimg)
            pred=nn.Softmax(dim=1)(pred)
            loss=nn.BCELoss()(pred,label)
            out = torch.argmax(pred.data, 1)
            label1 = torch.argmax(label.data, 1)
            batch_acc = torch.sum(out == label1).item() / len(out)
            try:
                auc=roc_auc_score(label.cpu(),pred.detach().cpu())
            except:
                auc=0.5
            bar.set_postfix(
                acc=batch_acc,
                loss=loss.item(),
                auc=auc
            )
            val_acc.append(batch_acc)
            val_losses.append(loss.item())
            val_auc.append(auc)
            preds.append(pred)
            labels.append(label)

    torch.autograd.set_grad_enabled(True)
    epoch_pred = torch.cat(preds,dim=0).detach().cpu().numpy()
    epoch_label = torch.cat(labels,dim=0).cpu().numpy()
    epoch_auc=roc_auc_score(epoch_label,epoch_pred)
    epoch_acc = np.mean(val_acc)
    epoch_loss=np.mean(np.array(val_losses))
    return epoch_loss,epoch_acc,epoch_auc


# gunet=xception(num_classes=2).cuda()

# gunet=load_model(gunet,'logs/test/2024-07-02 09:31:31.832895/detection_4_ckpt-0.1376572549343109.pth')
gunet=SwinTransformer(num_classes=2).cuda()
# gunet=SwinTransformer(img_size=384,embed_dim=128,depths=[2,2,18,2],num_heads=[4,8,16,32] ,window_size=12,num_classes=2).cuda()
# pretrained_dict = torch.load('swin_base_patch4_window12_384_22k.pth', map_location='cpu')
pretrained_dict = torch.load('swin_tiny_patch4_window7_224_22k.pth', map_location='cpu')
gunet.load_state_dict(pretrained_dict['model'],strict=False)
# gunet
# gunet=load_model(gunet,'logs/train_vit.py/2024-07-16 13:33:49.008721/detection_47_ckpt-1.6093875505470665.pth')
# torch.backends.cudnn.enabled = False
batch_size=16
log_dir='logs/'
filename=sys.argv[0]
date_p = datetime.datetime.now()
str_p = str(date_p)
png_dir = os.path.join(log_dir,filename,str_p, 'train/png')
if not os.path.exists(png_dir):
    os.makedirs(png_dir)
test_dir = os.path.join(log_dir,filename,str_p, 'test/png')
if not os.path.exists(test_dir):
    os.makedirs(test_dir)
logger = get_logger(os.path.join(log_dir,filename,str_p)+'/log.txt')
codepath=os.path.join(log_dir,filename,str_p,'code')
if not os.path.exists(codepath):
    os.makedirs(codepath)
shutil.copyfile(sys.argv[0],os.path.join(codepath,sys.argv[0]))
shutil.copyfile('dataset.py',os.path.join(codepath,'dataset.py'))
shutil.copytree('models',os.path.join(codepath,'models'))
# sampler=samplers.MPerClassSampler()
train_dataset=MultiFakeDataset(config.data_path, 'train', trans=trans, dataset='Deepfakes', augment=augmentation,image_size=224)
jsonpaths=['jsons/ff++-Deepfakes-raw.json','jsons/ff++-Face2Face-raw.json','jsons/ff++-FaceShifter-raw.json','jsons/ff++-FaceSwap-raw.json','jsons/ff++-NeuralTextures-raw.json','jsons/celebdfv1.json','jsons/celebdfv2.json','jsons/dfdc.json','jsons/dfd.json']
jsonpaths=['jsons/ff++-Deepfakes-raw.json','jsons/ff++-Face2Face-raw.json','jsons/ff++-FaceShifter-raw.json','jsons/ff++-FaceSwap-raw.json','jsons/ff++-NeuralTextures-raw.json','jsons/celebdfv1.json','jsons/celebdfv2.json']
jsonpaths=['jsons/cropff++-Deepfakes-raw.json','jsons/cropff++-Face2Face-raw.json','jsons/cropff++-FaceShifter-raw.json','jsons/cropff++-FaceSwap-raw.json','jsons/cropff++-NeuralTextures-raw.json','jsons/celebdfv1.json','jsons/celebdfv2.json']
# jsonpaths=['jsons/celebdf.json','jsons/dfdc.json','jsons/dfd.json']
train_loader = DataLoaderX(
        train_dataset, batch_size=batch_size, num_workers=config.workers, pin_memory=True,drop_last=True)
scaler = torch.cuda.amp.GradScaler()
# gunet_optimizer = optim.AdamW(gunet.parameters(), 1e-4, weight_decay=1e-5)
# noisenet_optimizer = optim.AdamW(noisenet.parameters(), 1e-4, weight_decay=1e-5)
gunet_optimizer=SAM(gunet.parameters(),torch.optim.SGD,lr=0.001, momentum=0.9)
logger.info('start training!')

best_loss=10
bestt1sim=0
bestt2sim=0
best_auc=0
for epoch in range(config.epochs):
    val_sacc = []
    val_acc = []
    val_auc=[]
    t1sims=[]
    t2sims=[]
    train_loss=[]

    gunet.train()
    with tqdm(train_loader, desc='Batch') as bar:
        for b, batch in enumerate(bar):
            fakeimg, realimg, masklabel,mask = batch
            # print(fakeimg.shape)
            anchorimg=torch.cat([fakeimg,realimg],dim=0)
            label=torch.tensor([1]*fakeimg.shape[0]+[0]*realimg.shape[0])
            initmasklabel=torch.cat((masklabel.clone(),torch.zeros_like(masklabel)),dim=0)
            masklabel=masklabel.reshape(masklabel.shape[0]*masklabel.shape[1])
            realmask=torch.zeros_like(masklabel)
            masklabel=torch.cat((masklabel,realmask),dim=0)
            anchorimg = anchorimg.cuda(non_blocking=True)
            masklabel = masklabel.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            initmasklabel = initmasklabel.cuda(non_blocking=True)
            # print(label)
            select=(label==0)
            # print(select.shape)
            label = F.one_hot(label, 2).float()
            masklabel = F.one_hot(masklabel,2).float()

            gunet_optimizer.zero_grad()
            # with torch.cuda.amp.autocast():
            for i in range(2):
                pred,maskpred,attn=gunet(anchorimg)
                pred=nn.Softmax(dim=1)(pred)
                if i==0:
                    pred_first=pred
                # print(pred)
                clsssloss=nn.BCELoss()(pred,label)
                maskpred=nn.Softmax(dim=1)(maskpred)
                attn=nn.Softmax(dim=1)(attn)
                maskloss=nn.BCELoss()(maskpred.float(),masklabel)
                attenloss=nn.L1Loss()(attn.float(),initmasklabel)
                loss = clsssloss+maskloss#+attenloss 
                # loss=clsssloss
                gunet_optimizer.zero_grad()
                loss.backward()
                if i==0:
                    gunet_optimizer.first_step(zero_grad=True)
                else:
                    gunet_optimizer.second_step(zero_grad=True)

            
            out = torch.argmax(pred_first.data, 1)
            label1 = torch.argmax(label.data, 1)
            batch_acc = torch.sum(out == label1).item() / len(out)

            try:
                auc=roc_auc_score(label.cpu(),pred.detach().cpu())
            except:
                auc=0.5
            bar.set_postfix(
                acc=batch_acc,
                maskloss=maskloss.item(),
                clsloss=clsssloss.item(),
                loss=loss.item(),
                auc=auc
            )  

            val_acc.append(batch_acc)
            val_auc.append(auc)
            train_loss.append(loss.item())
            if b%200==0:
                visualization = Visualizer(kp_size=5,draw_border= True, colormap='gist_rainbow').visualize(realimg=realimg, fakeimg=fakeimg, mask=mask)
                imageio.imsave(os.path.join(png_dir, f'{str(epoch)}_{b}.jpg'), (255 * visualization).astype(np.uint8))

    epoch_acc = np.mean(val_acc)
    epoch_auc = np.mean(val_auc)
    epoch_loss=np.mean(np.array(train_loss))
    logger.info('Epoch:[{}/{}]\t "Train Epoch Loss={:.5f}\t Train Epoch Acc={:.5f}\t Train Epoch Auc={:.5f}'.format(epoch , 50, epoch_loss, epoch_acc,epoch_auc))
    epoch_loss=0
    epoch_aucs=0
    index=0
    for jsonpath in jsonpaths:
        if 'cropff' in jsonpath:
            split='test'
        else:
            split='val'
        validate_dataset=DFDataset(config.data_path, split, trans=trans, augment=augmentation, jsonpath=jsonpath,image_size=224)
        validate_loader = DataLoaderX(
                validate_dataset, batch_size=batch_size, num_workers=config.workers, pin_memory=True,drop_last=True)
        epoch_loss2, epoch_acc, epoch_auc=eval1(validate_loader)
        # if index<=4:
        #     name=jsonpath.split('-')[1]
        # else:
        name=jsonpath.split('/')[-1]
        logger.info('Test Dataset:[{}]\t "FF: Val Epoch Loss={:.5f}\t Val Epoch Acc={:.5f}\t Val Epoch Auc={:.5f}'.format(name, epoch_loss2, epoch_acc,epoch_auc))
        epoch_loss+=epoch_loss2
        epoch_aucs+=epoch_auc
        index+=1
    
    if epoch_loss < best_loss or epoch_aucs> best_auc:
        best_loss = epoch_loss
        best_auc = epoch_aucs
        ckpt_path = os.path.join(log_dir,filename,str_p,'binet_' +str(epoch)+"_ckpt-%s.pth" % (epoch_loss))
        save_checkpoint(
            ckpt_path.replace('binet','detection'),
            gunet.state_dict(),
            epoch=epoch + 1,
            acc1=epoch_acc)
    
