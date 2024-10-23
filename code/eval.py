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
from dataset import IdentitySwap_EvalDataset,IdentityCelebaMegaFSDataset,IdentityFFMegaFSDataset,IdentitySwap_EvalDataset,IdentityFFMegaFSDataset
from dataset import IdentityVGGFACESimswapDataset
from torch import nn, optim
import net_conf as config
import datetime
from tqdm import tqdm
import imageio
import segmentation_models_pytorch as smp
import segmentation_models as smp
from models.unet import SourceRecoverNet_Attention2,TargetRecoverNet,SourceRecoverNet
from models.utils import Vgg19,get_logger
from logger import Logger, Visualizer
from models.mae import MAE22
from discriminator import Discriminator
from utils.comparefeature_ffhq import search
from face_attribute.test import attributefeatures
from strong_transform import augmentation, trans
from prefetch_generator import BackgroundGenerator
from sklearn.metrics import roc_auc_score

from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance

# from facenet_pytorch import InceptionResnetV1

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


def buildmodel():
    aux_params=dict(
        pooling='avg',             # one of 'avg', 'max'
        dropout=0.5,               # dropout ratio, default is None
        activation=None,      # activation function, default is None
        classes=2,                 # define number of output labels
    )
    unet = smp.DeepLabV3Plus(
        encoder_name="efficientnet-b0",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pretrained weights for encoder initialization                    
        classes=1,
        activation='sigmoid',
        aux_params=aux_params
    )
    return unet
def eval1(validate_dataset):
    fidf = FrechetInceptionDistance(normalize=True).cuda()
    kidf = KernelInceptionDistance(normalize=True).cuda()
    iscore = InceptionScore(normalize=True).cuda()
    torch.autograd.set_grad_enabled(False)
    gunet.eval()
    
    val_sacc = []
    val_acc = []
    val_auc=[]
    t1sims=[]
    t2sims=[]
    with tqdm(validate_dataset, desc='Batch') as bar:
        for b, batch in enumerate(bar):
            anchorimg,targetimg1,targetimg2,mask,label = batch
            anchorimg = anchorimg.cuda(non_blocking=True)
            targetimg1 = targetimg1.cuda(non_blocking=True)
            targetimg2 = targetimg2.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            
            t2=gunet(anchorimg)
            
            source_id,target_id=facemodel(targetimg1),facemodel(targetimg2)
            source_id_rec,target_id_rec=facemodel(t2),facemodel(t2)
            
            t1sim=nn.CosineSimilarity()(target_id[-1],target_id_rec[-1]).mean()
            t2sim=nn.CosineSimilarity()(target_id[-1],target_id_rec[-1])
            t2simmean=t2sim.mean()
            anchor_id=facemodel(anchorimg)
            t3sim=nn.CosineSimilarity()(anchor_id[-1],target_id_rec[-1])

            out=torch.tensor(t3sim<0.9).float()
            sacc=torch.sum(out==label[:,1])/label.shape[0]
            
            kidf.update(targetimg1, real=True)
            kidf.update(t2, real=False)

            fidf.update(targetimg1, real=True)
            fidf.update(t2, real=False)
           
            
            acc = sacc
            
            bar.set_postfix(
                sacc=sacc.item(),
                acc=acc.item(),
                t1sim=t1sim.item(),
                t2sim=t2simmean.item()
            )
            t1sims.append(t1sim.item())
            t2sims.append(t2simmean.item())
            val_acc.append(acc.item())
            val_sacc.append(sacc.item())
            
    torch.autograd.set_grad_enabled(True)
    tgtkid = 0
    tgtfid = fidf.compute()
    epoch_acc = np.mean(val_acc)
    epoch_sacc = np.mean(val_sacc)
    epoch_t1sim=np.mean(np.array(t1sims))
    epoch_t2sim=np.mean(np.array(t2sims))
    fids=tgtfid.item()
    kids=0
    return epoch_t1sim,epoch_t2sim,fids,kids,epoch_acc,epoch_sacc

def evalfake(validate_dataset):
    fidf = FrechetInceptionDistance(normalize=True).cuda()
    kidf = KernelInceptionDistance(normalize=True).cuda()
    iscore = InceptionScore(normalize=True).cuda()
    torch.autograd.set_grad_enabled(False)
    gunet.eval()
    
    val_sacc = []
    val_acc = []
    val_auc=[]
    t1sims=[]
    t2sims=[]
    with tqdm(validate_dataset, desc='Batch') as bar:
        for b, batch in enumerate(bar):
            anchorimg,targetimg1,targetimg2,mask,label = batch
            anchorimg = anchorimg.cuda(non_blocking=True)
            targetimg1 = targetimg1.cuda(non_blocking=True)
            targetimg2 = targetimg2.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            fakelabel=torch.tensor([[0,1]]*anchorimg.shape[0]).float().cuda()
            select=(label[:,1]==fakelabel[:,1])
            anchorimg=anchorimg[select]
            targetimg1=targetimg1[select]
            targetimg2=targetimg2[select]
            label=label[select]
            t2=gunet(anchorimg)
            
            source_id,target_id=facemodel(targetimg1),facemodel(targetimg2)
            source_id_rec,target_id_rec=facemodel(t2),facemodel(t2)
            
            t1sim=nn.CosineSimilarity()(target_id[-1],target_id_rec[-1]).mean()
            t2sim=nn.CosineSimilarity()(target_id[-1],target_id_rec[-1])
            t2simmean=t2sim.mean()
            anchor_id=facemodel(anchorimg)
            t3sim=nn.CosineSimilarity()(anchor_id[-1],target_id_rec[-1])

            out=torch.tensor(t3sim<0.9).float()
            sacc=torch.sum(out==label[:,1])/label.shape[0]
            
            kidf.update(targetimg1, real=True)
            kidf.update(t2, real=False)

            fidf.update(targetimg1, real=True)
            fidf.update(t2, real=False)
           
            
            acc = sacc
            bar.set_postfix(
                sacc=sacc.item(),
                acc=acc.item(),
                t1sim=t1sim.item(),
                t2sim=t2simmean.item()
            )
            t1sims.append(t1sim.item())
            t2sims.append(t2simmean.item())
            val_acc.append(acc.item())
            val_sacc.append(sacc.item())
            
    torch.autograd.set_grad_enabled(True)
    tgtkid = 0
    tgtfid = fidf.compute()
    epoch_acc = np.mean(val_acc)
    epoch_sacc = np.mean(val_sacc)
    epoch_t1sim=np.mean(np.array(t1sims))
    epoch_t2sim=np.mean(np.array(t2sims))
    fids=tgtfid.item()
    kids=0
    return epoch_t1sim,epoch_t2sim,fids,kids,epoch_acc,epoch_sacc


from facenet_pytorch import InceptionResnetV1,MTCNN
facemodel = InceptionResnetV1(pretrained='vggface2').cuda().eval()
modelname='deeplabv3plus'
gunet=SourceRecoverNet().cuda()
torch.autograd.set_grad_enabled(False)



validate_dataset1 = IdentityCelebaMegaFSDataset(
    config.data_path, 'val', trans=trans, augment=augmentation)
validate_dataset2 = IdentityCelebaMegaFSDataset(
    config.data_path, 'val', trans=trans, augment=augmentation,d='IDInjection')
validate_dataset3 = IdentityCelebaMegaFSDataset(
    config.data_path, 'val', trans=trans, augment=augmentation,d='LCR')
validate_dataset4 = IdentitySwap_EvalDataset(
    config.data_path, 'val', trans=trans, augment=augmentation, dataset='Deepfakes')
validate_dataset5 = IdentityFFMegaFSDataset(
    config.data_path, 'val', trans=trans, augment=augmentation)   
validate_dataset6 = IdentitySwap_EvalDataset(
    config.data_path, 'val', trans=trans, augment=augmentation, dataset='FaceShifter')
# validate_dataset = IdentityVGGFACESimswapDataset(
#     config.data_path, 'val', trans=trans, augment=augmentation)

deepfakemodel='logs/compare/DeepReversion/train.py/2024-03-28 13:52:10.723430/deeplabv3plus_splitidentity_detection_real_landmarks_11_ckpt-0.01911842012732009.pth'
faceshiftermodel='logs/compare/DeepReversion/train-fs.py/2024-03-29 12:50:10.450359/deeplabv3plus_splitidentity_detection_real_landmarks_2_ckpt-0.019789362627394565.pth'
ffmegamodel='logs/compare/DeepReversion/train-ffmega.py/2024-03-29 19:17:29.199320/deeplabv3plus_splitidentity_detection_real_landmarks_3_ckpt-0.01913473193001534.pth'
ftmmodel='logs/compare/DeepReversion/train-ftm.py/2024-03-30 13:38:08.671464/deeplabv3plus_splitidentity_detection_real_landmarks_18_ckpt-0.019871618395591987.pth'
idinjectionmodel='logs/compare/DeepReversion/train-idinject.py/2024-03-30 15:59:58.442368/deeplabv3plus_splitidentity_detection_real_landmarks_11_ckpt-0.019280878519785653.pth'
lcrmodel='logs/compare/DeepReversion/train-lcr.py/2024-03-30 18:22:18.951830/deeplabv3plus_splitidentity_detection_real_landmarks_8_ckpt-0.021324508947300466.pth'
batch_size=16
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.enabled = False
# validate_dataset3=validate_dataset3[:32]

modelpath=[deepfakemodel,faceshiftermodel,ffmegamodel,ftmmodel,idinjectionmodel,lcrmodel]
i=0
validate_datasets=[validate_dataset4,validate_dataset6,validate_dataset5,validate_dataset1,validate_dataset2,validate_dataset3]

for i in range(3,6):
    validate_loader = DataLoaderX(
        validate_datasets[i], batch_size=batch_size, num_workers=config.workers, pin_memory=True)
    gunet=load_model(gunet,modelpath[i])
    # gunet = nn.DataParallel(gunet.cuda())
    # recovery1 = nn.DataParallel(recovery1.cuda())
    # recovery2 = nn.DataParallel(recovery2.cuda())
    gunet.eval()
    
    # epoch_loss1,epoch_acc1,epoch_auc1,epoch_t1simd,epoch_t2simd=eval(Deepfakes_validate_loader)
    epoch_t1simf,epoch_t2simf,fid,kid,epoch_acc,epoch_sacc=eval1(validate_loader)
    print('Epoch:[{}/{}]\t "ALL: Val Epoch t1sim={:.5f}\t Val Epoch t2sim={:.5f} \t Val FID={:.5f} \t Val KID={:.5f}\t Val ACC={:.5f}\t Val SACC={:.5f}'.format(0 , config.epochs, epoch_t1simf,epoch_t2simf,fid,kid,epoch_acc,epoch_sacc))
    epoch_t1simf,epoch_t2simf,fid,kid,epoch_acc,epoch_sacc=evalfake(validate_loader)
    print('Epoch:[{}/{}]\t "FAKE: Val Epoch t1sim={:.5f}\t Val Epoch t2sim={:.5f} \t Val FID={:.5f} \t Val KID={:.5f}\t Val ACC={:.5f}\t Val SACC={:.5f}'.format(0 , config.epochs, epoch_t1simf,epoch_t2simf,fid,kid,epoch_acc,epoch_sacc))
