import os
import cv2
import random
import json
from PIL import Image
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from faker.generate import blended_face,gen_one_datapoint
from utils.dataaug import fourier_content_perturbation,fourier_style_perturbation
from imutils import face_utils
from faker.generate import random_get_hull
import dlib
import net_conf as config
from funcs import IoUfrom2bboxes,crop_face,RandomDownScale
import logging
import albumentations as alb
import blend as B
import patchify
# import anomalib.post_processing
from compdataset import randomaug
class DFDataset(Dataset):
    def __init__(self, datapath, dataselect, trans, augment, jsonpath,image_size=320):
        self.datapath = datapath
        self.trans = trans
        self.aug = augment
        self.num_classes = 2
        self.jsonpath = jsonpath
        self.dataselect = dataselect
        self.image_size=(image_size,image_size)
        self.detector = dlib.get_frontal_face_detector()    #使用dlib库提供的人脸提取器
        self.predictor = dlib.shape_predictor('/home/peipeng/Detection/DA-SBI/preprocess/shape_predictor_81_face_landmarks.dat')
        self.landmarks_json='/home/peipeng/Detection/splitidentityv2/jsons/landmarks.json'
        self.next_epoch()

    def next_epoch(self):
        with open(self.landmarks_json, 'r') as f:
            self.landmarks_record =  json.load(f)
        for k,v in self.landmarks_record.items():
            self.landmarks_record[k] = np.array(v)

        with open(self.jsonpath, 'r') as f:
            data = json.load(f)
        if self.dataselect == 'train':
            trainset = data['train']
            
            self.dataset = trainset
        if self.dataselect == 'val':
            valset = data['val']
            
            print(len(valset))
            self.dataset = valset
        if self.dataselect == 'test':
            testset = data['test']
            
            print(len(testset))
            self.dataset = testset
        # self.dataset=self.dataset[:160]+self.dataset[-160:]
        random.shuffle(self.dataset)

    def __getitem__(self, item):
        flag=True
        while flag:
            try:
                sample = self.dataset[item]
                # print(sample)
                # try:
                anchor, label = sample
                # print(anchor)
                anchorimg = np.array(Image.open(anchor))
                # anchorimg=anchorimg[20:300,20:300,:]
                rects = self.detector(np.array(anchorimg), 0)
                # print(anchor,len(rects))
                landmark = self.predictor(np.array(anchorimg),rects[0])
                landmark = face_utils.shape_to_np(landmark)
                # landmark= self.reorder_landmark(landmark)
                # print(landmark.shape)
                anchorimg,landmark,_,__=crop_face(anchorimg,landmark,margin=True,crop_by_bbox=False)
                anchorimg=cv2.resize(anchorimg,self.image_size,interpolation=cv2.INTER_LINEAR).astype('float32')/255
                anchorimg=anchorimg.transpose((2,0,1))
                label=torch.tensor(label).long()
                flag=False
            except:
                item=(item+1)%len(self.dataset)

        return anchorimg, label

    def reorder_landmark(self,landmark):
        landmark_add=np.zeros((13,2))
        for idx,idx_l in enumerate([77,75,76,68,69,70,71,80,72,73,79,74,78]):
            landmark_add[idx]=landmark[idx_l]
        landmark[68:]=landmark_add
        return landmark
    def __len__(self):
        return len(self.dataset)

class MultiFakeDataset(Dataset):
    def __init__(self, datapath, dataselect, trans, augment, dataset,image_size=320):
        self.datapath = datapath
        self.trans = trans
        self.aug = augment
        self.num_classes = 2
        self.datasetname=dataset
        if dataset=='Deepfakes':
            self.jsonpath = 'jsons/cropff++-Deepfakes-raw.json'
        if dataset=='FaceShifter':
            self.jsonpath = 'jsons/cropff++-FaceShifter-raw.json'
        self.dataselect = dataselect
        self.patchsize=32
        self.landmarks_json='jsons/landmarks_dlib_retina_croprand.json'
        self.transforms=self.get_transforms()
        self.source_transforms = self.get_source_transforms()
        self.image_size=(image_size,image_size)
        self.datapath='/mnt/UserData1/peipeng/Data/faceforensics_img_lmkcroprandom/'
        self.targetpath='/mnt/UserData1/peipeng/Data/faceforensics_rec/AE/'
        self.next_epoch()
        
    def next_epoch(self):
        with open(self.landmarks_json, 'r') as f:
            landmarks = json.load(f)
            self.landmarks=landmarks
        with open(self.jsonpath, 'r') as f:
            data = json.load(f)
        if self.dataselect == 'train':
            trainset = data['train']
            self.dataset = [i for i in trainset if i[1]==0]
            self.dataset =[i for i in self.dataset if i[0] in self.landmarks.keys()]
            print('traindataset',len(self.dataset))
        if self.dataselect == 'val':
            valset = data['val']
            print(len(valset))
            self.dataset = valset
        if self.dataselect == 'test':
            testset = data['test']
            self.dataset = testset
            print('testset',len(self.dataset)) 
        random.shuffle(self.dataset)
        self.data_list = self.dataset
        # self.dataset=self.dataset[:100]  + self.dataset[-100:]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        
        flag=True
        while flag:
            try:
                sample,label = self.dataset[item]
                item=(item+1)%len(self.dataset)
                img=np.array(Image.open(sample))
                if self.dataselect=='train':
                    if np.random.rand()<0.5:
                        img=self.hflip(img)
                mask=self.genmask(img,self.image_size[0],self.image_size[0]//4)       
                img_r,img_f,mask_f=self.self_blending(img.copy(),mask)
                # print(mask_f.shape)
                if self.dataselect=='train':
                    transformed=self.transforms(image=img_f.astype('uint8'),image1=img_r.astype('uint8'))
                    img_f=transformed['image']
                    img_r=transformed['image1']

                img_f=cv2.resize(img_f,self.image_size,interpolation=cv2.INTER_LINEAR).astype('float32')/255
                img_r=cv2.resize(img_r,self.image_size,interpolation=cv2.INTER_LINEAR).astype('float32')/255
                mask_f=cv2.resize(mask_f,self.image_size,interpolation=cv2.INTER_LINEAR).astype('float32')

                patches2=patchify.patchify(mask_f,(32,32),32)
                rp=patches2.reshape((patches2.shape[0],patches2.shape[1],32*32))
                rp=rp.reshape((patches2.shape[0]*patches2.shape[1],32*32))
                rp=rp.sum(axis=1)
                masklabel=np.array(rp!=0,dtype=np.int64)
                # print(masklabel)
                img_f=img_f.transpose((2,0,1))
                img_r=img_r.transpose((2,0,1))
                mask_f=mask_f.reshape((1,)+self.image_size)
                flag=False
            
                # print(mask_f.shape)
            except:
                # print('error')
                continue
        return img_f, img_r,masklabel,mask_f

    def genmask(self, img1,imgsize,patchsize=32):
        img1=cv2.resize(img1,(imgsize,imgsize))
        h,w=img1.shape[:2]
        mask=np.zeros_like(img1)[:,:,0]
        patches2=patchify.patchify(mask,(patchsize,patchsize),patchsize)
        for i in range(patches2.shape[0]):
            for j in range(patches2.shape[1]):
                if random.randint(0,2)!=0:
                    p1=np.zeros((patchsize,patchsize))
                    n=random.randint(3,20)
                    merged=np.random.randint(0,patchsize,(n,2))
                    cv2.fillConvexPoly(p1, cv2.convexHull(merged), 255.)
                    patches2[i,j,:,:]=p1
        mask2=patchify.unpatchify(patches2,(imgsize,imgsize))/255
        return mask2

    def get_source_transforms(self):
        return alb.Compose([
                alb.Compose([
                        alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
                        alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=1),
                        alb.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1,0.1), p=1),
                    ],p=1),
    
                alb.OneOf([
                    RandomDownScale(p=1),
                    alb.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
                ],p=1),
                
            ], p=1.)

        
    def get_transforms(self):
        return alb.Compose([
            
            alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
            alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=0.3),
            alb.RandomBrightnessContrast(brightness_limit=(-0.3,0.3), contrast_limit=(-0.3,0.3), p=0.3),
            alb.ImageCompression(quality_lower=40,quality_upper=100,p=0.5),
            
        ], 
        additional_targets={f'image1': 'image'},
        p=1.)


    def randaffine(self,img,mask):
        f=alb.Affine(
                translate_percent={'x':(-0.03,0.03),'y':(-0.015,0.015)},
                scale=[0.95,1/0.95],
                fit_output=False,
                p=1)
            
        g=alb.ElasticTransform(
                alpha=50,
                sigma=7,
                alpha_affine=0,
                p=1,
            )

        transformed=f(image=img,mask=mask)
        img=transformed['image']
        
        mask=transformed['mask']
        transformed=g(image=img,mask=mask)
        mask=transformed['mask']
        return img,mask

        
    def self_blending(self,img,mask=None):
        H,W=len(img),len(img[0])

        source = img.copy()
        # if img2 is not None:
        #     source= img2
        if np.random.rand()<0.5:
            source = self.source_transforms(image=source.astype(np.uint8))['image']
        else:
            img = self.source_transforms(image=img.astype(np.uint8))['image']

        source, mask = self.randaffine(source,mask)

        img_blended,mask=B.dynamic_blend(source,img,mask)
        img_blended = img_blended.astype(np.uint8)
        img = img.astype(np.uint8)

        return img,img_blended,mask
    
    def reorder_landmark(self,landmark):
        landmark_add=np.zeros((13,2))
        for idx,idx_l in enumerate([77,75,76,68,69,70,71,80,72,73,79,74,78]):
            landmark_add[idx]=landmark[idx_l]
        landmark[68:]=landmark_add
        return landmark

    def hflip(self,img,mask=None):
        H,W=img.shape[:2]
        
        # if mask is not None:
        #     mask=mask[:,::-1]
        # else:
        #     mask=None
        img=img[:,::-1].copy()
        return img
    
    def collate_fn(self,batch):
        img_f,img_r=zip(*batch)
        data={}
        data['img']=torch.cat([torch.tensor(img_r).float(),torch.tensor(img_f).float()],0)
        data['label']=torch.tensor([0]*len(img_r)+[1]*len(img_f))
        return data
        

    def worker_init_fn(self,worker_id):                                                          
        np.random.seed(np.random.get_state()[1][0] + worker_id)


class FFDADataset(Dataset):
    def __init__(self, datapath, dataselect, trans, augment, dataset,image_size=320):
        self.datapath = datapath
        self.trans = trans
        self.aug = augment
        self.num_classes = 2
        self.datasetname=dataset
        if dataset=='Deepfakes':
            self.jsonpath = 'jsons/ff++-Deepfakes-raw.json'
        if dataset=='FaceShifter':
            self.jsonpath = 'jsons/ff++-FaceShifter-raw.json'
        self.dataselect = dataselect
        self.patchsize=32
        self.landmarks_json='jsons/landmarks_dlib_retina.json'
        self.transforms=self.get_transforms()
        self.source_transforms = self.get_source_transforms()
        self.image_size=(image_size,image_size)
        self.datapath='/mnt/UserData1/peipeng/Data/faceforensics_imgs/'
        self.targetpath='/mnt/UserData1/peipeng/Data/faceforensics_rec/AE/'
        self.deviantpath='/mnt/UserData1/peipeng/Data/deviantart/'
        self.next_epoch()
        
    def next_epoch(self):
        with open(self.landmarks_json, 'r') as f:
            landmarks = json.load(f)
            self.landmarks=landmarks
        with open(self.jsonpath, 'r') as f:
            data = json.load(f)
        if self.dataselect == 'train':
            trainset = data['train']
            self.dataset = [i for i in trainset if i[1]==0]
            self.dataset =[i for i in self.dataset if i[0] in self.landmarks.keys()]
            print('traindataset',len(self.dataset))
        if self.dataselect == 'val':
            valset = data['val']
            print(len(valset))
            self.dataset = valset
        if self.dataselect == 'test':
            testset = data['test']
            self.dataset = testset
            print('testset',len(self.dataset)) 
        random.shuffle(self.dataset)
        self.data_list = self.dataset
        # self.dataset=self.dataset[:100]  + self.dataset[-100:]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        
        flag=True
        while flag:
            try:
                sample,label = self.dataset[item]
                item=(item+1)%len(self.dataset)
                img=np.array(Image.open(sample))
                img2=np.array(Image.open(sample.replace(self.datapath,self.targetpath)).resize((img.shape[1],img.shape[0])))
                landmark,bboxes=self.landmarks[sample]
                landmark,bboxes=np.array(landmark)[0],np.array(bboxes)
                # print(landmark.shape,bboxes.shape)
                bbox_lm=np.array([landmark[:,0].min(),landmark[:,1].min(),landmark[:,0].max(),landmark[:,1].max()])
                iou_max=-1
                for i in range(len(bboxes)):
                    iou=IoUfrom2bboxes(bbox_lm,bboxes[i].flatten())
                    if iou_max<iou:
                        bbox=bboxes[i]
                        iou_max=iou

                landmark=self.reorder_landmark(landmark)
                if self.dataselect=='train':
                    if np.random.rand()<0.5:
                        img,_,landmark,bbox=self.hflip(img,None,landmark,bbox)
                        # alpha=0.2
                        # randimgpath=self.deviantpath+random.sample(os.listdir(self.deviantpath),1)[0]
                        # randimg=np.array(Image.open(randimgpath).convert('RGB').resize(self.image_size)).astype('float32')
                        # img = (1 - alpha) * img + alpha * randimg

                mask=None
                img_r,img_f,mask_f=self.self_blending(img.copy(),landmark.copy(),mask=mask)
                # print(mask_f.shape)
                if self.dataselect=='train':
                    transformed=self.transforms(image=img_f.astype('uint8'),image1=img_r.astype('uint8'))
                    img_f=transformed['image']
                    img_r=transformed['image1']

                img_f,_,__,___,y0_new,y1_new,x0_new,x1_new=crop_face(img_f,landmark,margin=False,crop_by_bbox=False,abs_coord=True,phase=self.dataselect)
                
                img_r=img_r[y0_new:y1_new,x0_new:x1_new]
                mask_f=mask_f[y0_new:y1_new,x0_new:x1_new]


                # img_f,img_r,mask_f=img_f[20:300,20:300,:],img_r[20:300,20:300,:],mask_f[20:300,20:300]
                img_f=cv2.resize(img_f,self.image_size,interpolation=cv2.INTER_LINEAR).astype('float32')/255
                img_r=cv2.resize(img_r,self.image_size,interpolation=cv2.INTER_LINEAR).astype('float32')/255
                # if self.dataselect=='train':
                #     alpha=0.2
                #     randimgpath=self.deviantpath+random.sample(os.listdir(self.deviantpath),1)[0]
                #     randimg=np.array(Image.open(randimgpath).convert('RGB').resize(self.image_size)).astype('float32')
                #     img_f = (1 - alpha) * img_f + alpha * randimg/255
                #     randimgpath=self.deviantpath+random.sample(os.listdir(self.deviantpath),1)[0]
                #     randimg=np.array(Image.open(randimgpath).convert('RGB').resize(self.image_size)).astype('float32')
                #     img_r = (1 - alpha) * img_r + alpha * randimg/255

                mask_f=cv2.resize(mask_f,self.image_size,interpolation=cv2.INTER_LINEAR).astype('float32')
                patches2=patchify.patchify(mask_f,(32,32),32)
                rp=patches2.reshape((patches2.shape[0],patches2.shape[1],32*32))
                rp=rp.reshape((patches2.shape[0]*patches2.shape[1],32*32))
                rp=rp.sum(axis=1)
                masklabel=np.array(rp!=0,dtype=np.int64)
                # print(masklabel)
                img_f=img_f.transpose((2,0,1))
                img_r=img_r.transpose((2,0,1))
                mask_f=mask_f.reshape((1,)+self.image_size)
                flag=False
    
                # print(mask_f.shape)
            except:
                # print('error')
                continue

        return img_f, img_r,masklabel,mask_f

    def genmask(self, img1,imgsize,patchsize=64):
        img1=cv2.resize(img1,(imgsize,imgsize))
        h,w=img1.shape[:2]
        mask=np.zeros_like(img1)[:,:,0]
        patches2=patchify.patchify(mask,(patchsize,patchsize),patchsize)
        for i in range(patches2.shape[0]):
            for j in range(patches2.shape[1]):
                if random.randint(0,2)!=0:
                    p1=np.zeros((patchsize,patchsize))
                    n=random.randint(3,20)
                    merged=np.random.randint(0,patchsize,(n,2))
                    cv2.fillConvexPoly(p1, cv2.convexHull(merged), 255.)
                    patches2[i,j,:,:]=p1
        mask2=patchify.unpatchify(patches2,(imgsize,imgsize))/255
        return mask2


    def get_source_transforms(self):
        return alb.Compose([
                alb.Compose([
                        alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
                        alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=1),
                        alb.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1,0.1), p=1),
                    ],p=1),
    
                alb.OneOf([
                    RandomDownScale(p=1),
                    alb.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
                ],p=1),
                
            ], p=1.)

        
    def get_transforms(self):
        return alb.Compose([
            
            alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
            alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=0.3),
            alb.RandomBrightnessContrast(brightness_limit=(-0.3,0.3), contrast_limit=(-0.3,0.3), p=0.3),
            alb.ImageCompression(quality_lower=40,quality_upper=100,p=0.5),
            
        ], 
        additional_targets={f'image1': 'image'},
        p=1.)


    def randaffine(self,img,mask):
        f=alb.Affine(
                translate_percent={'x':(-0.03,0.03),'y':(-0.015,0.015)},
                scale=[0.95,1/0.95],
                fit_output=False,
                p=1)
            
        g=alb.ElasticTransform(
                alpha=50,
                sigma=7,
                alpha_affine=0,
                p=1,
            )

        transformed=f(image=img,mask=mask)
        img=transformed['image']
        
        mask=transformed['mask']
        transformed=g(image=img,mask=mask)
        mask=transformed['mask']
        return img,mask

        
    def self_blending(self,img,landmark,img2=None,mask=None):
        H,W=len(img),len(img[0])
        if np.random.rand()<0.25:
            landmark=landmark[:68]
        # if exist_bi:
        logging.disable(logging.FATAL)
        if mask is None:
            mask=random_get_hull(landmark,img)[:,:,0]
        logging.disable(logging.NOTSET)
        # else:
        #     mask=np.zeros_like(img[:,:,0])
        #     cv2.fillConvexPoly(mask, cv2.convexHull(landmark), 1.)


        source = img.copy()
        if img2 is not None:
            source= img2
        if np.random.rand()<0.5:
            source = self.source_transforms(image=source.astype(np.uint8))['image']
        else:
            img = self.source_transforms(image=img.astype(np.uint8))['image']

        source, mask = self.randaffine(source,mask)

        img_blended,mask=B.dynamic_blend(source,img,mask)
        img_blended = img_blended.astype(np.uint8)
        img = img.astype(np.uint8)

        return img,img_blended,mask
    
    def reorder_landmark(self,landmark):
        landmark_add=np.zeros((13,2))
        for idx,idx_l in enumerate([77,75,76,68,69,70,71,80,72,73,79,74,78]):
            landmark_add[idx]=landmark[idx_l]
        landmark[68:]=landmark_add
        return landmark

    def hflip(self,img,mask=None,landmark=None,bbox=None):
        H,W=img.shape[:2]
        landmark=landmark.copy()
        bbox=bbox.copy()

        if landmark is not None:
            landmark_new=np.zeros_like(landmark)

            
            landmark_new[:17]=landmark[:17][::-1]
            landmark_new[17:27]=landmark[17:27][::-1]

            landmark_new[27:31]=landmark[27:31]
            landmark_new[31:36]=landmark[31:36][::-1]

            landmark_new[36:40]=landmark[42:46][::-1]
            landmark_new[40:42]=landmark[46:48][::-1]

            landmark_new[42:46]=landmark[36:40][::-1]
            landmark_new[46:48]=landmark[40:42][::-1]

            landmark_new[48:55]=landmark[48:55][::-1]
            landmark_new[55:60]=landmark[55:60][::-1]

            landmark_new[60:65]=landmark[60:65][::-1]
            landmark_new[65:68]=landmark[65:68][::-1]
            if len(landmark)==68:
                pass
            elif len(landmark)==81:
                landmark_new[68:81]=landmark[68:81][::-1]
            else:
                raise NotImplementedError
            landmark_new[:,0]=W-landmark_new[:,0]
            
        else:
            landmark_new=None

        if bbox is not None:
            bbox_new=np.zeros_like(bbox)
            bbox_new[0,0]=bbox[1,0]
            bbox_new[1,0]=bbox[0,0]
            bbox_new[:,0]=W-bbox_new[:,0]
            bbox_new[:,1]=bbox[:,1].copy()
            if len(bbox)>2:
                bbox_new[2,0]=W-bbox[3,0]
                bbox_new[2,1]=bbox[3,1]
                bbox_new[3,0]=W-bbox[2,0]
                bbox_new[3,1]=bbox[2,1]
                bbox_new[4,0]=W-bbox[4,0]
                bbox_new[4,1]=bbox[4,1]
                bbox_new[5,0]=W-bbox[6,0]
                bbox_new[5,1]=bbox[6,1]
                bbox_new[6,0]=W-bbox[5,0]
                bbox_new[6,1]=bbox[5,1]
        else:
            bbox_new=None

        if mask is not None:
            mask=mask[:,::-1]
        else:
            mask=None
        img=img[:,::-1].copy()
        return img,mask,landmark_new,bbox_new
    
    def collate_fn(self,batch):
        img_f,img_r=zip(*batch)
        data={}
        data['img']=torch.cat([torch.tensor(img_r).float(),torch.tensor(img_f).float()],0)
        data['label']=torch.tensor([0]*len(img_r)+[1]*len(img_f))
        return data
        

    def worker_init_fn(self,worker_id):                                                          
        np.random.seed(np.random.get_state()[1][0] + worker_id)


class FFDACompDataset(Dataset):
    def __init__(self, datapath, dataselect, trans, jsonpath, augment, image_size=320):
        self.datapath = datapath
        self.trans = trans
        self.aug = augment
        self.num_classes = 2
        self.jsonpath = jsonpath
        self.dataselect = dataselect
        self.patchsize=32
        self.landmarks_json='jsons/landmarks_dlib_retina_crop_comp.json'
        self.transforms=self.get_transforms()
        self.source_transforms = self.get_source_transforms()
        self.image_size=(image_size,image_size)
        self.datapath='/mnt/UserData1/peipeng/Data/faceforensics_img_lmkcroprandom/'
        self.targetpath='/mnt/UserData1/peipeng/Data/faceforensics_rec/AE/'
        self.detector = dlib.get_frontal_face_detector()    #使用dlib库提供的人脸提取器
        self.predictor = dlib.shape_predictor('/home/peipeng/Detection/DA-SBI/preprocess/shape_predictor_81_face_landmarks.dat')
        self.next_epoch()
        
    def next_epoch(self):
        with open(self.landmarks_json, 'r') as f:
            landmarks = json.load(f)
            self.landmarks=landmarks
        with open(self.jsonpath, 'r') as f:
            data = json.load(f)
        if self.dataselect == 'train':
            trainset = data['train']
            self.dataset = [i for i in trainset if i[1]==0]
            self.fakedataset = [i for i in trainset if i[1]==1]
            self.dataset =[i for i in self.dataset if i[0] in self.landmarks.keys()]
            print('traindataset',len(self.dataset))
        if self.dataselect == 'val':
            valset = data['val']
            print(len(valset))
            self.dataset = valset
        if self.dataselect == 'test':
            testset = data['test']
            self.dataset = testset
            print('testset',len(self.dataset)) 
        random.shuffle(self.dataset)
        self.data_list = self.dataset
        # self.dataset=self.dataset[:100]  + self.dataset[-100:]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        
        flag=True
        while flag:
            try:
                if random.randint(0,1)==0:
                    sample,label = self.dataset[item]
                    item=(item+1)%len(self.dataset)
                    img=np.array(Image.open(sample))
                    img2=np.array(Image.open(sample.replace(self.datapath,self.targetpath)).resize((img.shape[1],img.shape[0])))
                    landmark,bboxes=self.landmarks[sample]
                    landmark,bboxes=np.array(landmark)[0],np.array(bboxes)
                    # print(landmark.shape,bboxes.shape)
                    bbox_lm=np.array([landmark[:,0].min(),landmark[:,1].min(),landmark[:,0].max(),landmark[:,1].max()])
                    iou_max=-1
                    for i in range(len(bboxes)):
                        iou=IoUfrom2bboxes(bbox_lm,bboxes[i].flatten())
                        if iou_max<iou:
                            bbox=bboxes[i]
                            iou_max=iou

                    landmark=self.reorder_landmark(landmark)
                    if self.dataselect=='train':
                        if np.random.rand()<0.5:
                            img,_,landmark,bbox=self.hflip(img,None,landmark,bbox)
                            # img2,_,_,_=self.hflip(img2,None,landmark,bbox)
                            
                    img_r,img_f,mask_f=self.self_blending(img.copy(),landmark.copy())
                    # print(mask_f.shape)
                    if self.dataselect=='train':
                        transformed=self.transforms(image=img_f.astype('uint8'),image1=img_r.astype('uint8'))
                        img_f=transformed['image']
                        img_r=transformed['image1']
                        
                    
                    img_f,_,__,___,y0_new,y1_new,x0_new,x1_new=crop_face(img_f,landmark,margin=False,crop_by_bbox=False,abs_coord=True,phase=self.dataselect)
                    
                    img_r=img_r[y0_new:y1_new,x0_new:x1_new]
                    # img_f,img_r,mask_f=img_f[20:300,20:300,:],img_r[20:300,20:300,:],mask_f[20:300,20:300]
                    img_f=cv2.resize(img_f,self.image_size,interpolation=cv2.INTER_LINEAR).astype('float32')/255
                    img_r=cv2.resize(img_r,self.image_size,interpolation=cv2.INTER_LINEAR).astype('float32')/255
                    mask_f=cv2.resize(mask_f,self.image_size,interpolation=cv2.INTER_LINEAR).astype('float32')
                    patches2=patchify.patchify(mask_f,(32,32),32)
                    rp=patches2.reshape((patches2.shape[0],patches2.shape[1],32*32))
                    rp=rp.reshape((patches2.shape[0]*patches2.shape[1],32*32))
                    rp=rp.sum(axis=1)
                    masklabel=np.array(rp!=0,dtype=np.int64)
                    # print(masklabel)
                    img_f=img_f.transpose((2,0,1))
                    img_r=img_r.transpose((2,0,1))
                    mask_f=mask_f.reshape((1,)+self.image_size)
                    flag=False
                else:
                    sample,label = self.dataset[item%len(self.dataset)]
                    sample2,label2 = self.fakedataset[item%len(self.fakedataset)]
                    img_r=np.array(Image.open(sample))
                    img_f=np.array(Image.open(sample2))
                    # img_r=self.compdetect(img_r)
                    # img_f=self.compdetect(img_f)
                    if self.dataselect=='train':
                        transformed=self.transforms(image=img_f.astype('uint8'),image1=img_r.astype('uint8'))
                        img_f=transformed['image']
                        img_r=transformed['image1']
                    img_f=cv2.resize(img_f,self.image_size,interpolation=cv2.INTER_LINEAR).astype('float32')/255
                    img_r=cv2.resize(img_r,self.image_size,interpolation=cv2.INTER_LINEAR).astype('float32')/255
                    img_f=img_f.transpose((2,0,1))
                    img_r=img_r.transpose((2,0,1))
                    flag=False
                # print(mask_f.shape)
            except:
                # print('error')
                continue

        return img_f, img_r

    def compdetect(self, img):
        rects = self.detector(np.array(img), 0)
        landmark = self.predictor(np.array(img),rects[0])
        landmark = face_utils.shape_to_np(landmark)
        if self.dataselect=='train':
            if np.random.rand()<0.5:
                img,_,landmark,_=self.hflip(img,None,landmark)
        cropimg,_,__,___,y0_new,y1_new,x0_new,x1_new=crop_face(img,landmark,margin=False,crop_by_bbox=False,abs_coord=True,phase=self.dataselect)
        return cropimg

    def genmask(self, img1,imgsize,patchsize=64):
        img1=cv2.resize(img1,(imgsize,imgsize))
        h,w=img1.shape[:2]
        mask=np.zeros_like(img1)[:,:,0]
        patches2=patchify.patchify(mask,(patchsize,patchsize),patchsize)
        for i in range(patches2.shape[0]):
            for j in range(patches2.shape[1]):
                if random.randint(0,2)!=0:
                    p1=np.zeros((patchsize,patchsize))
                    n=random.randint(3,20)
                    merged=np.random.randint(0,patchsize,(n,2))
                    cv2.fillConvexPoly(p1, cv2.convexHull(merged), 255.)
                    patches2[i,j,:,:]=p1
        mask2=patchify.unpatchify(patches2,(imgsize,imgsize))/255
        return mask2


    def get_source_transforms(self):
        return alb.Compose([
                alb.Compose([
                        alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
                        alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=1),
                        alb.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1,0.1), p=1),
                    ],p=1),
    
                alb.OneOf([
                    RandomDownScale(p=1),
                    alb.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
                ],p=1),
                
            ], p=1.)

        
    def get_transforms(self):
        return alb.Compose([
            
            alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
            alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=0.3),
            alb.RandomBrightnessContrast(brightness_limit=(-0.3,0.3), contrast_limit=(-0.3,0.3), p=0.3),
            alb.ImageCompression(quality_lower=40,quality_upper=100,p=0.5),
            
        ], 
        additional_targets={f'image1': 'image'},
        p=1.)


    def randaffine(self,img,mask):
        f=alb.Affine(
                translate_percent={'x':(-0.03,0.03),'y':(-0.015,0.015)},
                scale=[0.95,1/0.95],
                fit_output=False,
                p=1)
            
        g=alb.ElasticTransform(
                alpha=50,
                sigma=7,
                alpha_affine=0,
                p=1,
            )

        transformed=f(image=img,mask=mask)
        img=transformed['image']
        
        mask=transformed['mask']
        transformed=g(image=img,mask=mask)
        mask=transformed['mask']
        return img,mask

        
    def self_blending(self,img,landmark,img2=None,mask=None):
        H,W=len(img),len(img[0])
        if np.random.rand()<0.25:
            landmark=landmark[:68]
        # if exist_bi:
        logging.disable(logging.FATAL)
        if mask is None:
            mask=random_get_hull(landmark,img)[:,:,0]
        logging.disable(logging.NOTSET)
        # else:
        #     mask=np.zeros_like(img[:,:,0])
        #     cv2.fillConvexPoly(mask, cv2.convexHull(landmark), 1.)


        source = img.copy()
        if img2 is not None:
            source= img2
        if np.random.rand()<0.5:
            source = self.source_transforms(image=source.astype(np.uint8))['image']
        else:
            img = self.source_transforms(image=img.astype(np.uint8))['image']

        source, mask = self.randaffine(source,mask)

        img_blended,mask=B.dynamic_blend(source,img,mask)
        img_blended = img_blended.astype(np.uint8)
        img = img.astype(np.uint8)

        return img,img_blended,mask
    
    def reorder_landmark(self,landmark):
        landmark_add=np.zeros((13,2))
        for idx,idx_l in enumerate([77,75,76,68,69,70,71,80,72,73,79,74,78]):
            landmark_add[idx]=landmark[idx_l]
        landmark[68:]=landmark_add
        return landmark

    def hflip(self,img,mask=None,landmark=None,bbox=None):
        H,W=img.shape[:2]
        landmark=landmark.copy()
        bbox=bbox.copy()

        if landmark is not None:
            landmark_new=np.zeros_like(landmark)

            
            landmark_new[:17]=landmark[:17][::-1]
            landmark_new[17:27]=landmark[17:27][::-1]

            landmark_new[27:31]=landmark[27:31]
            landmark_new[31:36]=landmark[31:36][::-1]

            landmark_new[36:40]=landmark[42:46][::-1]
            landmark_new[40:42]=landmark[46:48][::-1]

            landmark_new[42:46]=landmark[36:40][::-1]
            landmark_new[46:48]=landmark[40:42][::-1]

            landmark_new[48:55]=landmark[48:55][::-1]
            landmark_new[55:60]=landmark[55:60][::-1]

            landmark_new[60:65]=landmark[60:65][::-1]
            landmark_new[65:68]=landmark[65:68][::-1]
            if len(landmark)==68:
                pass
            elif len(landmark)==81:
                landmark_new[68:81]=landmark[68:81][::-1]
            else:
                raise NotImplementedError
            landmark_new[:,0]=W-landmark_new[:,0]
            
        else:
            landmark_new=None

        if bbox is not None:
            bbox_new=np.zeros_like(bbox)
            bbox_new[0,0]=bbox[1,0]
            bbox_new[1,0]=bbox[0,0]
            bbox_new[:,0]=W-bbox_new[:,0]
            bbox_new[:,1]=bbox[:,1].copy()
            if len(bbox)>2:
                bbox_new[2,0]=W-bbox[3,0]
                bbox_new[2,1]=bbox[3,1]
                bbox_new[3,0]=W-bbox[2,0]
                bbox_new[3,1]=bbox[2,1]
                bbox_new[4,0]=W-bbox[4,0]
                bbox_new[4,1]=bbox[4,1]
                bbox_new[5,0]=W-bbox[6,0]
                bbox_new[5,1]=bbox[6,1]
                bbox_new[6,0]=W-bbox[5,0]
                bbox_new[6,1]=bbox[5,1]
        else:
            bbox_new=None

        if mask is not None:
            mask=mask[:,::-1]
        else:
            mask=None
        img=img[:,::-1].copy()
        return img,mask,landmark_new,bbox_new
    
    def collate_fn(self,batch):
        img_f,img_r=zip(*batch)
        data={}
        data['img']=torch.cat([torch.tensor(img_r).float(),torch.tensor(img_f).float()],0)
        data['label']=torch.tensor([0]*len(img_r)+[1]*len(img_f))
        return data
        

    def worker_init_fn(self,worker_id):                                                          
        np.random.seed(np.random.get_state()[1][0] + worker_id)

class RealDataset(Dataset):
    def __init__(self, datapath, dataselect, trans, augment, jsonpath,image_size=320):
        self.datapath = datapath
        self.trans = trans
        self.aug = augment
        self.num_classes = 2
        self.jsonpath = jsonpath
        self.dataselect = dataselect
        self.image_size=(image_size,image_size)
        self.detector = dlib.get_frontal_face_detector()    #使用dlib库提供的人脸提取器
        self.predictor = dlib.shape_predictor('/home/peipeng/Detection/DA-SBI/preprocess/shape_predictor_81_face_landmarks.dat')
        self.landmarks_json='/home/peipeng/Detection/splitidentityv2/jsons/landmarks.json'
        self.next_epoch()

    def next_epoch(self):
        with open(self.landmarks_json, 'r') as f:
            self.landmarks_record =  json.load(f)
        for k,v in self.landmarks_record.items():
            self.landmarks_record[k] = np.array(v)

        with open(self.jsonpath, 'r') as f:
            data = json.load(f)
        if self.dataselect == 'train':
            trainset = data['train']
            
            self.dataset = trainset
            self.dataset = [a for a in self.dataset if a[1]==0]
        if self.dataselect == 'val':
            valset = data['val']
            
            print(len(valset))
            self.dataset = valset
        if self.dataselect == 'test':
            testset = data['test']
            
            print(len(testset))
            self.dataset = testset
        # self.dataset=self.dataset[:160]+self.dataset[-160:]
        random.shuffle(self.dataset)


    def __getitem__(self, item):
        sample = self.dataset[item]
        anchor, label = sample
        anchorimg = Image.open(anchor).resize(self.image_size)
        if self.dataselect=='train':
            label=0
            atimg=self.aug(image=np.array(anchorimg))
            anchorimg = atimg['image']
        anchorimg=self.trans(anchorimg)
        label=torch.tensor(label).long()
        return anchorimg, label

    def __len__(self):
        return len(self.dataset)

class FreqAugDataset(Dataset):
    def __init__(self, datapath, dataselect, trans, augment, jsonpath,image_size=320):
        self.datapath = datapath
        self.trans = trans
        self.aug = augment
        self.num_classes = 2
        self.jsonpath = jsonpath
        self.dataselect = dataselect
        self.image_size=(image_size,image_size)
        self.detector = dlib.get_frontal_face_detector()    #使用dlib库提供的人脸提取器
        self.predictor = dlib.shape_predictor('/home/peipeng/Detection/DA-SBI/preprocess/shape_predictor_81_face_landmarks.dat')
        self.landmarks_json='/home/peipeng/Detection/splitidentityv2/jsons/landmarks.json'
        self.next_epoch()

    def next_epoch(self):
        with open(self.landmarks_json, 'r') as f:
            self.landmarks_record =  json.load(f)
        for k,v in self.landmarks_record.items():
            self.landmarks_record[k] = np.array(v)

        with open(self.jsonpath, 'r') as f:
            data = json.load(f)
        if self.dataselect == 'train':
            trainset = data['train']
            
            self.dataset = trainset
            self.dataset = [a for a in self.dataset if a[1]==0]
        if self.dataselect == 'val':
            valset = data['val']
            
            print(len(valset))
            self.dataset = valset
        if self.dataselect == 'test':
            testset = data['test']
            
            print(len(testset))
            self.dataset = testset
        # self.dataset=self.dataset[:160]+self.dataset[-160:]
        random.shuffle(self.dataset)
        self.img_list=[a[0] for a in self.dataset]

    def __getitem__(self, item):
        sample = self.dataset[item]
        anchor, label = sample
        anchorimg = Image.open(anchor).resize(self.image_size)
        if self.dataselect=='train':
            oimgpath=random.sample(list(set(self.img_list)-set([anchor])),1)[0]
            oimg=Image.open(oimgpath).resize(self.image_size)
            if random.randint(0,1)==1:
                label=1
                anchorimg = fourier_content_perturbation(anchorimg, oimg, beta=random.uniform(0.05, 0.5), ratio=random.uniform(0.01, 0.1))
            else:
                label=0
                anchorimg = fourier_style_perturbation(anchorimg, oimg, beta=random.uniform(0.05, 0.5), ratio=random.uniform(0.01, 0.1))
            atimg=self.aug(image=np.array(anchorimg))
            anchorimg = atimg['image']
        anchorimg=self.trans(anchorimg)
        label=torch.tensor(label).long()
        return anchorimg, label

    def __len__(self):
        return len(self.dataset)

class Competition(Dataset):
    def __init__(self, datapath, dataselect, trans, augment, jsonpath,exceptdata=None,size=320,aug=False):
        self.datapath = datapath
        self.trans = trans
        # self.aug = augment
        self.num_classes = 2
        self.jsonpath = jsonpath
        self.dataselect = dataselect
        self.patchsize=32
        self.aug=aug
        self.imagesize=(size,size)
        self.exceptdata=exceptdata
        self.detector = dlib.get_frontal_face_detector()    #使用dlib库提供的人脸提取器
        self.predictor = dlib.shape_predictor('/home/peipeng/Detection/DA-SBI/preprocess/shape_predictor_81_face_landmarks.dat')
        self.landmarks_json='/home/peipeng/Detection/splitidentityv2/jsons/landmarks.json'
        self.next_epoch()
        
    def next_epoch(self):
        # with open(self.landmarks_json, 'r') as f:
        #     self.landmarks_record =  json.load(f)
        # for k,v in self.landmarks_record.items():
        #     self.landmarks_record[k] = np.array(v)
        with open(self.jsonpath, 'r') as f:
            data = json.load(f)
        if self.dataselect == 'train':
            trainset = data['train']
            trainset =[f for f in trainset if f[1]==0]*5+[f for f in trainset if f[1]==1]
            self.dataset = trainset
            print('traindataset',len(self.dataset))
        if self.dataselect == 'val':
            valset = data['val']
            # valset = [f for f in valset if f[1]==0]
            print(len(valset))
            self.dataset = valset
        if self.dataselect == 'test':
            testset = data['test']
            # testset = [f for f in testset if f[1]==0]
            self.dataset = testset
            print('testset',len(self.dataset)) 
        random.shuffle(self.dataset)
        # self.dataset=self.dataset[:100]  + self.dataset[-100:]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        
        sample,label = self.dataset[item]
        # print(sample,label)
        anchorimg=np.array(Image.open(sample).convert('RGB'))
        # anchorimg=self.compdetect(anchorimg)
        if self.aug and self.dataselect == 'train':
            anchorimg,label=randomaug(anchorimg,label)
        
        if self.dataselect == 'train':
            anchorimg=self.strongaug(np.array(anchorimg))
        else:
            anchorimg=self.valtrans(np.array(anchorimg))
            # anchorimg = atimg['image']
        anchorimg=cv2.resize(anchorimg,self.imagesize,interpolation=cv2.INTER_LINEAR)
        anchorimg = self.trans(anchorimg)
        # if label==1:
        #     data_type=[0,1]
        # else:
        #     data_type=[1,0]
        if self.dataselect == 'val':
            if label==1:
                label=[0,1]
            else:
                label=[1,0]
            label=np.array(label,dtype=np.int64)
            tlabel=torch.tensor(label).float()
        else:
            tlabel=torch.tensor(label)
        return anchorimg, tlabel

    # def compdetect(self, img):
    #     try:
    #         rects = self.detector(np.array(img), 0)
    #         landmark = self.predictor(np.array(img),rects[0])
    #         landmark = face_utils.shape_to_np(landmark)
    #         if self.dataselect=='train':
    #             if np.random.rand()<0.5:
    #                 img,_,landmark,_=self.hflip(img,None,landmark)
    #         cropimg,_,__,___,y0_new,y1_new,x0_new,x1_new=crop_face(img,landmark,margin=False,crop_by_bbox=False,abs_coord=True,phase=self.dataselect)
    #     except:
    #         cropimg=img
    #     return cropimg

    def strongaug(self,img):
        # import albumentations as alb
        # https://github.com/selimsef/dfdc_deepfake_challenge/blob/master/training/datasets/classifier_dataset.py
        transform = alb.Compose(
            [
                # alb.SmallestMaxSize(max_size=400),
                # alb.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                alb.RandomCrop(height=self.imagesize[0], width=self.imagesize[1],always_apply=True),
                alb.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                alb.RandomBrightnessContrast(p=0.5),
                alb.Rotate((-30,30),p=0.3),
                alb.HorizontalFlip(),
                
                alb.OneOf([
                    alb.MotionBlur(p=0.25),
                    alb.ImageCompression(50,90),
                    alb.GaussianBlur(p=0.5),
                    alb.Blur(blur_limit=3, p=0.25),
                ], p=0.2),
                alb.OneOf([
                    alb.MotionBlur(),
                    alb.GaussNoise(),
                ], p=0.2),
                alb.HueSaturationValue(p=0.2),
                alb.OneOf([
                    alb.Sharpen(),
                    alb.RandomBrightnessContrast(),
                ], p=0.6)
                # alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                # ToTensorV2(),
            ]
        )
        image=transform(image=img)['image']
        return image

    def valtrans(self,img):
        # import albumentations as alb
        # https://github.com/selimsef/dfdc_deepfake_challenge/blob/master/training/datasets/classifier_dataset.py
        transform = alb.Compose(
            [
                # alb.SmallestMaxSize(max_size=400),
                # alb.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                alb.CenterCrop(height=self.imagesize[0], width=self.imagesize[1],always_apply=True),
                
            ]
        )
        image=transform(image=img)['image']
        return image

class AEDataset(Dataset):
    def __init__(self, datapath, dataselect, trans, augment, jsonpath,exceptdata=None):
        self.datapath = datapath
        self.trans = trans
        self.aug = augment
        self.num_classes = 2
        self.jsonpath = jsonpath
        self.dataselect = dataselect
        self.patchsize=32
        self.exceptdata=exceptdata
        
        self.next_epoch()
        
    def next_epoch(self):
       
        with open(self.jsonpath, 'r') as f:
            data = json.load(f)
        if self.dataselect == 'train':
            trainset = data['train']
            trainset =[f for f in trainset if f[1]==0]
            self.dataset = trainset
            print('traindataset',len(self.dataset))
        if self.dataselect == 'val':
            valset = data['val']
            # valset = [f for f in valset if f[1]==0]
            print(len(valset))
            self.dataset = valset
        if self.dataselect == 'test':
            testset = data['test']
            # testset = [f for f in testset if f[1]==0]
            self.dataset = testset
            print('testset',len(self.dataset)) 
        random.shuffle(self.dataset)
        # self.dataset=self.dataset[:100]  + self.dataset[-100:]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        
        sample,label = self.dataset[item]
        # print(sample,label)
        anchorimg=Image.open(sample).convert('RGB').resize((256,256))
        if self.dataselect == 'train':
            atimg=self.aug(image=np.array(anchorimg))
            anchorimg = atimg['image']
        anchorimg = self.trans(anchorimg)

        tlabel=torch.tensor(label)
        return anchorimg, tlabel


class IdentitySwapDataset(Dataset):
    def __init__(self, datapath, dataselect, trans, augment, jsonpath,exceptdata=None):
        self.datapath = datapath
        self.trans = trans
        self.aug = augment
        self.num_classes = 2
        self.jsonpath = jsonpath
        self.dataselect = dataselect
        self.patchsize=32
        self.exceptdata=exceptdata
        self.landmarks_json='/home/peipeng/Detection/splitidentityv2/jsons/landmarks.json'
        self.next_epoch()
        
    def next_epoch(self):
        with open(self.landmarks_json, 'r') as f:
            self.landmarks_record =  json.load(f)
        # for k,v in self.landmarks_record.items():
        #     self.landmarks_record[k] = np.array(v)
        with open(self.jsonpath, 'r') as f:
            data = json.load(f)
        if self.dataselect == 'train':
            trainset = data['train']
            trainset = [f for f in trainset if f[1]==0]
            self.dataset = trainset
            print('traindataset',len(self.dataset))
        if self.dataselect == 'val':
            valset = data['val']
            valset = [f for f in valset if f[1]==0]
            print(len(valset))
            self.dataset = valset
        if self.dataselect == 'test':
            testset = data['test']
            testset = [f for f in testset if f[1]==0]
            self.dataset = testset
            print('testset',len(self.dataset)) 
        random.shuffle(self.dataset)
        self.data_list = self.dataset
        # self.dataset=self.dataset[:100]  + self.dataset[-100:]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        
        sample,label = self.dataset[item]
        anchorimg,mask,data_type,background_face,foreground_face=gen_one_datapoint(sample,self.data_list,self.landmarks_record)
        anchorimg=Image.fromarray(anchorimg).convert('RGB').resize((224,224))
        mask=Image.fromarray(mask[:,:,0]).convert('L').resize((224,224))
        background_face=Image.fromarray(background_face).convert('RGB').resize((224,224))
        foreground_face=Image.fromarray(foreground_face).convert('RGB').resize((224,224))
        anchorimg = self.trans(anchorimg)
        background_face=self.trans(background_face)
        foreground_face=self.trans(foreground_face)
        maskimg=self.trans(np.array(mask))
        tlabel=torch.tensor(np.array(data_type)).float() 
        return anchorimg,background_face,foreground_face, maskimg, tlabel


class IdentitySwap_EvalDataset(Dataset):
    def __init__(self, datapath, dataselect, trans, augment, dataset,exceptdata=None):
        self.datapath = datapath
        self.trans = trans
        self.aug = augment
        self.num_classes = 2
        self.datasetname=dataset
        if dataset=='Deepfakes':
            self.jsonpath = 'jsons/ff++-Deepfakes-raw.json'
        if dataset=='FaceShifter':
            self.jsonpath = 'jsons/ff++-FaceShifter-raw.json'
        self.dataselect = dataselect
        self.patchsize=32
        self.exceptdata=exceptdata
        self.landmarks_json='/home/peipeng/Detection/splitidentityv2/jsons/landmarks.json'
        self.next_epoch()
        
    def next_epoch(self):
        with open(self.jsonpath, 'r') as f:
            data = json.load(f)
        if self.dataselect == 'train':
            trainset = data['train']
            self.dataset = trainset
            print('traindataset',len(self.dataset))
        if self.dataselect == 'val':
            valset = data['val']
            print(len(valset))
            self.dataset = valset
        if self.dataselect == 'test':
            testset = data['test']
            self.dataset = testset
            print('testset',len(self.dataset)) 
        random.shuffle(self.dataset)
        self.data_list = self.dataset
        # self.dataset=self.dataset[:100]  + self.dataset[-100:]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        
        flag=0
        while flag==0:
            sample,label = self.dataset[item]
            # print(label)
            item=(item+1)%len(self.dataset)
            anchorimg = Image.open(sample).convert('RGB')
            
            if label==0:
                targetimg1=anchorimg
                targetimg2=anchorimg
                maskimg=Image.fromarray(np.zeros(targetimg1.size))
                tlabel=np.array([1,0])
                flag=1
            if label==1:
                tlabel=np.array([0,1])
                targetimgnames=sample.split('/')[-2].split('.')[0].split('_')
                targetimgpath=sample.replace('manipulated_sequences/'+self.datasetname,'original_sequences/youtube')
                maskpath=sample.replace('raw','masks')
                targetimg1path=targetimgpath.replace('_'+targetimgnames[1],'')
                targetimg2path=targetimgpath.replace(targetimgnames[0]+'_','')
                if os.path.exists(targetimg1path) and os.path.exists(targetimg2path):
                    targetimg1 = Image.open(targetimg1path).convert('RGB')
                    targetimg2 = Image.open(targetimg2path).convert('RGB')
                    if self.datasetname=='FaceShifter':
                        maskimg=Image.fromarray(np.zeros(targetimg1.size))
                    else:    
                        maskimg = Image.open(maskpath).convert('L')
                    flag=1
                else:
                    continue
            # print(label,tlabel)
            # box = (16, 16, 304, 304)
            box = (32,32,288,288)
            anchorimg = anchorimg.crop(box)
            targetimg1 = targetimg1.crop(box)
            targetimg2 = targetimg2.crop(box)
            maskimg = maskimg.crop(box)
            anchorimg = anchorimg.resize((224,224))
            targetimg1 = targetimg1.resize((224,224))
            targetimg2 = targetimg2.resize((224,224))
            maskimg = maskimg.resize((224,224))  
            if self.dataselect == 'train':
                atimg=self.aug(image=np.array(anchorimg),image2=np.array(targetimg1))
                anchorimg,targetimg11 = atimg['image'], atimg['image2']
            anchorimg = self.trans(anchorimg)
            targetimg1=self.trans(targetimg1)
            targetimg2=self.trans(targetimg2)
            maskimg=self.trans(maskimg)
            tlabel=torch.tensor(tlabel).float() 
        return anchorimg, targetimg1, targetimg2, maskimg, tlabel

class FFDataset(Dataset):
    def __init__(self, datapath, dataselect, trans, augment, dataset,exceptdata=None):
        self.datapath = datapath
        self.trans = trans
        self.aug = augment
        self.num_classes = 2
        self.datasetname=dataset
        if dataset=='Deepfakes':
            self.jsonpath = 'jsons/ff++-Deepfakes-raw.json'
        if dataset=='FaceShifter':
            self.jsonpath = 'jsons/ff++-FaceShifter-raw.json'
        self.dataselect = dataselect
        self.patchsize=32
        self.exceptdata=exceptdata
        self.detector = dlib.get_frontal_face_detector()    #使用dlib库提供的人脸提取器
        self.predictor = dlib.shape_predictor('faker/shape_predictor_68_face_landmarks.dat')
        self.landmarks_json='/home/peipeng/Detection/splitidentityv2/jsons/landmarks.json'
        self.next_epoch()
        
    def next_epoch(self):
        with open(self.jsonpath, 'r') as f:
            data = json.load(f)
        if self.dataselect == 'train':
            trainset = data['train']
            self.dataset = trainset
            print('traindataset',len(self.dataset))
        if self.dataselect == 'val':
            valset = data['val']
            print(len(valset))
            self.dataset = valset
        if self.dataselect == 'test':
            testset = data['test']
            self.dataset = testset
            print('testset',len(self.dataset)) 
        random.shuffle(self.dataset)
        self.data_list = self.dataset
        # self.dataset=self.dataset[:100]  + self.dataset[-100:]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        
        flag=0
        while flag==0:
            sample,label = self.dataset[item]
            # print(label)
            item=(item+1)%len(self.dataset)
            anchorimg = Image.open(sample).convert('RGB')
            
            if label==0:
                targetimg1=anchorimg
                targetimg2=anchorimg
                maskimg=Image.fromarray(np.zeros(targetimg1.size))
                tlabel=np.array([1,0])
                flag=1
            if label==1:
                tlabel=np.array([0,1])
                targetimgnames=sample.split('/')[-2].split('.')[0].split('_')
                targetimgpath=sample.replace('manipulated_sequences/'+self.datasetname,'original_sequences/youtube')
                maskpath=sample.replace('raw','masks')
                targetimg1path=targetimgpath.replace('_'+targetimgnames[1],'')
                targetimg2path=targetimgpath.replace(targetimgnames[0]+'_','')
                if os.path.exists(targetimg1path) and os.path.exists(targetimg2path):
                    targetimg1 = Image.open(targetimg1path).convert('RGB')
                    targetimg2 = Image.open(targetimg2path).convert('RGB')
                    if self.datasetname=='FaceShifter':
                        maskimg=Image.fromarray(np.zeros(targetimg1.size))
                    else:    
                        maskimg = Image.open(maskpath).convert('L')
                    flag=1
                else:
                    continue
            # print(label,tlabel)
            # box = (16, 16, 304, 304)
            box = (32,32,288,288)
            anchorimg = anchorimg.crop(box)
            targetimg1 = targetimg1.crop(box)
            targetimg2 = targetimg2.crop(box)
            maskimg = maskimg.crop(box)
            anchorimg = anchorimg.resize((224,224))
            targetimg1 = targetimg1.resize((224,224))
            targetimg2 = targetimg2.resize((224,224))
            maskimg = maskimg.resize((224,224)) 
            try:
                anchorimg=np.array(anchorimg)
                rects = self.detector(np.array(anchorimg), 0)
                landmarks = np.matrix([[p.x, p.y] for p in self.predictor(np.array(anchorimg),rects[0]).parts()])
                minx,maxx=landmarks[18:68][:,0].min(),landmarks[18:68][:,0].max()
                miny,maxy=landmarks[18:68][:,1].min(),landmarks[18:68][:,1].max()
                mask=np.zeros_like(anchorimg)
                mask[miny:maxy,minx:maxx,:]=255
                maskimg=mask[:,:,:1]
            except:
                maskimg=np.zeros((224,224,1))
            if self.dataselect == 'train':
                atimg=self.aug(image=np.array(anchorimg),image2=np.array(targetimg1))
                anchorimg,targetimg11 = atimg['image'], atimg['image2']
            anchorimg = self.trans(anchorimg)
            targetimg1=self.trans(targetimg1)
            targetimg2=self.trans(targetimg2)
            maskimg=self.trans(maskimg).float()
            tlabel=torch.tensor(tlabel).float() 
        return anchorimg, targetimg1, targetimg2, maskimg, tlabel



class IdentityDFDataset(Dataset):
    def __init__(self, datapath, dataselect, trans, augment, jsonpath,exceptdata=None):
        self.datapath = datapath
        self.trans = trans
        self.aug = augment
        self.num_classes = 2
        self.jsonpath = jsonpath
        self.dataselect = dataselect
        self.patchsize=32
        self.exceptdata=exceptdata
        self.next_epoch()
        
    def next_epoch(self):
        with open(self.jsonpath, 'r') as f:
            data = json.load(f)
        if self.dataselect == 'train':
            trainset = data['train']
            self.dataset = trainset
            print('traindataset',len(self.dataset))
        if self.dataselect == 'val':
            valset = data['val']
            print(len(valset))
            self.dataset = valset
        if self.dataselect == 'test':
            testset = data['test']
            self.dataset = testset
            print('testset',len(self.dataset)) 
        random.shuffle(self.dataset)
        # self.dataset=self.dataset[:100]  + self.dataset[-100:]

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, item):
        
        flag=0
        while flag==0:
            sample,label = self.dataset[item]
            # print(label)
            item=(item+5)%len(self.dataset)
            anchorimg = Image.open(sample).convert('RGB')
            
            if label==0:
                targetimg1=anchorimg
                targetimg2=anchorimg
                maskimg=Image.fromarray(np.zeros(targetimg1.size))
                tlabel=np.array([1,0])
                flag=1
            if label==1:
                tlabel=np.array([0,1])
                targetimgnames=sample.split('/')[-2].split('.')[0].split('_')
                # print(targetimgnames)
                targetimgpath=sample.replace('manipulated_sequences/Deepfakes','original_sequences/youtube')
                maskpath=sample.replace('raw','masks')
                targetimg1path=targetimgpath.replace('_'+targetimgnames[1],'')
                targetimg2path=targetimgpath.replace(targetimgnames[0]+'_','')
                if os.path.exists(targetimg1path) and os.path.exists(targetimg2path):
                    targetimg1 = Image.open(targetimg1path).convert('RGB')
                    targetimg2 = Image.open(targetimg2path).convert('RGB')
                    maskimg = Image.open(maskpath).convert('L')
                    flag=1
                else:
                    continue
            # print(label,tlabel)
            anchorimg = anchorimg.resize((224,224))
            targetimg1 = targetimg1.resize((224,224))
            targetimg2 = targetimg2.resize((224,224))
            maskimg = maskimg.resize((224,224))    
            anchorimg = self.trans(anchorimg)
            targetimg1=self.trans(targetimg1)
            targetimg2=self.trans(targetimg2)
            maskimg=self.trans(maskimg)
            tlabel=torch.tensor(tlabel).float() 
            # print(tlabel)
        return anchorimg, targetimg1, targetimg2, maskimg, tlabel

class IdentityVGGFACESimswapDataset(Dataset):
    def __init__(self, datapath, dataselect, trans, augment):
        self.datapath = datapath
        self.trans = trans
        self.aug = augment
        self.num_classes = 2
        self.jsonpath = 'jsons/vggface2fake_simswap2_select10.json'
        self.dataselect = dataselect
        self.patchsize=32
       
        self.detector = dlib.get_frontal_face_detector()    #使用dlib库提供的人脸提取器
        self.predictor = dlib.shape_predictor('faker/shape_predictor_68_face_landmarks.dat')
        self.next_epoch()
        
    def next_epoch(self):
        with open(self.jsonpath, 'r') as f:
            data = json.load(f)
        if self.dataselect == 'train':
            trainset = data['train']
            self.dataset = trainset
            print('traindataset',len(self.dataset))
        if self.dataselect == 'val':
            valset = data['val']
            print(len(valset))
            self.dataset = valset
        if self.dataselect == 'test':
            testset = data['test']
            self.dataset = testset
            print('testset',len(self.dataset)) 
        random.shuffle(self.dataset)
        # self.dataset=self.dataset[:100]  + self.dataset[-100:]

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, item):

        forgery,source,target,label = self.dataset[item]
        anchorimg=Image.open(forgery).convert('RGB')
        targetimg2=Image.open(source).convert('RGB')
        targetimg1=Image.open(target).convert('RGB')
        targetimg1 = targetimg1.resize((224,224))
        targetimg2 = targetimg2.resize((224,224))

        mask=np.zeros((224,224,1))
        if label==0:
            tlabel=np.array([1,0])
        if label==1:
            tlabel=np.array([0,1])
        anchorimg = anchorimg.resize((224,224))
        if self.dataselect == 'train':
            anchorimg = self.aug(image=np.array(anchorimg))['image']
        anchorimg = self.trans(anchorimg)
        mask = self.trans(mask)
        targetimg1=self.trans(targetimg1)
        targetimg2=self.trans(targetimg2)
        tlabel=torch.tensor(tlabel).float() 
        return anchorimg, targetimg1, targetimg2, mask, tlabel


class IdentityFFHQSimswapDataset(Dataset):
    def __init__(self, datapath, dataselect, trans, augment):
        self.datapath = datapath
        self.trans = trans
        self.aug = augment
        self.num_classes = 2
        self.jsonpath = 'jsons/ffhqfake_e4s2.json'
        self.dataselect = dataselect
        self.patchsize=32
       
        self.detector = dlib.get_frontal_face_detector()    #使用dlib库提供的人脸提取器
        self.predictor = dlib.shape_predictor('faker/shape_predictor_68_face_landmarks.dat')
        self.next_epoch()
        
    def next_epoch(self):
        with open(self.jsonpath, 'r') as f:
            data = json.load(f)
        if self.dataselect == 'train':
            trainset = data['train']
            self.dataset = trainset
            print('traindataset',len(self.dataset))
        if self.dataselect == 'val':
            valset = data['val']
            print(len(valset))
            self.dataset = valset
        if self.dataselect == 'test':
            testset = data['test']
            self.dataset = testset
            print('testset',len(self.dataset)) 
        random.shuffle(self.dataset)
        # self.dataset=self.dataset[:100]  + self.dataset[-100:]

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, item):

        forgery,source,target,label = self.dataset[item]
        anchorimg=Image.open(forgery).convert('RGB')
        targetimg2=Image.open(source).convert('RGB')
        targetimg1=Image.open(target).convert('RGB')
        targetimg1 = targetimg1.resize((224,224))
        targetimg2 = targetimg2.resize((224,224))

        mask=np.zeros((224,224,1))
        if label==0:
            tlabel=np.array([1,0])
        if label==1:
            tlabel=np.array([0,1])
        anchorimg = anchorimg.resize((224,224))
        if self.dataselect == 'train':
            anchorimg = self.aug(image=np.array(anchorimg))['image']
        anchorimg = self.trans(anchorimg)
        mask = self.trans(mask)
        targetimg1=self.trans(targetimg1)
        targetimg2=self.trans(targetimg2)
        tlabel=torch.tensor(tlabel).float() 
        return anchorimg, targetimg1, targetimg2, mask, tlabel

class IdentityFFMegaFSDataset(Dataset):
    def __init__(self, datapath, dataselect, trans, augment):
        self.datapath = datapath
        self.trans = trans
        self.aug = augment
        self.num_classes = 2
        self.jsonpath = 'jsons/ff_megafs.json'
        self.dataselect = dataselect
        self.patchsize=32
       
        self.detector = dlib.get_frontal_face_detector()    #使用dlib库提供的人脸提取器
        self.predictor = dlib.shape_predictor('faker/shape_predictor_68_face_landmarks.dat')
        self.next_epoch()
        
    def next_epoch(self):
        with open(self.jsonpath, 'r') as f:
            data = json.load(f)
        if self.dataselect == 'train':
            trainset = data['train']*10
            self.dataset = trainset
            print('traindataset',len(self.dataset))
        if self.dataselect == 'val':
            valset = data['val']
            print(len(valset))
            self.dataset = valset
        if self.dataselect == 'test':
            testset = data['test']
            self.dataset = testset
            print('testset',len(self.dataset)) 
        random.shuffle(self.dataset)
        # self.dataset=self.dataset[:100]  + self.dataset[-100:]

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, item):

        imgpath = self.dataset[item]
        img=Image.open(imgpath).convert('RGB')
        img = np.array(img.resize((224*3,224)))
        label=random.randint(0,1)

        mask=np.zeros((224,224,1))
        if label==0:
            tlabel=np.array([1,0])
            targetimg1=img[:,224:448,:]
            anchorimg=targetimg1
            targetimg2=targetimg1
        if label==1:
            tlabel=np.array([0,1])
            anchorimg=img[:,448:,:]
            targetimg1=img[:,224:448,:]
            targetimg2=img[:,:224,:]
        # print(anchorimg.shape,targetimg1.shape,targetimg2.shape)
        if self.dataselect == 'train':
            anchorimg = self.aug(image=np.array(anchorimg))['image']
        anchorimg = self.trans(anchorimg)
        mask = self.trans(mask)
        targetimg1=self.trans(targetimg1)
        targetimg2=self.trans(targetimg2)
        tlabel=torch.tensor(tlabel).float() 
        return anchorimg, targetimg1, targetimg2, mask, tlabel

class IdentityCelebaMegaFSDataset(Dataset):
    def __init__(self, datapath, dataselect, trans, augment,d='FTM'):
        self.datapath = datapath
        self.trans = trans
        self.aug = augment
        self.num_classes = 2
        if d=='FTM':
            self.jsonpath = 'jsons/celeba_FTM.json'
        if d=='IDInjection':
            self.jsonpath = 'jsons/celeba_IDInjection.json'
        if d=='LCR':
            self.jsonpath = 'jsons/celeba_LCR.json'
        self.dataselect = dataselect
        self.patchsize=32
       
        self.detector = dlib.get_frontal_face_detector()    #使用dlib库提供的人脸提取器
        self.predictor = dlib.shape_predictor('faker/shape_predictor_68_face_landmarks.dat')
        self.targetpath='/mnt/UserData1/peipeng/Data/CelebAFake/real/CelebAMask-HQ/CelebA-HQ-img/'
        self.next_epoch()
        
    def next_epoch(self):
        with open(self.jsonpath, 'r') as f:
            data = json.load(f)
        if self.dataselect == 'train':
            trainset = data['train']
            self.dataset = trainset
            print('traindataset',len(self.dataset))
        if self.dataselect == 'val':
            valset = data['val']
            print(len(valset))
            self.dataset = valset
        if self.dataselect == 'test':
            testset = data['test']
            self.dataset = testset
            print('testset',len(self.dataset)) 
        random.shuffle(self.dataset)
        # self.dataset=self.dataset[:100]  + self.dataset[-100:]

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, item):

        imgpath = self.dataset[item]
        img=Image.open(imgpath).convert('RGB')
        img = np.array(img.resize((224,224)))
        label=random.randint(0,1)
        mask=np.zeros((224,224,1))
        if label==0:
            tlabel=np.array([1,0])
            targetimgpath=self.targetpath+imgpath.split('/')[-1].split('_')[0]+'.jpg'
            targetimg1=Image.open(targetimgpath).convert('RGB')
            targetimg1 = np.array(targetimg1.resize((224,224)))
            anchorimg=targetimg1
            targetimg2=targetimg1
        if label==1:
            tlabel=np.array([0,1])
            target1path=self.targetpath+imgpath.split('/')[-1].split('_')[0]+'.jpg'
            target2path=self.targetpath+imgpath.split('/')[-1].split('_')[1]
            if os.path.exists(target1path) and os.path.exists(target2path):
                anchorimg=img
                targetimg1=Image.open(target1path).convert('RGB')
                targetimg2=Image.open(target2path).convert('RGB')
                targetimg1 = np.array(targetimg1.resize((224,224)))
                targetimg2 = np.array(targetimg2.resize((224,224)))
            else:
                anchorimg=img
                targetimg1=img
                targetimg2=img
                label=0
        try:
            rects = self.detector(np.array(anchorimg), 0)
            landmarks = np.matrix([[p.x, p.y] for p in self.predictor(np.array(anchorimg),rects[0]).parts()])
            minx,maxx=landmarks[18:68][:,0].min(),landmarks[18:68][:,0].max()
            miny,maxy=landmarks[18:68][:,1].min(),landmarks[18:68][:,1].max()
            mask=np.zeros_like(anchorimg)
            mask[miny:maxy,minx:maxx,:]=255
            mask=mask[:,:,:1]
        except:
            mask=np.zeros((224,224,1))
        # print(anchorimg.shape,targetimg1.shape,targetimg2.shape)
        # if self.dataselect == 'train':
        #     anchorimg = self.aug(image=np.array(anchorimg))['image']
        anchorimg = self.trans(anchorimg)
        mask = self.trans(mask).float()
        targetimg1=self.trans(targetimg1)
        targetimg2=self.trans(targetimg2)
        tlabel=torch.tensor(tlabel).float() 
        return anchorimg, targetimg1, targetimg2, mask, tlabel

class IdentitySimswapDataset(Dataset):
    def __init__(self, datapath, dataselect, trans, augment, jsonpath,exceptdata=None):
        self.datapath = datapath
        self.trans = trans
        self.aug = augment
        self.num_classes = 2
        self.jsonpath = jsonpath
        self.dataselect = dataselect
        self.patchsize=32
        self.exceptdata=exceptdata
        self.detector = dlib.get_frontal_face_detector()    #使用dlib库提供的人脸提取器
        self.predictor = dlib.shape_predictor('faker/shape_predictor_68_face_landmarks.dat')
        self.next_epoch()
        
    def next_epoch(self):
        with open(self.jsonpath, 'r') as f:
            data = json.load(f)
        if self.dataselect == 'train':
            trainset = data['train']
            self.dataset = trainset
            print('traindataset',len(self.dataset))
        if self.dataselect == 'val':
            valset = data['val']
            print(len(valset))
            self.dataset = valset
        if self.dataselect == 'test':
            testset = data['test']
            self.dataset = testset
            print('testset',len(self.dataset)) 
        random.shuffle(self.dataset)
        # self.dataset=self.dataset[:100]  + self.dataset[-100:]

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, item):
        flag=1
        while flag==1:
            try:
                forgery,source,target,label = self.dataset[item]
                # anchorimg=Image.open(forgery).convert('RGB')
                targetimg2=Image.open(source).convert('RGB')
                targetimg1=Image.open(target).convert('RGB')
                targetimg1 = targetimg1.resize((224,224))
                targetimg2 = targetimg2.resize((224,224))
                rects = self.detector(np.array(targetimg1), 0)
                landmarks = np.matrix([[p.x, p.y] for p in self.predictor(np.array(targetimg1),rects[0]).parts()])
                
                if landmarks.shape[0]==68:
                    flag==0
                    break
                else:
                    item=(item+1)%len(self.dataset)
            except:
                item=(item+1)%len(self.dataset)

        if label==0:
            tlabel=np.array([1,0])
            anchorimg=Image.open(target).convert('RGB')
            anchorimg = anchorimg.resize((224,224))
            mask=np.zeros((224,224,1))
            
        if label==1:
            tlabel=np.array([0,1])
            anchorimg,mask= blended_face(target,forgery,landmarks)
            anchorimg = Image.fromarray(anchorimg)
            # print(mask.shape)
            mask=Image.fromarray(np.array(mask[:,:,0]*255,dtype=np.uint8)).resize((224,224))
            
            # print('mask',mask.shape)
            
        anchorimg = anchorimg.resize((224,224))
        
        anchorimg = self.trans(anchorimg)
        mask = self.trans(mask)
        targetimg1=self.trans(targetimg1)
        targetimg2=self.trans(targetimg2)
        tlabel=torch.tensor(tlabel).float() 
        return anchorimg, targetimg1, targetimg2, mask, tlabel