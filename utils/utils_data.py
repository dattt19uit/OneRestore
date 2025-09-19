import torch, os
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data as data
from einops import rearrange
from utils.dynamic_text import get_dynamic_idx, get_dynamic_label

class ImageLoader:
    def __init__(self, root):
        self.img_dir = root

    def __call__(self, img):
        file = f'{self.img_dir}/{img}'
        img = Image.open(file).convert('RGB')
        return img

def imagenet_transform(phase):

    if phase == 'train':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
       
    elif phase == 'test':
        transform = transforms.Compose([
            transforms.Resize([224,224]),
            transforms.ToTensor()
        ])

    return transform

class Dataset_embedding(data.Dataset):
    def __init__(self, cfg_data, phase='train'):
        self.phase = phase 
        self.transform = imagenet_transform(phase)
        self.type_name = cfg_data.type_name
        self.type2idx = {self.type_name[i]: i for i in range(len(self.type_name))}

        if phase == 'train':
            self.loader = ImageLoader(cfg_data.train_dir)
            name = os.listdir(f'{cfg_data.train_dir}/{self.type_name[0]}')
            self.data = []
            for i in range(len(self.type_name)):
                for j in range(len(name)):
                    dynamic_label = get_dynamic_label(phase, self.type_name[i], name[j])
                    # self.data.append([self.type_name[i], name[j]])
                    self.data.append([self.type_name[i], dynamic_label, name[j]])
        elif phase == 'test':
            self.loader = ImageLoader(cfg_data.test_dir)
            name = os.listdir(f'{cfg_data.test_dir}/{self.type_name[0]}')
            self.data = []
            for i in range(1, len(self.type_name)):
                for j in range(len(name)):
                    dynamic_label = get_dynamic_label(phase, self.type_name[i], name[j])
                    # self.data.append([self.type_name[i], name[j]])
                    self.data.append([self.type_name[i], dynamic_label, name[j]])
        print(f'The amount of {phase} data is {len(self.data)}')

    def __getitem__(self, index):

        type_name, dynamic_label, image_name = self.data[index]
        # scene = self.type2idx[type_name]
        scene = get_dynamic_idx(self.phase, type_name, image_name)
        image = self.transform(self.loader(f'{type_name}/{image_name}'))

        return (dynamic_label, image)

    def __len__(self):
        return len(self.data)

def init_embedding_data(cfg_em,  phase):
    if phase == 'train':
        train_dataset = Dataset_embedding(cfg_em, 'train')
        test_dataset = Dataset_embedding(cfg_em, 'test')
        train_loader = data.DataLoader(train_dataset,
                                    batch_size=cfg_em.batch,
                                    shuffle=True, 
                                    num_workers=cfg_em.num_workers,
                                    pin_memory=True)
        test_loader = data.DataLoader(test_dataset,
                                    batch_size=cfg_em.batch,
                                    shuffle=False, 
                                    num_workers=cfg_em.num_workers,
                                    pin_memory=True)
        print(len(train_dataset),len(test_dataset))
            
    elif phase == 'inference':
        test_dataset = Dataset_embedding(cfg_em, 'test')
        test_loader = data.DataLoader(test_dataset, 
                                    batch_size=1,
                                    shuffle=False, 
                                    num_workers=cfg_em.num_workers,
                                    pin_memory=True)
    
    return train_loader, test_loader