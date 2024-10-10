import torch
from torch.utils.data import Dataset
import numbers
import random
import os
import h5py


class transformCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, x, is_flow=False):
        for t in self.transforms:
            if t:
                x = t(x, is_flow)
        return x
    

class randomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

        self.top_x = None
        self.top_y = None 
        self.low_x = None
        self.low_y = None
    
    def set_params(self, x):
        h, w = x.shape[-2:]
        th, tw = self.size
        assert th < h and tw < w, 'Pre defined Crop size is too smal'

        if w == tw and h == th:
            self.top_x = 0
            self.top_y = 0
            self.low_x = w
            self.low_y = h
    
        self.top_x = random.randint(0, w - tw)
        self.top_y = random.randint(0, h - th)
        self.low_x = self.top_x + tw
        self.low_y = self.top_y + th
    
    def __call__(self, x, is_flow=False):
        return x[..., self.top_y:self.low_y, self.top_x:self.low_x]


class randomFlip(object):
    def __init__(self, h_flip=0.5, v_flip=0.5):
        self.h_flip_p = h_flip
        self.v_flip_p = v_flip
        self.dims = None

    def set_params(self, x):
        self.dims = []
        if random.random() > self.h_flip_p:
            self.dims.append(-1)

        if random.random() > self.v_flip_p:
            self.dims.append(-2)
    
    def __call__(self, x, is_flow=False):

        flipped = torch.flip(x, dims=self.dims)
        if is_flow:
            for d in self.dims:
                idx = -(d + 1)  # swap since flow is x, y
                flipped[..., idx, :, :] *= -1
        return flipped
    

class noisyDataloader(Dataset):
    def __init__(self,args):
        
        self.data_path = args.train_data_path

        transforms_list = [eval(t)(**kwargs) for t, kwargs in args.transforms.items()]
        if len(transforms_list) == 0:
            transforms_list = [None]
        
        self.transforms = transformCompose(transforms_list)

        self.data_paths = [self.data_path + '/' + d.name for d in os.scandir(self.data_path) if d.name.endswith('h5')] 
        self.data_paths.sort()
        self.len_data_paths = len(self.data_paths)

    def __len__(self, ):
        return self.len_data_paths
    
    def __getitem__(self, index):

        data_path = self.data_paths[index]
        item = {}
        loaded_data = h5py.File(data_path)
        
        # set augmentations
        for t in self.transforms.transforms:
            t.set_params(loaded_data['image000000000'][:])

        evs = torch.from_numpy(loaded_data['voxel'][:]).float()
        # its al redy a displacement !
        flows = torch.from_numpy(loaded_data['flow'][:]).float()  
        img = torch.from_numpy(loaded_data['image000000000'][:]).float() / 256

        item['events'] = self.transforms(evs, is_flow=False)
        item['flow'] = self.transforms(flows, is_flow=True)
        item['img'] = self.transforms(img, is_flow=False) 
        
        return  item