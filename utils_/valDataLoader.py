import torch
from torch.utils.data import Dataset, ConcatDataset
from tqdm import tqdm
import json
import os
import h5py
import numpy as np


def sample_evs(events, i0, i1):
    evs = events[:, i0:i1].copy()
    evs[2] -= evs[2, 0]         # ts normalization
    evs[3] = (evs[3] - 0.5) * 2 # polarity val [-1, 1]
    assert evs[3].max() <= 1
    assert evs[3].min() >= -1  
    xs = evs[0:1, :].reshape(-1)
    ys = evs[1:2, :].reshape(-1)
    ts = evs[2:3, :].reshape(-1)
    ps = evs[3:4, :].reshape(-1)
    return xs, ys, ts, ps 


@torch.no_grad()
def voxel_to_grid(xs, ys, ts, ps, B, h, w):
    
    xs = torch.from_numpy(xs).long()
    ys = torch.from_numpy(ys).long()
    ts = torch.from_numpy(ts).float()
    ps = torch.from_numpy(ps).float()

    bins = []
    dt = ts[-1] - ts[0] + 1e-9
    t_norm = (ts - ts[0]) / dt * (B - 1)
    zeros = torch.zeros(t_norm.size())

    for bi in range(B):
        bili_w = torch.max(zeros, 1.0 - torch.abs(t_norm - bi))
        p = ps * bili_w
        plane = torch.zeros(h, w)
        plane.index_put_((ys, xs), p, accumulate=True)
        bins.append(plane)
    
    return torch.stack(bins)


def valDataLoad(args):
    data_paths = [os.path.join(args.val_data_path, d.name) for d in os.scandir(args.val_data_path) if d.name.endswith('h5')]
    data_paths.sort()
    datasets = []
    for data_path in tqdm(data_paths, desc='loading_valset'):
        datasets.append(validDataloader(args, data_path))
    return ConcatDataset(datasets)

class validDataloader(Dataset):
    def __init__(self, args, data_path):
    
        self.data_path = data_path
        self.B = args.num_bins
        self.ev_rate = args.ev_rate 
        self.data_name = self.data_path.split('/')[-1][:-3]
        
        with h5py.File(self.data_path) as f:
            self.img_k = list(f['images'].keys())
            k = self.img_k[0]
            self.h, self.w = f['images'][k].shape[-2:]
        
        self.num_evs = int(self.h * self.w * args.ev_rate)
        self.img_num = len(self.img_k) - 1
    
    def __len__(self, ):
        return self.img_num
    
    def __getitem__(self, index):

        item = {}
        img_k0 = self.img_k[index]
        img_k1 = self.img_k[index + 1]

        with h5py.File(self.data_path) as data:

            i0 = data['images'][img_k0].attrs['event_idx']
            i1 = data['images'][img_k1].attrs['event_idx']
            img = np.asarray(data['images'][img_k1], dtype=np.float32)[None]/255.0

            max_len = i1 - i0

            evs = np.vstack([
            np.asarray(data['events/xs'][i0:i1], dtype=np.float64),
            np.asarray(data['events/ys'][i0:i1], dtype=np.float64),
            np.asarray(data['events/ts'][i0:i1], dtype=np.float64),
            np.asarray(data['events/ps'][i0:i1], dtype=np.float64)], dtype=np.float64).reshape(4, -1)

        ev_time = evs[2, -1] - evs[2, 0]
        if ev_time > 0:
            voxels = self.process_ev(evs, max_len)
        else:
            voxels = torch.zeros(self.B, self.h, self.w)

        item['frame'] = img
        item['events'] = voxels
        item['len'] = self.img_num 
        item['name'] = self.data_name
        return item

    def process_ev(self, events, max_len):

        voxels = []
        last_idx, next_idx = 0, self.num_evs

        while max_len > next_idx:
            xs, ys, ts, ps = sample_evs(events, last_idx, next_idx)
            voxels.append(voxel_to_grid(xs, ys, ts, ps, self.B, self.h, self.w))
            last_idx = next_idx
            next_idx = np.clip(next_idx + self.num_evs, 0, max_len + 1)
        if max_len <= next_idx:
            # if not enough events, generate some noisy
            # (between 10 and 100) random events
            if next_idx - last_idx < 3 or max_len < 3:
                num_evs = torch.FloatTensor(1, 1).uniform_(10, 100).long().item()
                xs = np.random.uniform(0, self.w, size=num_evs).astype(np.float32).reshape(-1)
                ys = np.random.uniform(0, self.h, size=num_evs).astype(np.float32).reshape(-1)
                ts = np.random.uniform(0, 1000, size=num_evs).astype(np.float32).reshape(-1)
                ps = np.random.uniform(0, 2, size=num_evs).astype(np.float32).reshape(-1)
                ps = 2 * (ps - 0.5)

                voxels.append(voxel_to_grid(xs, ys, ts, ps, self.B, self.h, self.w))
            else:
                xs, ys, ts, ps = sample_evs(events, last_idx, max_len)
                voxels.append(voxel_to_grid(xs, ys, ts, ps, self.B, self.h, self.w))
        
        return torch.stack(voxels)
    

def testDataLoad(args): 
    data_paths = [os.path.join(args.test_data_path, d.name) for d in os.scandir(args.test_data_path) if d.name.endswith('h5')]
    data_paths.sort()
    datasets = []
    for data_path in tqdm(data_paths, desc='loading_testset'):
        datasets.append(testDataloader(args, data_path))
    return ConcatDataset(datasets)

class testDataloader(Dataset):
    def __init__(self, args, data_path):
        
        self.data_path = data_path
        self.B = args.num_bins
        self.ev_rate = args.ev_rate
        with open(args.test_data_json) as f:
            self.h_param = json.load(f)
            self.data_name = self.data_path.split('/')[-1][:-3]
            div_time = 1e9 # the time is in nanoseconds
            start_time = 0
            end_time = -1
            if self.data_name in self.h_param['sequences'].keys():
                div_time = self.h_param['sequences'][self.data_name]['div_time']
                start_time = self.h_param['sequences'][self.data_name]['start_time_s']
                end_time = self.h_param['sequences'][self.data_name]['end_time_s']
        warmup_time = 1
        self.start_k = -1
        self.img_k = []

        with h5py.File(self.data_path) as f:
            
            img_k = list(f['images'].keys())
            img_k.sort()
            self.h, self.w = f['images'][img_k[0]].shape[-2:]
            t0 = f['events']['ts'][0]
            for k in img_k[:-1]:
                idx = min(f['events']['ts'].shape[0]-1 ,f['images'][k].attrs['event_idx'])
                img_ts = f['events']['ts'][idx]
                
                ts_ = (img_ts - t0) / div_time # the time is in nanoseconds
                if ts_ < max(0, start_time - warmup_time):
                    continue 
                if self.start_k == -1 and ts_ > start_time:
                    self.start_k = {k: ts_}
                self.img_k.append(k)

                if ts_ > end_time and end_time != -1:
                    break
        
        self.num_evs = int(self.h * self.w * args.ev_rate)
        self.img_num = len(self.img_k) - 1


    def __len__(self, ):
        return  self.img_num
    
    def __getitem__(self, index):

        item = {}
        img_k0 = self.img_k[index]
        img_k1 = self.img_k[index + 1]

        with h5py.File(self.data_path) as data:

            i0 = data['images'][img_k0].attrs['event_idx']
            i1 = data['images'][img_k1].attrs['event_idx']
            img = np.asarray(data['images'][img_k1], dtype=np.float32)[None]/255.0
            max_len = i1 - i0

            evs = np.vstack([
            np.asarray(data['events/xs'][i0:i1], dtype=np.float64),
            np.asarray(data['events/ys'][i0:i1], dtype=np.float64),
            np.asarray(data['events/ts'][i0:i1], dtype=np.float64),
            np.asarray(data['events/ps'][i0:i1], dtype=np.float64)], dtype=np.float64).reshape(4, -1)

        ev_time = evs[2, -1] - evs[2, 0]
        if ev_time > 0:
            voxels = self.process_ev(evs, max_len)
        else:
            voxels = torch.zeros(self.B, self.h, self.w)

        item['frame'] = img
        item['events'] = voxels
        item['len'] = self.img_num 
        item['name'] = self.data_name
        return item
    
    def process_ev(self, events, max_len):

        voxels = []
        last_idx, next_idx = 0, self.num_evs

        while max_len > next_idx:
            xs, ys, ts, ps = sample_evs(events, last_idx, next_idx)
            voxels.append(voxel_to_grid(xs, ys, ts, ps, self.B, self.h, self.w))
            last_idx = next_idx
            next_idx = np.clip(next_idx + self.num_evs, 0, max_len + 1)
        if max_len <= next_idx:
            # if not enough events, generate some noisy
            # (between 10 and 100) random events
            if next_idx - last_idx < 3 or max_len < 3:
                num_evs = torch.FloatTensor(1, 1).uniform_(10, 100).long().item()
                xs = np.random.uniform(0, self.w, size=num_evs).astype(np.float32).reshape(-1)
                ys = np.random.uniform(0, self.h, size=num_evs).astype(np.float32).reshape(-1)
                ts = np.random.uniform(0, 1000, size=num_evs).astype(np.float32).reshape(-1)
                ps = np.random.uniform(0, 2, size=num_evs).astype(np.float32).reshape(-1)
                ps = 2 * (ps - 0.5)

                voxels.append(voxel_to_grid(xs, ys, ts, ps, self.B, self.h, self.w))
            else:
                xs, ys, ts, ps = sample_evs(events, last_idx, max_len)
                voxels.append(voxel_to_grid(xs, ys, ts, ps, self.B, self.h, self.w))
        
        return torch.stack(voxels)
            