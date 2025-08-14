import os
import numpy as np
import h5py
import glob

import torch
from torch.utils.data import Dataset

from util.voxelize import voxelize
from util.data_util import sa_create, collate_fn
from util.data_util import data_prepare_scannet as data_prepare


class Market(Dataset):
    def __init__(self, split='train', data_root='trainval', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False, loop=1):
        super().__init__()
        self.split = split
        self.data_root = data_root
        self.voxel_size = voxel_size
        self.voxel_max = voxel_max
        self.transform = transform
        self.shuffle_index = shuffle_index
        self.loop = loop

        # Accept either a single HDF5 file or a directory of HDF5 files
        self.data_list = []
        if os.path.isfile(self.data_root):
            # Single HDF5 file
            try:
                with h5py.File(self.data_root, 'r') as f:
                    if 'seg_points' in f:
                        self.data_list = [self.data_root]
                    else:
                        raise ValueError(f"No 'seg_points' dataset in {self.data_root}")
            except (OSError, IOError) as e:
                raise ValueError(f"Cannot open HDF5 file {self.data_root}: {e}")
        else:
            # Directory of HDF5 files
            for fname in os.listdir(self.data_root):
                fpath = os.path.join(self.data_root, fname)
                if os.path.isfile(fpath):
                    try:
                        with h5py.File(fpath, 'r') as f:
                            if 'seg_points' in f:
                                self.data_list.append(fpath)
                    except (OSError, IOError):
                        continue
            if len(self.data_list) == 0:
                raise ValueError(f"No HDF5 files found in directory {self.data_root}")
            
        # Create sample list (each HDF5 file contains multiple samples)
        self.sample_list = []
        for h5_file in self.data_list:
            with h5py.File(h5_file, 'r') as f:
                num_samples = f['seg_points'].shape[0]
                for i in range(num_samples):
                    self.sample_list.append((h5_file, i))
        
        # Split data
        total_samples = len(self.sample_list)
        if split == 'train':
            self.sample_list = self.sample_list[:int(0.8 * total_samples)]
        elif split == 'val':
            self.sample_list = self.sample_list[int(0.8 * total_samples):]
        #else:  # test
        #    self.sample_list = self.sample_list[int(0.9 * total_samples):]
            
        print("voxel_size: ", voxel_size)
        print("Totally {} samples in {} set.".format(len(self.sample_list), split))

    def __getitem__(self, idx):
        sample_idx = idx % len(self.sample_list)
        h5_file, sample_id = self.sample_list[sample_idx]
        
        # Load sample from HDF5 file
        with h5py.File(h5_file, 'r') as f:
            coord = f['seg_points'][sample_id]  # (20480, 3)
            feat = f['seg_colors'][sample_id]   # (20480, 3)
            label = f['seg_labels'][sample_id]  # (20480, 2) one-hot
        
        # Convert one-hot labels to class indices
        # [0,1] -> 1 (target), [1,0] -> 0 (alien/background)
        label = np.argmax(label, axis=-1)
        
        # Remove padded zero points
        non_zero_mask = np.any(coord != 0, axis=1) | np.any(feat != 0, axis=1)
        coord = coord[non_zero_mask]
        feat = feat[non_zero_mask]
        label = label[non_zero_mask]
        
        coord, feat, label = data_prepare(coord, feat, label, self.split, self.voxel_size, self.voxel_max, self.transform, self.shuffle_index)
        return coord, feat, label

    def __len__(self):
        return len(self.sample_list) * self.loop


if __name__ == '__main__':
    data_root = '/home/s2671222/Stratified-Transformer/Dataset/market/jam_hartleys_strawberry_300gm_1200_2048_segmentation_20480_12000'
    voxel_size = 0.04
    voxel_max = 80000

    point_data = Market(split='train', data_root=data_root, voxel_size=voxel_size, voxel_max=voxel_max)
    print('point data size:', point_data.__len__())
    
    import torch, time, random
    manual_seed = 123
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    
    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)
        
    train_loader = torch.utils.data.DataLoader(point_data, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_fn)
    
    for idx in range(1):
        end = time.time()
        voxel_num = []
        for i, (coord, feat, label, offset) in enumerate(train_loader):
            print('time: {}/{}--{}'.format(i+1, len(train_loader), time.time() - end))
            print('tag', coord.shape, feat.shape, label.shape, offset.shape, torch.unique(label))
            voxel_num.append(label.shape[0])
            end = time.time()
            if i >= 5:  # Just test first few batches
                break
    print(np.sort(np.array(voxel_num)))