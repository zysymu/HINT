import os
import glob
import torch
import numpy as np
import torchvision.transforms.functional as F
from torchvision import transforms
from PIL import Image

class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, flist, mask_flist, augment=True, training=True):
        super(Dataset, self).__init__()
        self.config = config
        self.augment = augment
        self.training = training

        self.data = self.load_flist(flist)
        self.mask_data = self.load_flist(mask_flist)

        self.input_size = config.INPUT_SIZE
        self.mask = config.MASK

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        item = self.load_item(index)
        return item

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):
        size = self.input_size

        # load image
        img = np.load(self.data[index])
        img_t = self.to_tensor(img)
        
        # apply transforms
        if size != 0:
            img_t = self.resize(size)(img_t)

        # load mask (VERY IMPORTANT TO CONVERT IT TO FLOAT)
        mask_t = self.load_mask(img_t).float()
        
        # replace nans with 0s in test data
        if not self.training:
            img_t = torch.nan_to_num(img_t, nan=0)

        return img_t, mask_t


    def load_mask(self, img_t):
        positive = 1
        
        # if mode set to training, generate mask on the fly
        if self.training:
            # get random axis choice
            percentile = 25
            axis = np.random.choice(['i_line', 'x_line'], 1)[0]

            # crop subset
            if axis == 'i_line':
                sample_size = np.round(img_t.size(1)*(percentile/100)).astype('int')
                sample_start = np.random.choice(range(img_t.size(1)-sample_size), 1)[0]
                sample_end = sample_start+sample_size

                target_mask = torch.zeros(img_t.size(), dtype=int)
                target_mask[:, :, sample_start:sample_end] = positive

            else:
                sample_size = np.round(img_t.size(1)*(percentile/100)).astype('int')
                sample_start = np.random.choice(range(img_t.size(1)-sample_size), 1)[0]
                sample_end = sample_start+sample_size

                target_mask = torch.zeros(img_t.size(), dtype=int)
                target_mask[:, sample_start:sample_end, :] = positive
            
            return target_mask
        
        # if we're on test mode, extract mask from image (our image already has a strip of nan values)
        else:
            target_mask = torch.zeros(img_t.size(), dtype=int)
            nans = torch.argwhere(torch.isnan(img_t[:, :, :]))
            indices = [(nans[:, i].min(), nans[:, i].max()+1) for i in range(len(img_t.size()))]
            target_mask[indices[0][0]:indices[0][1], indices[1][0]:indices[1][1], indices[2][0]:indices[2][1]] = positive
            
            return target_mask


    def to_tensor(self, img):
        img_t = F.to_tensor(img).float()
        return img_t


    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.npy'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=str, encoding='utf-8')
                except Exception as e:
                    print(e)
                    return [flist]
        
        return []
    

    def resize(self, load_size):
        return transforms.Compose([
            transforms.Resize(size=load_size, interpolation=Image.BILINEAR),
        ])
