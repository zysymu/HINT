import os
import glob
import torch
import numpy as np
import torchvision.transforms.functional as F
from torchvision import transforms
from PIL import Image

class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, training):
        super(Dataset, self).__init__()
        self.config = config
        self.training = training
        
        # get flist based on the dataset mode
        if self.training == 'train':
            flist = config.TRAIN_INPAINT_IMAGE_FLIST
        if self.training == 'val':
            flist = config.VAL_INPAINT_IMAGE_FLIST
        if self.training == 'test':
            flist = config.TEST_INPAINT_IMAGE_FLIST
        
        self.data = self.load_flist(flist)
        self.input_size = config.INPUT_SIZE
        
        # variables related to the dynamic size of the training mask depending on the number of examples
        if self.training == 'train':
            self.percentile = 1
        if self.training == 'val':
            self.percentile = 25
        
        self.current_iteration = 0
        self.max_iterations = int(float((self.config.MAX_ITERS)))
        self.fixed_percentile_at = 0.8  # pct of iterations after which percentile becomes fixed at 25


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        if self.training == 'train':
            # update percentile
            self.current_iteration += 1
            if self.current_iteration <= self.fixed_percentile_at * self.max_iterations:
                # linear growth from 5 to 24 over the first `fixed_percentile_at`% of iterations
                self.percentile = int(5 + (self.current_iteration / (self.fixed_percentile_at * self.max_iterations)) * 19)
            else:
                # set percentile to 25 for the remaining iterations
                self.percentile = 25

        item = self.load_item(index)
        
        return item


    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)


    def load_item(self, index):
        # load image
        img = np.load(self.data[index])
        img_t = self.process_image(img)
        
        # load mask (VERY IMPORTANT TO CONVERT IT TO FLOAT)
        mask_t = self.load_mask(img_t).float()

        # test (inference) mode 
        if self.training == 'test':
            img_t = torch.nan_to_num(img_t, nan=0)    
        
        # make 3 channels by copying the image
        img_t = img_t.repeat(3, 1, 1) # could try adding slices from before & after

        return img_t, mask_t
    
    
    def process_image(self, img):
        img_t = self.to_tensor(img)
        
        # apply transforms
        if self.input_size != 0:
            img_t = self.resize(self.input_size)(img_t)
        
        img_t = img_t / 255.0
        
        return img_t


    def load_mask(self, img_t):
        positive = 1
        
        # if mode set to training, generate mask on the fly
        if self.training in ['train', 'val']:
            # get random axis choice
            axis = np.random.choice(['i_line', 'x_line'], 1)[0]

            # crop subset
            if axis == 'i_line':
                sample_size = np.round(img_t.size(1)*(self.percentile/100)).astype('int')
                sample_start = np.random.choice(range(img_t.size(1)-sample_size), 1)[0]
                sample_end = sample_start+sample_size

                target_mask = torch.zeros(img_t.size(), dtype=int)
                target_mask[:, :, sample_start:sample_end] = positive

            else:
                sample_size = np.round(img_t.size(1)*(self.percentile/100)).astype('int')
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


    def resize(self, size):
        return transforms.Compose([
            transforms.Resize(size=size, interpolation=Image.BILINEAR, antialias=True),
        ])
