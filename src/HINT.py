import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .dataset import Dataset
from .models import InpaintingModel
from .utils import Progbar, create_dir, stitch_images
from .metrics import PSNR
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import wandb
import lpips
import torchvision
from tqdm import tqdm

'''
This repo is modified basing on Edge-Connect
https://github.com/knazeri/edge-connect
'''

class HINT():
    def __init__(self, config):
        self.config = config

        model_name = 'inpaint'

        self.debug = False
        self.model_name = model_name

        self.inpaint_model = InpaintingModel(config).to(config.DEVICE)
        self.transf = torchvision.transforms.Compose(
            [
                torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        self.loss_fn_vgg = lpips.LPIPS(net='vgg').to(config.DEVICE)

        self.psnr = PSNR(255.0).to(config.DEVICE)
        self.cal_mae = nn.L1Loss(reduction='sum')
        
        self.best_ssim = 0

        #train mode
        if self.config.MODE == 1:
            self.train_dataset = Dataset(
                config,
                training='train'
            )
            
            self.val_dataset = Dataset(
                config,
                training='val'
            )

        # test mode
        if self.config.MODE == 2:
            self.test_dataset = Dataset(
                config,
                training='test'
            )

        self.results_path = os.path.join(config.PATH, 'results')

        if config.RESULTS is not None:
            self.results_path = os.path.join(config.RESULTS)

        if config.DEBUG is not None and config.DEBUG != 0:
            self.debug = True

        self.log_file = os.path.join(config.PATH, 'log_' + model_name + '.dat')

    def load(self):
        self.inpaint_model.load()


    def save(self):
        self.inpaint_model.save()


    def train(self):
        wandb.watch(self.inpaint_model, self.psnr, log='all', log_freq=10)
        
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=4,
            drop_last=True,
            shuffle=True
        )

        epoch = 0
        keep_training = True
        max_iteration = int(float((self.config.MAX_ITERS)))
        total = len(self.train_dataset)
        while(keep_training):
            epoch += 1
            print('\n\nTraining epoch: %d' % epoch)

            progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])


            for items in train_loader:
                self.inpaint_model.train()
                
                images, masks = self.cuda(*items)

                outputs_img, gen_loss, dis_loss, logs, gen_gan_loss, gen_l1_loss, gen_content_loss, gen_style_loss= self.inpaint_model.process(images,masks)
                outputs_merged = (outputs_img * masks) + (images * (1-masks))

                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()

                logs.append(('psnr', psnr.item()))
                logs.append(('mae', mae.item()))

                self.inpaint_model.backward(gen_loss, dis_loss)
                iteration = self.inpaint_model.iteration

                if iteration >= max_iteration:
                    keep_training = False
                    break

                logs = [
                    ("epoch", epoch),
                    ("iter", iteration),
                ] + logs

                progbar.add(len(images), values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('l_')])
                if iteration % 10 == 0:
                        wandb.log({
                            'gen_loss': gen_loss, 'l1_loss': gen_l1_loss, 'style_loss': gen_style_loss,
                            'perceptual loss': gen_content_loss, 'gen_gan_loss': gen_gan_loss,
                            'dis_loss': dis_loss
                            }, step=iteration)

                ###################### visualization
                if iteration % 40 == 0:
                    create_dir(self.results_path)
                    inputs = (images * (1 - masks))
                    images_joint = stitch_images(
                        self.postprocess(images),
                        self.postprocess(inputs),
                        self.postprocess(outputs_img),
                        self.postprocess(outputs_merged),
                        img_per_row=1
                    )

                    path_joint = os.path.join(self.results_path,self.model_name,'train_joint')

                    create_dir(path_joint)
                    images_joint.save(os.path.join(path_joint,f'epoch{epoch}.png'))

                # log model at checkpoints
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:
                    self.log(logs)

                # save model at checkpoints
                if self.config.SAVE_INTERVAL and iteration % self.config.SAVE_INTERVAL == 0:
                    # perform validation
                    current_psnr, current_ssim = self.val()
                    
                    if current_ssim > self.best_ssim:
                        print(f'current ssim of value {current_ssim} better than previous ssim of value {self.best_ssim}!')
                        self.best_ssim = current_ssim
                        self.save()

        print('\nEnd training....')


    def val(self):
        self.inpaint_model.eval()

        test_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=4,
        )
        
        psnr_list = []
        ssim_list = []
        # l1_list = []
        # lpips_list = []
        
        index = 0
        
        print('performing validation...')
        
        for items in test_loader:
            images, masks = self.cuda(*items)
            index += 1

            inputs = (images * (1 - masks))
            with torch.no_grad():
                outputs_img = self.inpaint_model(images, masks)
            outputs_merged = (outputs_img * masks) + (images * (1 - masks))
            
            psnr, ssim = self.metric(images, outputs_merged)
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            
            # if torch.cuda.is_available():
            #     pl = self.loss_fn_vgg(self.transf(outputs_merged[0].cpu()).cuda(), self.transf(images[0].cpu()).cuda()).item()
            #     lpips_list.append(pl)
            # else:
            #     pl = self.loss_fn_vgg(self.transf(outputs_merged[0].cpu()), self.transf(images[0].cpu())).item()
            #     lpips_list.append(pl)                
            
            # l1_loss = torch.nn.functional.l1_loss(outputs_merged, images, reduction='mean').item()
            # l1_list.append(l1_loss)

            # print("psnr:{}/{}  ssim:{}/{} l1:{}/{}  lpips:{}/{}  {}".format(psnr, np.average(psnr_list),
            #                                                                 ssim, np.average(ssim_list),
            #                                                                 l1_loss, np.average(l1_list),
            #                                                                 pl, np.average(lpips_list),
            #                                                                 len(ssim_list)))

            if index and index % self.config.SAVE_INTERVAL == 0:
                images_joint = stitch_images(
                    self.postprocess(images),
                    self.postprocess(inputs),
                    self.postprocess(outputs_img),
                    self.postprocess(outputs_merged),
                    img_per_row=1
                )

                path_joint = os.path.join(self.results_path,self.model_name,'val_joint')

                create_dir(path_joint)
                images_joint.save(os.path.join(path_joint,f'index{index}.png'))

        # report back metrics on validation set
        return np.mean(psnr_list), np.mean(ssim_list)


    def test(self):
        self.inpaint_model.eval()
        create_dir(self.results_path)

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )
        
        index = 0
        for items in tqdm(test_loader):
            images, masks = self.cuda(*items)
            index += 1

            with torch.no_grad():
                outputs_img = self.inpaint_model(images, masks)
            outputs_merged = (outputs_img * masks) + (images * (1 - masks))

            path_result = os.path.join(self.results_path, self.model_name,'test_result')
            create_dir(path_result)
            name_npy = self.test_dataset.load_name(index-1)[:-4]+'.npy'

            images_result = self.postprocess(outputs_merged, integer=False)[0].cpu().detach().numpy().squeeze()
            np.save(os.path.join(path_result,name_npy), images_result)


    def log(self, logs):
        with open(self.log_file, 'a') as f:
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))


    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)


    def postprocess(self, img, integer=True):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        
        if integer:
            return img.int()
        else:
            return img


    def metric(self, gt, pre):
        pre = pre.clamp_(0, 1) * 255.0
        pre = pre.permute(0, 2, 3, 1)
        pre = pre.detach().cpu().numpy().astype(np.uint8)[0]

        gt = gt.clamp_(0, 1) * 255.0
        gt = gt.permute(0, 2, 3, 1)
        gt = gt.cpu().detach().numpy().astype(np.uint8)[0]

        psnr = min(100, compare_psnr(gt, pre))
        ssim = compare_ssim(gt, pre, multichannel=True, data_range=255)

        return psnr, ssim