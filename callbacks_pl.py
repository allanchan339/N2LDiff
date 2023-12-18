import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from torchvision.utils import make_grid
from utils.cond_utils import color_map, histro_equalize, noise_map
from diffusion import center_adjustment, on_cond_or_center_selector
from torchvision.transforms import ToPILImage
import os
from glob import glob
import cv2
import torch
import shutil
import numpy as np
from PIL import Image
from einops import rearrange, reduce, repeat
from diffusion import on_cond_selector
from utils.cond_utils import color_map, histro_equalize, noise_map
from utils.utils import pad_to_multiple, unpad_from_multiple

class ImageLogger(pl.Callback):
    def __init__(self, samples, test_dataloader ,config):
        super().__init__()
        self.samples = samples 
        self.test_dataloader = test_dataloader
        self.config = config

    def _share_step(self, stage, samples, trainer, pl_module):
        path_real = sorted(glob(os.path.join(self.config.test_folder, 'high' , '*')))
        path_fake = sorted(glob(os.path.join(self.config.results_folder, '*')))
        input_list = [os.path.basename(file_path) for file_path in path_real]

        img_lr, img_hr, img_lr_name = samples
        img_lr = img_lr.to(pl_module.device)
        img_hr = img_hr.to(pl_module.device)

        # pad and unpad
        if self.config.paddingMode:
            img_lr = pad_to_multiple(img_lr, 32)

        with torch.no_grad():
            center, feats_cond = pl_module.encoder(img_lr)

        if self.config.paddingMode:
            center = unpad_from_multiple(center, img_hr.shape)

        positions = [input_list.index(name) for name in img_lr_name if name in input_list]

        images = [torch.from_numpy(cv2.cvtColor(cv2.imread(path_fake[n]), cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()/255 for n in positions]

        if isinstance(trainer.logger, WandbLogger):
            if stage == 'test':
                if self.config.paddingMode:
                    img_lr = unpad_from_multiple(img_lr, img_hr.shape)

                for i, n in enumerate(positions):
                    ret ={
                            f"{stage}/output": [
                                wandb.Image(images[i], caption='pred'),
                                wandb.Image(img_lr[i], caption='input'),
                                wandb.Image(img_hr[i], caption='ref'),
                                wandb.Image(center[i], caption='center'),
                            ]
                        }
                    
                    trainer.logger.experiment.log(ret)

            elif stage == 'valid':
                ret ={
                        f"{stage}/output": [
                            wandb.Image(make_grid(images, nrow=5), caption='pred'),
                            wandb.Image(make_grid(img_lr, nrow=5), caption='input'),
                            wandb.Image(make_grid(img_hr, nrow=5), caption='ref'),
                            wandb.Image(make_grid(center, nrow=5), caption='center'),
                        ]
                    }

                trainer.logger.experiment.log(ret)


    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.strategy.is_global_zero:
            self._share_step('valid', self.samples, trainer, pl_module)
        trainer.strategy.barrier()

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.strategy.is_global_zero:
            for batch in self.test_dataloader:
                self._share_step('test', batch, trainer, pl_module)
        trainer.strategy.barrier()

class PaperImageExtractor(pl.Callback):
    
    def convertAndSave(self, img, name, path):
        # if 4 dim, cut to 3 dim
        if len(img.shape) == 4:
            img = img[0]
        
        # if in cuda, move to cpu 
        if img.is_cuda:
            img = img.detach().cpu()

        # check if in range [0,1], else redomain
        if img.min() < 0 and img.max() > 1:
            img = (img + 1)/2

        # clip to [0,1]
        img = torch.clamp(img, 0, 1)

        # convert to numpy
        img = (img.permute(1,2,0).numpy()*255).astype(np.uint8)

        # convert to PIL Image
        img = Image.fromarray(img)

        # save
        img.save(os.path.join(path, f'{name}.png'))

    def __init__(self) -> None:
        super().__init__()
        self.sub_folder = 'paper_images'
        self.t = 15
        
    def on_test_start(self, trainer, pl_module) -> None:
        device = pl_module.device
        self.config = pl_module.hparams
        # we need one sample only
        self.samples = next(iter(trainer.datamodule.test_dataloader()))
        self.img_lr, self.img_hr, self.img_lr_name = self.samples

        for i, name in enumerate(self.img_lr_name):
            img_lr = self.img_lr[i]
            img_hr = self.img_hr[i]

            name = name.split('/')[-1]
            name = name.split('.')[0]

            # make the subfolder for name, if exists then delete, then create
            path = os.path.join(self.sub_folder, name)
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)
            
            print(f'Created folder {path}')

            # convert img_lr[i] to PIL image and save 

            self.convertAndSave(img_lr, 'x_L', path)
            self.convertAndSave(img_hr, 'x_H', path)
            self.convertAndSave(img_hr - img_lr, 'x_0', path)

            # we need to use forward to get the pred_noise, and gt_noise and fit to [0,1]

            t = torch.LongTensor([self.t]).repeat(1).to(device)

            x_start = pl_module.model.img2res(
                            img_hr, img_lr) # to make
            x_start4 = rearrange(x_start, 'c h w -> 1 c h w').to(device)
            # GPU required 
            img_lr4 = rearrange(img_lr, 'c h w -> 1 c h w').to(device)
            cond4 = on_cond_selector(img_lr4,  self.config.on_cond)
            
            # from cond_utils import 
            c4 = color_map(img_lr4)
            n4 = noise_map(c4)
            h4 = histro_equalize(img_lr4)
            
            self.convertAndSave(c4, 'color_map', path)
            self.convertAndSave(n4, 'noise_map', path)
            self.convertAndSave(h4, 'he_map', path)

            center4, feats_cond = pl_module.encoder(img_lr4)
            
            self.convertAndSave(center4, 'center', path)


            noise_pred4, noise4 = pl_module.model(x_start4, t, cond4, center4, feats_cond=feats_cond)

            x_95 = pl_module.model.q_sample(x_start4, t, center4, noise4)
            
            x_95_pred = pl_module.model.q_sample(x_start4, t, center4, noise_pred4)

            self.convertAndSave(x_95_pred, f'x_noise_pred_{self.t}', path)
            self.convertAndSave(x_95, f'x_noise_{self.t}', path)
            self.convertAndSave(noise_pred4, 'noise_pred', path)
            self.convertAndSave(noise4, 'noise', path)

            # we need all time steps in sampling for the reconstruction and fit to [0,1]
            print(f'Sampling...{name}')
            imgs = pl_module.sample(img_lr4, True)
            imgs = reversed(imgs) # x_100 to x_0
            # we need to save the images
            for i, img in enumerate(imgs):
                self.convertAndSave(img, f'x_pred_{i}', path)
