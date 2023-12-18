from model import Unet
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelSummary,  StochasticWeightAveraging, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from dataset import LitLOLDataModule
from diffusion import LitDiffusion, EnlightDiffusion
import torch
from callbacks_pl import ImageLogger
from pytorch_lightning.strategies import DDPStrategy
from cond2 import Unet_cond2
import yaml
from utils.gpuoption import gpuoption

def main(train=False):

    # to fix 4090 NCCL P2P bug in driver
    if gpuoption():
        print('NCCL P2P is configured to disabled, new driver should fix this bug')

    # select config
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default='cfg/train/from_scratch.yaml')
    config = parser.parse_args()

    with open(config.cfg, "r") as infile:
        cfg = yaml.full_load(infile)

    config = argparse.Namespace(**cfg)
    
    # debug setting
    if config.fast_dev_run:
        if not config.accelerator == 'cpu':
            config.devices = 1
        # config.strategy = 'auto'
        config.use_wandb = False
        if config.batch_size >= 2:
            config.batch_size //= 2

    if config.use_dataset == 'LOL':
        train_folders = [config.train_folders_v1]
    elif config.use_dataset == "LOLv2":
        train_folders = [config.train_folders_v2]
    elif config.use_dataset == "LOL4K":
        train_folders = [config.train_folders_4k]

    elif config.use_dataset == 'LOL+LOLv2':
        train_folders = [config.train_folders_v1, config.train_folders_v2]
    elif config.use_dataset == 'LOL+LOLv2+VELOL':
        train_folders = [config.train_folders_v1,
                         config.train_folders_v2, config.train_folders_VE]
    else:
        NotImplementedError("dataset not supported")
    test_folder = config.test_folder

    # seed
    pl.seed_everything(seed=config.seed, workers=True)

    # strategy
    if config.strategy == 'ddp':
        strategy = DDPStrategy(static_graph=False, find_unused_parameters=True)
    else:
        strategy = 'auto'

    # data
    
    litdataModule = LitLOLDataModule(config, train_folders, [test_folder])
    litdataModule.setup()

    # logger
    if config.use_wandb:
        group = config.group  # train or test_only

    project = config.project_diffusion_from_scratch
    project = project if train else config.group_test+project
    logger = WandbLogger(project=project,
                        entity=config.entity, group=group, config=config)

    # for logging
    samples = next(iter(litdataModule.val_dataloader()))

    # model
    unet_cond = Unet_cond2(config, config.cond_in_dim, True)
    unet = Unet(config, in_dim=config.in_dim)

    diffusion = EnlightDiffusion(unet, config)

    if config.diffusion_path != '':
        litmodel = LitDiffusion.load_from_checkpoint(
            config.diffusion_path, diffusion_model=diffusion, encoder=unet_cond, config=config, strict=False)
    else:
        litmodel = LitDiffusion(diffusion, encoder=unet_cond, config=config)

    callbacks = [
        # StochasticWeightAveraging(swa_lrs=config.train_lr),
        ModelSummary(max_depth=3),
        LearningRateMonitor(),
        ModelCheckpoint(monitor='valid/combined',
                        save_last=False, mode='max', auto_insert_metric_name=False,
                        filename='epoch={epoch:02d}-monitor={valid/combined:.2f}'
                        ),
        ImageLogger(samples, litdataModule.test_dataloader(), config)            
    ]

    # trainer
    trainer = pl.Trainer.from_argparse_args(
    config,
    callbacks=callbacks,
    logger=logger if config.use_wandb else True,
    strategy=strategy
    )

    if train:
        trainer.fit(model=litmodel, datamodule=litdataModule)

        # after train, test it
        if not config.fast_dev_run:
            ckpt_path = trainer.checkpoint_callback.best_model_path

            litmodel = LitDiffusion.load_from_checkpoint(
                ckpt_path, diffusion_model=diffusion, encoder=unet_cond, config=config, strict=False)
        
            # new trainer
            trainer = pl.Trainer(
                accelerator=config.accelerator,
                devices=[config.devices[0]] if isinstance(config.devices, list) else config.devices,
                logger=logger if config.use_wandb else True,
                strategy=None,
                callbacks=callbacks,

            )
            trainer.test(litmodel, datamodule=litdataModule)

    else:
        # test only mode
        trainer.test(model=litmodel,
                        datamodule=litdataModule)


if __name__ == '__main__':
    # to make calculation much faster
    torch.set_float32_matmul_precision('high')
    train = True
    omit_training_error = True

    main(train=train)
