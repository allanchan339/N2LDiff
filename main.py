from model import Unet
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelSummary,  StochasticWeightAveraging, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from dataset import LitLOLDataModule
from diffusion import LitDiffusion, EnlightDiffusion
import torch
from callbacks_pl import ColorMapLogger, HELogger, ImageLogger
from pytorch_lightning.strategies import DDPStrategy
from cond import Unet_cond, LitCond
from histroEncoder import LitHE, Unet_he
from cond2 import Unet_cond2
import yaml
from utils.gpuoption import gpuoption
def main(on_diffusion=False, on_encoder=False, on_HistroEncoder=False,  train=False, omit_training_error=False,         encoder_from_scratch=False,
         HistroEncoder_from_scratch=False
         ):

    # to fix 4090 NCCL P2P bug in driver
    if gpuoption():
        print('NCCL P2P is configured to disabled, new driver should fix this bug')

    # select config
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default='cfg/two_step.yaml')
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
    if on_encoder:
        config.num_workers = config.cond_num_workers
    if on_HistroEncoder:
        config.num_workers = config.he_num_workers

    litdataModule = LitLOLDataModule(config, train_folders, [test_folder])
    litdataModule.setup()

    # logger
    if config.use_wandb:
        group = config.group  # train or test_only
        if on_diffusion:
            if config.on_diffusion_from_scratch == True:
                project = config.project_diffusion_from_scratch
            else:
                project = config.project 
        elif on_HistroEncoder and on_encoder:
            project = config.project_he
        elif on_encoder:
            project = config.project_encoder
        elif on_HistroEncoder:
            project = config.project_he
        else:
            NotImplementedError("project not exists")

        project = project if train else config.group_test+project
        logger = WandbLogger(project=project,
                            entity=config.entity, group=group, config=config)

    # for logging
    samples = next(iter(litdataModule.val_dataloader()))

    if not on_diffusion:
        litmodel_cond_path = ''
        
        if on_HistroEncoder and on_encoder:
            unet_cond = Unet_cond(config, config.cond_in_dim, True)
            unet_he = Unet_he(config, config.he_in_dim, True)

            assert config.encoder_path !='', "encoder.path must be a valid path"
            encoder = LitCond.load_from_checkpoint(
                            config.encoder_path, cond_model = unet_cond, config = config)

            if config.he_path != '' and not HistroEncoder_from_scratch:
                litHE = LitHE.load_from_checkpoint(
                    config.he_path, he_model=unet_he, encoder=encoder,config=config)
            else:
                litHE = LitHE(unet_he, encoder, config)
            
            litmodel_cond = litHE

            callbacks = [
                ModelSummary(max_depth=-1),
                LearningRateMonitor(),
                ModelCheckpoint(monitor=config.cond_optim_target,  
                                save_last=False, mode='max', auto_insert_metric_name=False,
                                filename='epoch={epoch:02d}-monitor={valid/ms-ssim:.2f}'
                                ),
                HELogger(samples)
            ]

        elif on_HistroEncoder:
            unet_he = Unet_he(config, config.he_in_dim, True)

            if config.he_path != '' and not HistroEncoder_from_scratch:
                litHE = LitHE.load_from_checkpoint(
                    config.he_path, he_model=unet_he, config=config)
            else:
                litHE = LitHE(unet_he, config)
            
            litmodel_cond = litHE

            callbacks = [
                ModelSummary(max_depth=-1),
                LearningRateMonitor(),
                ModelCheckpoint(monitor=config.cond_optim_target,  
                                save_last=False, mode='max', auto_insert_metric_name=False,
                                filename='epoch={epoch:02d}-monitor={valid/ms-ssim:.2f}'
                                ),
                HELogger(samples)
            ]

        elif on_encoder:
            # model
            unet_cond = Unet_cond(config, config.cond_in_dim, True)

            if config.encoder_path != '' and not encoder_from_scratch:
                litCond = LitCond.load_from_checkpoint(
                    config.encoder_path, cond_model=unet_cond, config=config)
            else:
                litCond = LitCond(unet_cond, config)
            
            litmodel_cond = litCond
            litmodel_cond_path = config.encoder_path

            # callbacks
            callbacks = [
                ModelSummary(max_depth=-1),
                LearningRateMonitor(),
                ModelCheckpoint(monitor=config.cond_optim_target,  
                                save_last=False, mode='max', auto_insert_metric_name=False,
                                filename='epoch={epoch:02d}-monitor={valid/ms-ssim:.2f}'
                                ),
                ColorMapLogger(samples)
            ]

        # trainer
        trainer = pl.Trainer.from_argparse_args(
            config,
            callbacks=callbacks,
            logger=logger if config.use_wandb else True,
            strategy=strategy
        )

        if train:
            # either train from scratch or continues train
            if omit_training_error:
                try:
                    trainer.fit(model=litmodel_cond, datamodule=litdataModule)
                except Exception as e:
                    print("Exception Type:",
                          e.args[0][0], "Message:", e.args[0][1])
            else:
                trainer.fit(model=litmodel_cond, datamodule=litdataModule)

            # after train, test it
            if not config.fast_dev_run:
                trainer.test(ckpt_path='best', datamodule=litdataModule)

        else:
            # test only mode
            trainer.test(ckpt_path=litmodel_cond_path,
                         datamodule=litdataModule)

    elif not config.on_diffusion_from_scratch:
        # model
        unet_he = Unet_he(config, config.he_in_dim, True)
        unet_cond = Unet_cond(config, config.cond_in_dim, True)
        unet = Unet(config, in_dim=config.in_dim)  
        diffusion = EnlightDiffusion(unet, config)

        # callbacks
        callbacks = [
            ModelSummary(max_depth=3),
            LearningRateMonitor(),
            ModelCheckpoint(monitor='valid/combined',
                            save_last=False, mode='max', auto_insert_metric_name=False,
                            filename='epoch={epoch:02d}-monitor={valid/combined:.2f}'
                            ),
            ImageLogger(samples, litdataModule.test_dataloader(), config)
        ]
        #FIXME: the backward is not doing, wtf? 
        # trainer
        trainer = pl.Trainer.from_argparse_args(
            config,
            callbacks=callbacks,
            logger=logger if config.use_wandb else True,
            strategy=strategy
        )
        if train:
            assert config.encoder_path !='', "encoder.path must be a valid path"
            encoder = LitCond.load_from_checkpoint(
                        config.encoder_path, cond_model = unet_cond, config = config)

            assert config.he_path !='', "he.path must be a valid path"
            litHE = LitHE.load_from_checkpoint(
                config.he_path, he_model=unet_he, encoder=encoder, config=config, strict=False)
        

            if config.diffusion_path != '':
                litmodel = LitDiffusion.load_from_checkpoint(
                    config.diffusion_path, diffusion_model=diffusion, encoder=encoder, config=config, histroEncoder=litHE, strict=False)
            else:
                litmodel = LitDiffusion(diffusion, encoder, config, litHE)

            if omit_training_error:
                try:
                    trainer.fit(model=litmodel, datamodule=litdataModule)
                except Exception as e:
                    print("Exception Type:",
                          e.args[0][0], "Message:", e.args[0][1])
            else:
                trainer.fit(model=litmodel, datamodule=litdataModule)

            # after train, test it
            if not config.fast_dev_run:
                ckpt_path = trainer.checkpoint_callback.best_model_path
                litmodel = LitDiffusion.load_from_checkpoint(
                    ckpt_path, diffusion_model=diffusion, encoder=encoder, config=config, histroEncoder=litHE, strict=False)

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
            encoder = LitCond(unet_cond, config)
            litHE = LitHE(unet_he, encoder, config)
            
            assert config.diffusion_path !='', "diffusion.path must be a valid path"
            litmodel = LitDiffusion.load_from_checkpoint(
                    config.diffusion_path, diffusion_model=diffusion, encoder=encoder, config=config, histroEncoder=litHE, strict=False)

            trainer.test(model=litmodel,
                         datamodule=litdataModule)

    else: 
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
    on_diffusion = True
    on_encoder = False
    encoder_from_scratch = False
    on_HistroEncoder = False
    HistroEncoder_from_scratch = False
    train = True
    omit_training_error = True

    main(on_diffusion=on_diffusion, on_encoder=on_encoder, on_HistroEncoder=on_HistroEncoder,
         train=train, omit_training_error=omit_training_error,
        encoder_from_scratch=encoder_from_scratch,     
         HistroEncoder_from_scratch = HistroEncoder_from_scratch
         )
