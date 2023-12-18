# for evaluate LOL dataset on model
import argparse
import yaml
from cond import Unet_cond, LitCond
from histroEncoder import LitHE, Unet_he
from diffusion import LitDiffusion, EnlightDiffusion
from cond2 import Unet_cond2
from model import Unet
from dataset import LitLOLDataModule
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from callbacks_pl import PaperImageExtractor, ImageLogger
from utils.gpuoption import gpuoption

def main(config):
    # to fix 4090 NCCL P2P bug in driver
    if gpuoption():
        print('NCCL P2P is configured to disabled, new driver should fix this bug')
    # seed
    pl.seed_everything(seed=config.seed, workers=True)
    
    # logger
    if config.use_wandb:
        group = config.group  # train or test_only
    
        project = config.project_diffusion_from_scratch

        project = config.group_test+project
        logger = WandbLogger(project=project,
                             entity=config.entity, group=group, config=config)

    litdataModule = LitLOLDataModule(config, [''], [config.test_folder])
    litdataModule.setup(stage='test')

    # model
    unet_cond = Unet_cond2(config, config.cond_in_dim, True)
    unet = Unet(config, in_dim=config.in_dim)
    diffusion = EnlightDiffusion(unet, config)

    assert config.diffusion_path != '', "diffusion.path must be a valid path"
    litmodel = LitDiffusion.load_from_checkpoint(
            config.diffusion_path, diffusion_model=diffusion, encoder=unet_cond, config=config, strict=False)

    trainer = pl.Trainer.from_argparse_args(
        config,
        logger=logger if config.use_wandb else True,
        callbacks=[PaperImageExtractor()] if config.PaperImageExtractor else [
            ImageLogger(None, litdataModule.test_dataloader(), config)]
    )

    trainer.test(model=litmodel,
                        datamodule=litdataModule)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", default='cfg/test/test_diffusion_from_scratch_LOL.yaml')
    config = parser.parse_args()

    with open(config.cfg, "r") as infile:
        cfg = yaml.full_load(infile)

    config = argparse.Namespace(**cfg)

    main(config)