import torch
from model import Unet
import pytorch_lightning as pl
from torch.optim import AdamW, lr_scheduler
from lion_pytorch import Lion
import torch.nn.functional as F
from utils.cond_utils import histro_equalize
from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from operator import itemgetter
from torch import nn
from commons import *
from utils.module_util import make_layer, initialize_weights
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import pyiqa


class Unet_he(nn.Module):
    def __init__(self, config, in_dim=3, get_feats=False, use_cond=False):
        super().__init__()
        self.get_feats = get_feats

        dim = config.unet_dim
        out_dim = config.unet_outdim
        dim_mults = config.dim_mults
        self.in_dim = in_dim
        dims = [in_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        groups = 0
        self.use_attn = config.use_attn
        self.use_wn = config.he_use_wn
        self.use_in = config.use_instance_norm
        self.weight_init = config.weight_init
        self.on_res = config.he_on_res
        self.time_pos_emb = SinusoidalPosEmb(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            Mish(),
            nn.Linear(dim * 4, dim)
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        if use_cond:
            use_cond = 2
        else:
            use_cond = 1

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResnetBlockCond(dim_in * use_cond, dim_out,
                                time_emb_dim=dim, groups=groups, use_in=self.use_in),
                ResnetBlockCond(dim_out, dim_out, time_emb_dim=dim,
                                groups=groups, use_in=self.use_in),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlockCond(
            mid_dim, mid_dim, time_emb_dim=dim, groups=groups, use_in=self.use_in)

        if self.use_attn:
            self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        else:
            self.mid_attn = nn.Identity()

        self.mid_block2 = ResnetBlockCond(
            mid_dim, mid_dim, time_emb_dim=dim, groups=groups, use_in=self.use_in)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResnetBlockCond(dim_out * 2, dim_in,
                            time_emb_dim=dim, groups=groups, use_in=self.use_in),
                ResnetBlockCond(dim_in, dim_in, time_emb_dim=dim,
                            groups=groups, use_in=self.use_in),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        self.final_conv = nn.Sequential(
            Block(dim, dim, groups=groups),
            nn.Conv2d(dim, out_dim, 1)
        )

        # if hparams['res'] and hparams['up_input']:
        # self.up_proj = nn.Sequential(
        #         nn.ReflectionPad2d(1), nn.Conv2d(3, dim, 3),
        #     )
        if self.use_wn:
            self.apply_weight_norm()
        if self.weight_init:
            self.apply(initialize_weights)

    def apply_weight_norm(self):
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)
                # print(f"| Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def forward(self, x):
        input = x
        feats = []
        h = []
        # cond = torch.cat(cond[2::4], 1) # from rrdb net
        # cond = self.cond_proj(torch.cat(cond[2::4], 1)) # cond[start at 2 -> every third item we take], finally get [20, 32*3, 20, 20]
        for i, (resnet, resnet2, downsample) in enumerate(self.downs):
            x = resnet(x)
            x = resnet2(x)
            # if i == 0:
            # x = x + cond
            # if hparams['res'] and hparams['up_input']:
            # x = x + self.up_proj(img_lr_up)
            h.append(x)
            feats.append(x)
            x = downsample(x)

        x = self.mid_block1(x)
        if self.use_attn:
            x = self.mid_attn(x)
        x = self.mid_block2(x)

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x)
            x = resnet2(x)
            feats.append(x)
            x = upsample(x)

        x = self.final_conv(x)

        # additional layer to force in [0,1]
        if self.on_res:
            # x = F.sigmoid(x) # to make answer in 0,1
            if self.in_dim > 3:
                x += input[:, 0:3, :, :]  # img, h, c, n
            else:
                x += input[:, :, :, :]  # img, h, c, n

            x = torch.clip(x, 0, 1)
        else:
            x = F.sigmoid(x)  # to make answer in 0,1

        if self.get_feats:
            return x, feats
        else:
            return x


class LitHE(pl.LightningModule):
    def __init__(self, he_model, encoder, config, on_diffusion=False):
        super().__init__()
        self.model = he_model
        self.encoder = encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.save_hyperparameters(config)

        self.on_diffusion = on_diffusion
        if not self.on_diffusion:
            self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(self.device) # normalize = False to expect input in domain [-1,1]

        self.validation_step_outputs = []
        self.test_step_outputs = []

    def configure_optimizers(self):
        if self.hparams.he_optimizer == 'Lion':
            optimizer = Lion(self.model.parameters(), lr=self.hparams.he_lr)
        elif self.hparams.he_optimizer == 'AdamW':
            # only affect on self.model
            optimizer = AdamW(self.model.parameters(),
                              lr=self.hparams.he_lr)
        else:
            NotImplementedError()

        if self.hparams.he_scheduler == 'plateau':
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler.ReduceLROnPlateau(optimizer, mode=self.hparams.he_optim_mode, min_lr=self.hparams.he_min_lr, factor=self.hparams.he_factor, patience=self.hparams.he_patience),
                    "monitor": self.hparams.he_optim_target,
                    "frequency": 1,
                    "interval": "epoch"
                    # If "monitor" references validation metrics, then "frequency" should be set to a
                    # multiple of "trainer.check_val_every_n_epoch".
                },
            }
        elif self.hparams.he_scheduler == 'cosine':
            from utils.scheduler import CosineWarmupScheduler
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": CosineWarmupScheduler(optimizer, warmup=self.hparams.warmup, max_iters=self.hparams.max_iters),
                },
            }
        elif self.hparams.he_scheduler is None:
            return optimizer
        else:
            NotImplementedError()

    def loss_fn(self, output, ref):
        # output in [0,1] by clip
        # ref in [0, 1] by rescale
        if self.hparams.he_loss_type == 'l1':
            loss = F.smooth_l1_loss(output, ref)
        elif self.hparams.he_loss_type == 'l2':
            # loss = F.mse_loss(noise_pred, noise)
            loss = F.mse_loss(output, ref)
        elif self.hparams.he_loss_type == 'ssim':
            torch._assert(output.max() <= 1, 'output should be in [0,1]')
            msssim_metric = MultiScaleStructuralSimilarityIndexMeasure(
                betas=(0.0448, 0.2856, 0.3001, 0.2363), data_range=1.0).to(self.device)
            loss = 1 - msssim_metric(output, ref)  # as higher is better
        elif self.hparams.he_loss_type == 'lpips':
            if not self.on_diffusion:
                self.lpips.to(output.device)
                # should be grad_enabled
                loss = self.lpips(output, ref)
            else:
                SyntaxError('lpips loss is not loaded after HE embedded on diffusion')
        else:
            NotImplementedError()
        return loss

    def _valid_test_share_step(self, batch, stage):

        # use ssim and psnr to determine
        msssim_metric = MultiScaleStructuralSimilarityIndexMeasure(
            data_range=1.0).to(self.device)  # require full image to test
        psnr_metric = PeakSignalNoiseRatio().to(self.device)

        img_lr, img_hr = batch
        if self.hparams.he_input_he:
            input = histro_equalize(img_lr)
        else:
            input = img_lr

        if self.hparams.he_cond == 'img_c':
            with torch.no_grad():
                cond, feats = self.encoder(img_lr)
            input = torch.cat((input, cond), dim=1)

        if self.hparams.he_ref_he:
            ref = histro_equalize(img_hr)
        else:
            ref = img_hr
        output, feats = self.model(input)
        output = output.clamp(0, 1)  # output must be positive

        msssim = msssim_metric(output, ref)
        psnr = psnr_metric(output, ref)
        lpips = self.lpips(output, ref)
        ret = {'psnr': psnr, 'msssim': msssim, 'lpips': lpips}

        if stage == 'valid':
            self.validation_step_outputs.append(ret)
        elif stage == 'test':
            self.test_step_outputs.append(ret)

        # return  {k: float(v) for k, v in ret.items()}

    def training_step(self, batch, batch_idx):
        stage = 'train'
        img_lr, img_hr = batch
        if self.hparams.he_input_he:
            input = histro_equalize(img_lr)
        else:
            input = img_lr

        if self.hparams.he_cond == 'img_c':
            with torch.no_grad():
                cond, feats = self.encoder(img_lr)
            input = torch.cat((input, cond), dim=1)

        if self.hparams.he_ref_he:
            ref = histro_equalize(img_hr)
        else:
            ref = img_hr
        output, feats = self.model(input)
        loss = self.loss_fn(output, ref)

        self.log(f'{stage}/loss', loss, prog_bar=True,
                 on_step=True, logger=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        return self._valid_test_share_step(batch, 'valid')

    def test_step(self, batch, batch_idx):
        return self._valid_test_share_step(batch, 'test')

    def _valid_test_epoch_end(self, outputs, stage):
        # [loop1, loop2, loop3, ,,, loopN]
        outputs = self.all_gather(outputs)
        # in loop1: {psnr: [result_gpu1, result_gpu2,..., result_gpuN], ssim:, lpips:}
        all_psnr = list(map(itemgetter('psnr'), outputs))
        all_msssim = list(map(itemgetter('msssim'), outputs))
        all_lpips = list(map(itemgetter('lpips'), outputs))

        all_psnr = torch.mean(torch.stack(all_psnr))
        all_msssim = torch.mean(torch.stack(all_msssim))
        all_lpips = torch.mean(torch.stack(all_lpips))
        
        ret = {f'{stage}/psnr': all_psnr, f'{stage}/ms-ssim': all_msssim, f'{stage}/lpips': all_lpips}
        # dont know, but i think sync_dist will make the number constant
        self.log_dict(ret, True, True, sync_dist=True)

        if stage == 'valid':
            self.validation_step_outputs.clear()  # free memory
        elif stage == 'test':
            self.test_step_outputs.clear()  # free memory

        self.trainer.strategy.barrier()

    def on_validation_epoch_end(self) -> None:
        self._valid_test_epoch_end(self.validation_step_outputs, 'valid')

    def on_test_epoch_end(self) -> None:
        self._valid_test_epoch_end(self.test_step_outputs, 'test')

    def forward(self, img_lr):
        if self.hparams.he_input_he:
            input = histro_equalize(img_lr)
        else:
            input = img_lr

        if self.hparams.he_cond == 'img_c':
            with torch.no_grad():
                cond, feats = self.encoder(img_lr)
            input = torch.cat((input, cond), dim=1)

        output = self.model(input)
        return output


if __name__ == '__main__':

    # we want a unet with input [b, 12, h, w] to [b, 3, h, w]; with get_feats and no use_cond
    from types import SimpleNamespace
    from utils.utils import mergeConfig
    config_model = SimpleNamespace(
        decay_steps=100000,  # we should introduce EMA again?
        seed=42,
        unet_dim=64,
        unet_outdim=3,
        dim_mults=(1, 2, 2, 4, 4, 8),
        use_attn=False,
        use_wn=True,
        use_instance_norm=True,
        weight_init=True,
        rrdb_num_feat=32,
        rrdb_num_block=8,
        gc=32//2,
        in_nc=3,
        out_nc=3,
        in_dim=6,
        use_cond=True,
    )
    config_he = SimpleNamespace(
        he_in_dim=3,
        he_loss_type='lpips',  # ssim or lpips
        he_lr=2e-3,
        he_optimizer='AdamW',  # Lion or AdamW
        he_scheduler=None,  # cosine, plateau or None
        he_min_lr=2e-5,
        he_optim_target='valid/psnr',
        he_optim_mode='max',
        he_patience=20,
        he_factor=0.5,
        he_use_wn=True,
        he_num_workers=4,  # TODO: find out why > 1 is not working
        he_on_res=True,
        he_ref_he=False,
        he_input_he=True,
        he_cond='img_c'  # None, g(xl)
    )
    config_ckpt = SimpleNamespace(
            encoder_path='sd_cond_encoder/a5k3jyzv/checkpoints/epoch=837-monitor=0.93.ckpt',
            cond_on_res = False,
            # encoder_path='',
            he_path='sd_he/1u2suvm8/checkpoints/epoch=897-monitor=0.91.ckpt',
            diffusion_path='',
        )
    config_cond = SimpleNamespace(
        cond_in_dim=3*4,
        cond_loss_type='ssim',
        cond_lr=2e-3,  # is it good ?
        cond_optimizer='Lion',
        cond_scheduler=None,  # cosine, plateau or None
        cond_min_lr=2e-5,
        cond_optim_target='valid/ms-ssim',
        cond_optim_mode = 'max',
        cond_patience = 20,
        cond_factor = 0.5,
        cond_use_wn=True,
        cond_num_workers = 4,
    )

    config = mergeConfig(config_model, config_he, config_ckpt, config_cond)
    model_he = Unet_he(config, in_dim=config.he_in_dim, get_feats=True, use_cond=False)

    from cond import LitCond, Unet_cond
    unet_cond = Unet_cond(config, config.cond_in_dim, True)
    assert config.encoder_path !='', "encoder.path must be a valid path"
    encoder = LitCond.load_from_checkpoint(
                    config.encoder_path, cond_model = unet_cond, config = config)

    img = torch.rand((16, 3, 160, 160))

    output, feats = model_he(img)

    lithe = LitHE(model_he, encoder, config, on_diffusion=False)

    print()
