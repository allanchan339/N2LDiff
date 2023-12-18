from commons import *
import torch
from torch import nn
import torch.nn.functional as F
from utils.module_util import make_layer, initialize_weights

class Unet(nn.Module):
    def __init__(self, config, in_dim = 3, use_cond = False):
        super().__init__()
        dim = config.unet_dim
        out_dim = config.unet_outdim
        dim_mults = config.dim_mults
        in_dim = in_dim
        dims = [in_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        groups = 0
        self.use_attn = config.use_attn
        self.use_wn = config.use_wn
        self.use_in = config.use_instance_norm
        self.weight_init = config.weight_init
        self.stronger_cond = config.stronger_cond
        self.time_pos_emb = SinusoidalPosEmb(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            Mish(),
            nn.Linear(dim * 4, dim)
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        if self.stronger_cond:
            dim_adjust = 2
        else:
            dim_adjust = 1
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_emb_dim=dim, groups=groups, use_in = self.use_in),
                ResnetBlock(dim_out * dim_adjust, dim_out, time_emb_dim=dim, groups=groups, use_in = self.use_in),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))
            #TODO: check "cond" in resnetblock
             
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(
            mid_dim, mid_dim, time_emb_dim=dim, groups=groups, use_in = self.use_in)
        
        if self.use_attn:
            self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        else: 
            self.mid_attn = nn.Identity()

        self.mid_block2 = ResnetBlock(
            mid_dim, mid_dim, time_emb_dim=dim, groups=groups, use_in = self.use_in)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in,
                            time_emb_dim=dim, groups=groups, use_in = self.use_in),
                ResnetBlock(dim_in, dim_in, time_emb_dim=dim, groups=groups, use_in = self.use_in),
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
    
    def apply_layer_norm(self):
        def _apply_layer_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.layer_norm(m)
                print(f"| Weight norm is applied to {m}.")
    def forward(self, x, time, cond, feats_cond):
        t = self.time_pos_emb(time)
        t = self.mlp(t)

        h = []
        x = torch.cat((x, cond), dim=1)
        # cond = torch.cat(cond[2::4], 1) # from rrdb net
        # cond = self.cond_proj(torch.cat(cond[2::4], 1)) # cond[start at 2 -> every third item we take], finally get [20, 32*3, 20, 20]
        for i, (resnet, resnet2, downsample) in enumerate(self.downs):
            x = resnet(x, t)
            if self.stronger_cond:
                x = torch.cat((x, feats_cond[i]), dim=1)
            x = resnet2(x, t)
            # if i == 0:
                # x = x + cond
                # if hparams['res'] and hparams['up_input']:
                # x = x + self.up_proj(img_lr_up)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = upsample(x)

        x = self.final_conv(x)

        return x

    def make_generation_fast_(self):
        def remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(remove_weight_norm)

