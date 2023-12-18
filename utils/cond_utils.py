from torchmetrics.functional import image_gradients
from torchvision.transforms.functional import equalize
from einops import rearrange, reduce, repeat
import torch

def noise_map(img):
        #FIXME: double check the calculation of noise map 
        dy, dx = image_gradients(img)
        dy, dx = torch.abs(dy), torch.abs(dx)
        map = torch.max(dy, dx)
        return map 
    
def color_map(img):
    mean_c = reduce(img, 'b c h w -> b 1 h w', 'mean') #[0,1]
    x_norm = img / (mean_c + 1e-8) # [0, 1/0.333 = 3] it make sense as point value 1 at that point > mean 0.333. therefore the weighting should be high
    x_norm /=3 # rescale it to [0,1]
    return x_norm.clamp(0,1)  # in [0,1]

def histro_equalize(img):
    x_uint8 = (img * 255).type(torch.uint8) 
    h = equalize(x_uint8)
    x_float = h.type(torch.float32) / 255.
    return x_float

def cond_data_transforms(img):
    c = color_map(img) #[3-5]
    n = noise_map(c) #[0-2]
    h = histro_equalize(img) #[6-8]
    final = torch.cat([img, h, c, n], dim=1) #[9-11]
    assert not torch.isnan(final).any()
    return final
