# NOTE: skimage.__version__ == '0.17.1'
# Example run: python ref.py --test_dir_pred /root/autodl-tmp/Result/RetinexNet/low --test_dir_gt /root/autodl-tmp/Dataset/Clean_Images/low
import os
import numpy as np
from glob import glob
import cv2
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import torch
import lpips
import argparse
from natsort import natsorted
from compare.ref import _lpips, _psnr, _ssim, _ssim_gray, center_crop, transform
import pandas as pd
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):

    # NOTE: add sorted
    path_real = natsorted(glob(os.path.join(args.test_dir_gt, '*')))
    # path_fake = natsorted(glob(os.path.join(args.test_dir_pred, '*')))
    path_fake = natsorted(glob(os.path.join(args.test_dir_pred, '*')))
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)

    list_psnr = []
    list_ssim = []
    list_lpips = []

    alpha_list = np.array([0.9, 1.0])
    beta_list = np.array([0.5, 1.0])
    # gamma_list = np.arange(1.5, 5.1, 1.75)
    gamma_list = np.array([1.5, 5])
    df = pd.DataFrame(columns=['alpha', 'beta', 'gamma', 'psnr', 'ssim', 'lpips'])

    for alpha in alpha_list:
        for beta in beta_list:
            for gamma in gamma_list:
                for i in range(len(path_real)):

                    # read images
                    # print("==========================>")
                    # print("path_real[i]", path_real[i])
                    # print("path_fake[i]", path_fake[i])
                    img_real = cv2.imread(path_real[i])
                    img_fake = cv2.imread(path_fake[i]) # get high image

                    img_fake = img_fake.astype(np.float32) / 255.0
                    img_fake = beta * \
                        (alpha*img_fake)**gamma  # make low image
                    img_fake = 255*img_fake
                    img_fake = cv2.convertScaleAbs(img_fake)
                    if img_real.shape != img_fake.shape:
                        img_real = center_crop(img_real, img_fake)

                    # convert to torch tensor for lpips calculation
                    tes_real = transform(img_real).to(device)
                    tes_fake = transform(img_fake).to(device)

                    # calculate scores
                    psnr_num = _psnr(img_real, img_fake)
                    ssim_num = _ssim(img_real, img_fake)
                    lpips_num = _lpips(tes_real, tes_fake, loss_fn_alex)

                    # append to list
                    list_psnr.append(psnr_num)
                    list_ssim.append(ssim_num)
                    list_lpips.append(lpips_num)

                # Average score for the dataset
                # print("======={}=======>".format(args.test_dir_gt))
                # print("======={}=======>".format(args.test_dir_pred))
                # print("Average PSNR:", "%.3f" % (np.mean(list_psnr)))
                # print("Average SSIM:", "%.3f" % (np.mean(list_ssim)))
                # print("Average LPIPS:", "%.3f" % (np.mean(list_lpips)))

                dict_df = {'alpha': alpha, 'beta': beta, 'gamma': gamma, 'psnr': np.mean(
                    list_psnr), 'ssim': np.mean(list_ssim), 'lpips': np.mean(list_lpips)}
                
                df = df.append(dict_df, ignore_index=True)

                list_lpips.clear()
                list_psnr.clear()
                list_ssim.clear()

    max_psnr = df['psnr'].max()
    max_ssim = df['ssim'].max()
    min_lpips = df['lpips'].min()

    # print the maximum values
    print('Max PSNR:', max_psnr)
    print('Max SSIM:', max_ssim)
    print('Min LPIPS:', min_lpips)
    
    best_dict = {'alpha': '',
                'beta': '',
                'gamma': '',
                'psnr': max_psnr,
                'ssim': max_ssim,
                'lpips': min_lpips}
    
    df = df.append(best_dict, ignore_index=True)
    df = df.round(decimals=3)
    
    print(df)

    df.to_csv(os.path.join('metric.csv'), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--beta', default=0.75, type=float, #choices=range(0.5,1), 
                        help='part of equation for O=\beta(\alpha*I)^\gamma')
    parser.add_argument('--alpha', default=0.95, type=float, #choices=range(0.9, 1), 
                        help='part of equation for O=\beta(\alpha*I)^\gamma')
    parser.add_argument('--gamma', default=3.5, type=float, #choices=range(1.5, 5), 
        help='part of equation for O=\beta(\alpha*I)^\gamma')

    parser.add_argument('--test_dir_gt', type=str,
                        default='data/VE-LOL-L/VE-LOL-L-Cap-Full/test/low',
                        help='directory for clean images',
                        )
    parser.add_argument('--test_dir_pred', type=str,
                        default='data/VE-LOL-L/VE-LOL-L-Cap-Full/test/high',
                        help='directory for enhanced or restored images',
                        )
    args = parser.parse_args()
    main(args)
