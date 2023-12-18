from glob import glob
import argparse
import cv2
import torch
import os 
import lpips
import pandas as pd
from ref import _psnr, _ssim, _lpips, center_crop, transform
# args: target directory, ground truth directory, target_type
# return: list of file names that outperform the other 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    # Create an empty DataFrame with the 'Model' column
    df = pd.DataFrame(columns=['Model'])
    # NOTE: add sorted
    if args.target_type == 'low':
        path_gt = "compare/Figures/GT/low/eval15/high"
    elif args.target_type == 'VELOL_Real':
        path_gt = "compare/Figures/GT/VELOL_Real/test/high"
    path_real = sorted(glob(os.path.join(path_gt, '*')))

    list_models = sorted(glob(os.path.join(args.folder, '*')))
    list_models.remove('compare/Figures/GT')
    list_models.remove('compare/Figures/FromScratch')
    list_models.remove('compare/Figures/TwoStep')
    list_models.append(os.path.join(args.folder, args.test_target))

    for model in list_models:
        path_fake = sorted(glob(os.path.join(model, args.target_type, '*')))

        row_data = {'Model': model.split("/")[-1]}

        for i in range(len(path_real)):

            # read images
            # print("==========================>")
            # print("path_real[i]", path_real[i])
            # print("path_fake[i]", path_fake[i])
            img_real = cv2.imread(path_real[i])
            img_fake = cv2.imread(path_fake[i])

            if img_real.shape != img_fake.shape:
                img_real = center_crop(img_real, img_fake)

            # convert to torch tensor for lpips calculation
            tes_real = transform(img_real).to(device)
            tes_fake = transform(img_fake).to(device)

            if args.compare == 'psnr':
                # calculate scores
                psnr_num = _psnr(img_real, img_fake)
                metric = psnr_num
            elif args.compare == 'ssim':
                ssim_num = _ssim(img_real, img_fake)
                metric = ssim_num
            elif args.compare == 'lpips':
                lpips_num = _lpips(tes_real, tes_fake, loss_fn_alex)
                metric = lpips_num

            column_name = path_real[i].split("/")[-1]
            row_data[column_name] = metric

        # Append the row to the DataFrame
        df = df.append(row_data, ignore_index=True)

    df = df.set_index('Model')
    
    # Display the DataFrame
    if args.compare in ['psnr', 'ssim']: 
        new_row = (df.idxmax() == args.test_target).astype(int)
    else:
        new_row = (df.idxmin() == args.test_target).astype(int)
    
    df = df.append(new_row.rename("Target_highest"))

    df = df.round(3)
    if args.filter:
        df = df.transpose()
        df = df.loc[df['Target_highest'] > 0]
        df = df.transpose()

    print(df)
    df.to_csv(os.path.join('compare', args.target_type + '_' + args.compare + '.csv'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare two directories of images')
    parser.add_argument('--folder', type=str, help='folder name', default='compare/Figures')
    parser.add_argument('--test_target',  type=str, help='target directory', default='TwoStep')
    parser.add_argument('--target_type', type=str, help='target type', default='low', choices=['low', 'VELOL_Real'])
    parser.add_argument('--compare', default='psnr', help='compare method', choices=['psnr', 'ssim', 'lpips'])
    parser.add_argument('--filter', action='store_true', help='filter out the best', default=True)
    args = parser.parse_args()

    main(args)