import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image

from models import EDSR
from utils import preprocess, postprocess, calc_psnr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--image-file', type=str, required=True)
    parser.add_argument('--scale', type=int, default=3)
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = EDSR(scale_factor=args.scale).to(device)
    try:
        model.load_state_dict(torch.load(args.weights_file, map_location=device))
    except:
        state_dict = model.state_dict()
        for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage)['model_state_dict'].items():
            if n in state_dict.keys():
                state_dict[n].copy_(p)
            else:
                raise KeyError(n)

    model.eval()

    image = pil_image.open(args.image_file).convert('RGB')

    image_width = (image.width // args.scale) * args.scale
    image_height = (image.height // args.scale) * args.scale

    hr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    lr = hr.resize((hr.width // args.scale, hr.height // args.scale), resample=pil_image.BICUBIC)
    bicubic = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
    bicubic.save(args.image_file.replace('.', '_bicubic_x{}.'.format(args.scale)))

    lr = preprocess(lr).to(device)
    hr = preprocess(hr).to(device)
    bic = preprocess(bicubic).to(device)

    with torch.no_grad():
        preds = model(lr)
        print(preds.size())

    sr_psnr = calc_psnr(hr, preds)
    bic_psnr = calc_psnr(hr, bic)
    
    print('SR PSNR: {:.2f}'.format(sr_psnr))
    print('BIC PSNR: {:.2f}'.format(bic_psnr))

    output = postprocess(preds)
    output = pil_image.fromarray(output)
    output.save(args.image_file.replace('.', '_EDSR_x{}.'.format(args.scale)))
