import argparse
import os
import math
import logging

import torch
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils

from torch.utils.data.dataloader import DataLoader
from torch import nn
from torch.cuda import amp

from models import EDSR
from utils import AverageMeter, ProgressMeter, calc_psnr
from dataset import Dataset
from tqdm import tqdm
from PIL import Image
from utils import preprocess

# 나비 테스트 이미지 경로 설정
test_image_path = 'examples/butterfly.png'
# 나비 테스트 이미지 불러오기
test_image = Image.open(test_image_path).convert('RGB')
# 나비 테스트 이미지 전처리
test_image = preprocess(test_image)


# python train.py --train-file data/train --eval-file data/eval --outputs-dir models --scale 3
# python train.py --train-file data/train --eval-file data/eval --outputs-dir models --scale 3 --checkpoint-file 
if __name__ == '__main__':
    """ 로그 설정 """
    logger = logging.getLogger(__name__)
    logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)

    """ Argparse 설정 """
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, required=True)
    parser.add_argument('--eval-file', type=str, required=True)
    parser.add_argument('--outputs-dir', type=str, required=True)
    parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--num-channels', type=int, default=3)
    parser.add_argument('--num-feats', type=int, default=64)
    parser.add_argument('--num-blocks', type=int, default=16)
    parser.add_argument('--res-scale', type=float, default=1.0)
    parser.add_argument('--psnr-lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=48)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--patch-size', type=int, default=120)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--checkpoint-file', type=str, default='checkpoint-file.pth')
    args = parser.parse_args()
    
    """ weight를 저장 할 경로 설정 """ 
    args.outputs_dir = os.path.join(args.outputs_dir,  f"EDSR_x{args.scale}")
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    """ GPU 디바이스 설정 """
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    """ Torch Seed 설정 """
    torch.manual_seed(args.seed)

    model = EDSR(scale_factor=args.scale, num_channels=args.num_channels, num_feats=args.num_feats, num_blocks=args.num_blocks, res_scale=args.res_scale).to(device)
    """ Loss 및 Optimizer 설정 """
    pixel_criterion = nn.L1Loss().to(device)
    psnr_optimizer = torch.optim.Adam(model.parameters(), args.psnr_lr, (0.9, 0.999))

    total_epoch = args.num_epochs
    start_epoch = 0
    best_psnr = 0

    """ 체크포인트 weight 불러오기 """
    if os.path.exists(args.checkpoint_file):
        checkpoint = torch.load(args.checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        psnr_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
        best_psnr = checkpoint['best_psnr']

    """ 로그 인포 프린트 하기 """
    logger.info(
                f"EDSR MODEL INFO:\n"
                f"\tScale:                         {args.scale}\n"
                f"\tNumber of channels:            {args.num_channels}\n"
                f"\tNumber of blocks:              {args.num_blocks}\n"
                f"\tNumber of features:            {args.num_feats}\n"
                f"\tResidual scale:                {args.res_scale}\n"

                f"EDSR TRAINING INFO:\n"
                f"\tTotal Epoch:                   {args.num_epochs}\n"
                f"\tStart Epoch:                   {start_epoch}\n"
                f"\tTrain directory path:          {args.train_file}\n"
                f"\tTest directory path:           {args.eval_file}\n"
                f"\tOutput weights directory path: {args.outputs_dir}\n"
                f"\tPSNR learning rate:            {args.psnr_lr}\n"
                f"\tPatch size:                    {args.patch_size}\n"
                f"\tBatch size:                    {args.batch_size}\n"
                )

    """ 스케줄러 설정 """
    psnr_scheduler = torch.optim.lr_scheduler.StepLR(psnr_optimizer, step_size=30, gamma=0.1)
    scaler = amp.GradScaler()

    """ 데이터셋 & 데이터셋 설정 """
    train_dataset = Dataset(args.train_file, args.patch_size, args.scale)
    train_dataloader = DataLoader(
                            dataset=train_dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=args.num_workers,
                            pin_memory=True
                        )
    eval_dataset = Dataset(args.eval_file, args.patch_size, args.scale)
    eval_dataloader = DataLoader(
                                dataset=eval_dataset, 
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=args.num_workers,
                                pin_memory=True
                                )
    
    """ 트레이닝 시작 & 테스트 시작"""
    for epoch in range(start_epoch, total_epoch):
        model.train()
        losses = AverageMeter(name="PSNR Loss", fmt=":.6f")
        psnr = AverageMeter(name="PSNR", fmt=":.6f")
        progress = ProgressMeter(
            num_batches=len(train_dataloader),
            meters=[losses, psnr],
            prefix=f"Epoch: [{epoch}]"
        )
        
        """  트레이닝 Epoch 시작 """
        for i, (lr, hr) in enumerate(train_dataloader):
            lr = lr.to(device)
            hr = hr.to(device)

            psnr_optimizer.zero_grad()

            with amp.autocast():
                preds = model(lr)
                loss = pixel_criterion(preds, hr)

            scaler.scale(loss).backward()
            scaler.step(psnr_optimizer)
            scaler.update()

            losses.update(loss.item(), len(lr))
            psnr.update(calc_psnr(preds, hr), len(lr))

            if i%100==0:
                progress.display(i)

        psnr_scheduler.step()

        """  테스트 Epoch 시작 """
        model.eval()
        with torch.no_grad():
            for i, (lr, hr) in enumerate(eval_dataloader):
                lr = lr.to(device)
                hr = hr.to(device)
                preds = model(lr)
                

        if psnr.avg > best_psnr:
            best_psnr = psnr.avg
            torch.save(
                model.state_dict(), os.path.join(args.outputs_dir, 'best.pth')
            )

        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': psnr_optimizer.state_dict(),
                'loss': loss,
                'best_psnr': best_psnr,
            }, os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch))
        )

        """ 나비 이미지 테스트 """
        with torch.no_grad():
            lr = test_image.to(device)
            preds = model(lr)
            vutils.save_image(preds.detach(), os.path.join(args.outputs_dir, f"PSNR_{epoch}.png"))