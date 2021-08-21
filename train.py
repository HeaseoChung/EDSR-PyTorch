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
from torch.utils.tensorboard import SummaryWriter

from models import EDSR
from utils import (
    AverageMeter,
    calc_psnr,
)
from dataset import Dataset


def net_trainer(train_dataloader, eval_dataloader, model, pixel_criterion, net_optimizer, epoch, best_psnr, scaler, writer, device, args):
        """ 모델 트레이닝 모드 """
        model.train()
        """ Loss & psnr 평균값 저장 인스턴스 """
        losses = AverageMeter(name="PSNR Loss", fmt=":.6f")
        psnr = AverageMeter(name="PSNR", fmt=":.6f")
        
        """  트레이닝 Epoch 시작 """
        for i, (lr, hr) in enumerate(train_dataloader):
            lr = lr.to(device)
            hr = hr.to(device)
            
            """ optimizer 초기화 """
            net_optimizer.zero_grad()

            """ precision mixed (float32 & float16) """
            with amp.autocast():
                preds = model(lr)
            loss = pixel_criterion(preds, hr)

            """ 학습이 잘 진행되고 있는지 확인용 이미지 저장 """
            if i == 0:
                vutils.save_image(lr.detach(), os.path.join(args.outputs_dir, f"LR_{epoch}.jpg"))
                vutils.save_image(hr.detach(), os.path.join(args.outputs_dir, f"HR_{epoch}.jpg"))
                vutils.save_image(preds.detach(), os.path.join(args.outputs_dir, f"preds_{epoch}.jpg"))
            
            """ Scaler 업데이트 """
            scaler.scale(loss).backward()
            scaler.step(net_optimizer)
            scaler.update()

            """ Loss 업데이트 """
            losses.update(loss.item(), len(lr))
        
        """ 1 epoch 마다 텐서보드 업데이트 """
        writer.add_scalar('L1Loss/train', losses.avg, epoch)

        """  테스트 Epoch 시작 """
        model.eval()
        for i, (lr, hr) in enumerate(eval_dataloader):
            lr = lr.to(device)
            hr = hr.to(device)
            with torch.no_grad():
                preds = model(lr)
            psnr.update(calc_psnr(preds, hr), len(lr))
    
        """ 1 epoch 마다 텐서보드 업데이트 """
        writer.add_scalar('psnr/test', psnr.avg, epoch)

        if psnr.avg > best_psnr:
            best_psnr = psnr.avg
            torch.save(
                model.state_dict(), os.path.join(args.outputs_dir, 'best.pth')
            )

        """ 모델 저장 """
        if epoch % 10 == 0:
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': net_optimizer.state_dict(),
                    'loss': loss,
                    'best_psnr': best_psnr,
                }, os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch))
            )


if __name__ == '__main__':
    """ 로그 설정 """
    logger = logging.getLogger(__name__)
    logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)

    """ Argparse 설정 """
    parser = argparse.ArgumentParser()
    """data args setup"""
    parser.add_argument('--train-file', type=str, required=True)
    parser.add_argument('--eval-file', type=str, required=True)
    parser.add_argument('--outputs-dir', type=str, required=True)

    """model args setup"""
    parser.add_argument('--scale', type=int, default=4, required=True)
    parser.add_argument('--n-channels', type=int, default=3)
    parser.add_argument('--n-resblocks', type=int, default=16)
    parser.add_argument('--n-feats', type=int, default=64)
    parser.add_argument('--res-scale', type=float, default=1.0)
    parser.add_argument('--resume-net', type=str, default='EDSR.pth')
    
    """Training details args setup"""
    parser.add_argument('--num-net-epochs', type=int, default=1000)
    parser.add_argument('--psnr-lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--patch-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--cuda', type=int, default=0)
    args = parser.parse_args()
    
    """ weight를 저장 할 경로 설정 """ 
    args.outputs_dir = os.path.join(args.outputs_dir,  f"EDSRx{args.scale}")
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    """ 텐서보드 설정 """
    writer = SummaryWriter(args.outputs_dir)

    """ GPU 디바이스 설정 """
    cudnn.benchmark = True
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    
    """ Torch Seed 설정 """
    torch.manual_seed(args.seed)

    """ EDSR 모델 설정 """
    generator = EDSR(args.scale, args.n_channels, args.n_feats, args.n_resblocks, args.res_scale).to(device)
    """ Loss 설정 """
    pixel_criterion = nn.L1Loss().to(device)
    """ Optimizer 설정 """
    net_optimizer = torch.optim.Adam(generator.parameters(), args.psnr_lr, (0.9, 0.99))
    """ 인터벌 에폭 설정 """
    interval_epoch = math.ceil(args.num_net_epochs // 8)
    epoch_indices = [interval_epoch, interval_epoch * 2, interval_epoch * 4, interval_epoch * 6]
    """ Learning rate 수치를 에폭에 따라 다르게 설정 """
    net_scheduler = torch.optim.lr_scheduler.MultiStepLR(net_optimizer, milestones=epoch_indices, gamma=0.5)
    """ Automatic mixed precision 설정 """
    scaler = amp.GradScaler()

    """ 시작 & 총 에폭 수 설정 """
    total_net_epoch = args.num_net_epochs
    start_net_epoch = 0
    
    """ best psnr 설정 """
    best_psnr = 0

    """ EDSR 체크포인트 weight 불러오기 """
    if os.path.exists(args.resume_net):
        checkpoint = torch.load(args.resume_net)
        generator.load_state_dict(checkpoint['model_state_dict'])
        net_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_net_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
        best_psnr = checkpoint['best_psnr']

    """ EDSR 로그 인포 프린트 하기 """
    logger.info(
                f"EDSR MODEL INFO:\n"
                f"\tScale factor:                  {args.scale}\n"
                f"\tNumber of channels:            {args.n_channels}\n"
                f"\tNumber of residual blocks:     {args.n_resblocks}\n"
                f"\tNumber of features:            {args.n_feats}\n"
                f"\tResidual scale:                {args.res_scale}\n"
                
                f"EDSR MODEL Training Details:\n"
                f"\tStart Epoch:                   {start_net_epoch}\n"
                f"\tTotal Epoch:                   {total_net_epoch}\n"
                f"\tTrain directory path:          {args.train_file}\n"
                f"\tTest directory path:           {args.eval_file}\n"
                f"\tTrained model directory path:  {args.outputs_dir}\n"
                f"\tPSNR learning rate:            {args.psnr_lr}\n"
                f"\tPatch size:                    {args.patch_size}\n"
                f"\tBatch size:                    {args.batch_size}\n"
                )

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
    """NET Training"""
    for epoch in range(start_net_epoch, total_net_epoch):
        net_trainer(train_dataloader=train_dataloader, eval_dataloader=eval_dataloader, model=generator, pixel_criterion=pixel_criterion, net_optimizer=net_optimizer, epoch=epoch, best_psnr=best_psnr, scaler=scaler, writer=writer, device=device, args=args)
        net_scheduler.step()
