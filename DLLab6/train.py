from metrics import compute_ssim, compute_psnr
import torch.nn as nn
import torch


# 损失函数用L1Loss
def train_one_epoch(model, train_dataloader, optimizer, loss_func, device):
    model.to(device)
    model.train()
    optimizer.zero_grad()
    loss_total = 0.0
    for LR, HR in train_dataloader:
        LR = LR.to(device)
        HR = HR.to(device)
        output = model(LR)
        loss = loss_func(output, HR)
        loss_total += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss_total / len(train_dataloader)


# 返回Loss,PSNR,SSIM
def eval_one_epoch(model, eval_dataloader, loss_func, device):
    loss_total = 0.0
    total_psnr = 0
    total_ssim = 0
    model.to(device)
    model.eval()
    with torch.no_grad():
        for lr, hr in eval_dataloader:
            lr = lr.to(device)
            hr = hr.to(device)
            output = model(lr)
            loss = loss_func(output, hr)
            psnr = compute_psnr(output, hr)
            ssim = compute_ssim(output, hr)
            loss_total += loss.item()
            total_psnr += psnr
            total_ssim += ssim
    return loss_total / len(eval_dataloader), total_psnr / len(eval_dataloader), total_ssim / len(eval_dataloader)
