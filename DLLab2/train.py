import torch
import torch.optim as optim
import torch.nn as nn


def train_one_epoch(model: nn.Module, device, data_loader, optimizer: optim.Optimizer, loss_func):
    model.train()
    model = model.to(device)
    optimizer.zero_grad()
    loss_total = 0.
    for img, label in data_loader:
        img = img.to(device)
        label = label.to(device)
        output = model(img)
        loss = loss_func(output, label)
        loss_total += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss_total / len(data_loader)

def eval_model(model: nn.Module, device, data_loader, loss_func):
    model.eval()
    model = model.to(device)
    right_num = 0
    total_num = 0
    loss_total = 0.
    for img, label in data_loader:
        img = img.to(device)
        label = label.to(device)
        output = model(img)
        loss = loss_func(output, label)
        right_num += (torch.argmax(output, dim=1) == label).sum().item()
        total_num += label.size(0)
        loss_total += loss.item()
    return loss_total / len(data_loader), right_num / total_num
