import torch


def train_one_epoch(model, data_loader, optimizer, loss_func, device):
    model.to(device)
    model.train()
    optimizer.zero_grad()
    loss_total = 0.0
    for img, label in data_loader:
        img = img.to(device)
        label = label.to(device)
        output = model(img)
        loss = loss_func(output, label)
        loss_total += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss_total / len(data_loader)

def eval_one_epoch(model, data_loader, loss_func, device):
    loss_total = 0.0
    right_num = 0
    total_num = 0
    model.to(device)
    model.eval()
    with torch.no_grad():
        for img, label in data_loader:
            img = img.to(device)
            label = label.to(device)
            output = model(img)
            loss = loss_func(output, label)
            right_num += (torch.argmax(output, dim=1) == label).sum().item()
            total_num += output.size(0)
            loss_total += loss.item()
    return loss_total / len(data_loader), right_num / total_num
