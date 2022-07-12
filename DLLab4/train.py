import torch


def train_one_epoch_shopping(model, data_loader, optimizer, loss_func, device, emb):
    model.to(device)
    emb.to(device)
    model.train()
    optimizer.zero_grad()
    loss_total = 0.0
    for word_list, label in data_loader:
        word_list = word_list.to(device)
        word_list = emb(word_list)
        label = label.to(device)
        output = model(word_list)
        loss = loss_func(output, label)
        loss_total += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss_total / len(data_loader)


def eval_one_epoch_shopping(model, data_loader, loss_func, device, emb):
    loss_total = 0.0
    right_num = 0
    total_num = 0
    model.to(device)
    emb.to(device)
    model.eval()
    with torch.no_grad():
        for word_list, label in data_loader:
            word_list = word_list.to(device)
            word_list = emb(word_list)
            label = label.to(device)
            output = model(word_list)
            loss = loss_func(output, label)
            right_num += (torch.argmax(output, dim=1) == label).sum().item()
            total_num += output.size(0)
            loss_total += loss.item()
    return loss_total / len(data_loader), right_num / total_num


def train_one_epoch_climate(model, data_loader, optimizer, loss_func, device):
    model.train()
    optimizer.zero_grad()
    loss_total = 0.0
    for src_data, dst_data in data_loader:
        src_data = src_data.to(device)
        dst_data = dst_data.to(device)
        output = model(src_data)
        loss = loss_func(output, dst_data)
        loss_total += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss_total / len(data_loader)
