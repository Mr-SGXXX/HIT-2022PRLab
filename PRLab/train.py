import paddle

def train_one_epoch(model, data_loader, optimizer, loss_func):
    model.train()
    optimizer.clear_grad()
    loss_total = 0.0
    for img, label in data_loader:
        label = label.astype("int64")
        output = model(img)
        loss = loss_func(output, label)
        loss_total += loss
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
    return loss_total / len(data_loader)

def eval_one_epoch(model, data_loader, loss_func):
    loss_total = 0.0
    right_num = 0
    total_num = 0
    model.eval()
    with paddle.no_grad():
        for img, label in data_loader:
            label = label.astype("int64")
            output = model(img)
            loss = loss_func(output, label)
            right_num += paddle.sum(paddle.argmax(output, axis=1) == label).item()
            total_num += output.shape[0]
            loss_total += loss.item()
    return loss_total / len(data_loader), right_num / total_num