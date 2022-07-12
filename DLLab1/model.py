import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.utils.data as data
from torchvision import transforms

Device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 用于标准化
Transformer = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ]
)


class MLP(nn.Module):
    """
    多层感知机模型
    这里使用了三个线性层，每个线性层后接一个Sigmoid激活层
    """
    def __init__(self):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(28 * 28, 256)
        self.l2 = nn.Linear(256, 64)
        self.l3 = nn.Linear(64, 10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 得先把图片变换为向量
        x = x.view(-1, 28 * 28)
        x = self.sigmoid(self.l1(x))
        x = self.sigmoid(self.l2(x))
        x = self.sigmoid(self.l3(x))
        return x


def train(model, epoch, batch_size):
    # 训练集加载
    train_data = torchvision.datasets.MNIST("./data", True, Transformer, download=True)
    train_loader = data.DataLoader(train_data, batch_size, True)

    # 损失函数及优化器
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # 模型训练过程
    for e in range(epoch):
        max_loss = 0
        for i, (img, label) in enumerate(train_loader):
            img = img.to(Device)
            label = label.to(Device)
            output = model(img)
            optimizer.zero_grad()   # 模型梯度清零
            loss = loss_func(output, label)
            loss.backward()     # 梯度反向传播
            optimizer.step()    # 参数更新
            if max_loss < loss.item():
                max_loss = loss.item()
        if (e + 1) % 5 == 0:
            print(f"epoch = {e + 1} / {epoch}\tloss = {str(max_loss)}", file=open("./MLP_torch.log", 'a'))


def eval(model, batch_size):
    # 测试集加载
    test_data = torchvision.datasets.MNIST("./data", False, Transformer, download=True)
    test_loader = data.DataLoader(test_data, batch_size, True)
    right_num = 0

    # 测试过程
    for img, label in test_loader:
        img = img.to(Device)
        label = label.to(Device)
        output = model(img)
        for i in range(img.shape[0]):
            if label[i].item() == output[i].argmax().item():
                right_num += 1
    print(f"right_num: {right_num} / {len(test_data)} \t Accuracy: {right_num / len(test_data)}",
          file=open("./MLP_torch.log", 'a'))
    return right_num / len(test_data)


def main():
    print(f"多层感知机-Pytorch版", file=open("./MLP_torch.log", 'w'))
    best_acc = 0.0
    # 重复训练五次找到最佳模型
    for t in range(5):
        model = MLP().to(Device)
        train(model, 100, 256)
        acc = eval(model, 256)
        if acc > best_acc:
            torch.save(model.state_dict(), "./model_torch_best.pth")
            best_acc = acc
        print(f"第{t + 1}/5次训练\t准确率：{acc}", file=open("./MLP_torch.log", 'a'))
    print(f"最高准确率：{best_acc}", file=open("./MLP_torch.log", 'a'))


if __name__ == "__main__":
    main()
