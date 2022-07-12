import paddle
import paddle.nn as nn
import paddle.optimizer as optim
import paddle.io as io
from paddle.vision.transforms import transforms
from paddle.vision.datasets import MNIST

Transformer = transforms.Compose(
    [
        # transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ]
)


class MLP(nn.Layer):
    def __init__(self):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(28 * 28, 256)
        self.l2 = nn.Linear(256, 64)
        self.l3 = nn.Linear(64, 10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: paddle.Tensor):
        x = x.reshape((-1, 28 * 28))
        x = self.sigmoid(self.l1(x))
        x = self.sigmoid(self.l2(x))
        x = self.sigmoid(self.l3(x))
        return x


def train(model, epoch, batch_size):
    train_data = MNIST("./data/MNIST/raw/train-images-idx3-ubyte.gz", "./data/MNIST/raw/train-labels-idx1-ubyte.gz",
                       "train",
                       transform=Transformer, download=True)
    train_loader = io.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(parameters=model.parameters())
    for e in range(epoch):
        max_loss = 0
        for i, (img, label) in enumerate(train_loader):
            img = img
            label = label
            output = model(img)
            optimizer.clear_grad()
            loss = loss_func(output, label)
            loss.backward()
            optimizer.step()
            if max_loss < loss.item():
                max_loss = loss.item()
        if (e + 1) % 5 == 0:
            print(f"epoch = {e + 1} / {epoch}\tloss = {str(max_loss)}", file=open("./MLP_paddle.log", 'a'))


def eval(model, batch_size):
    test_data = MNIST("./data/MNIST/raw/t10k-images-idx3-ubyte.gz", "./data/MNIST/raw/t10k-labels-idx1-ubyte.gz",
                      "test",
                      transform=Transformer, download=True)
    test_loader = io.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    right_num = 0
    for img, label in test_loader:
        img = img
        label = label
        output = model(img)
        for i in range(img.shape[0]):
            if label[i].item() == output[i].argmax().item():
                right_num += 1
    print(f"right_num: {right_num} / {len(test_data)} \t Accuracy: {right_num / len(test_data)}",
          file=open("./MLP_paddle.log", 'a'))
    return right_num / len(test_data)


def main():
    print(f"多层感知机-Paddle版", file=open("./MLP_paddle.log", 'w'))
    paddle.disable_static()
    paddle.device.set_device(paddle.device.get_device())
    best_acc = 0.0
    for t in range(5):
        model = MLP()
        train(model, 100, 256)
        acc = eval(model, 256)
        if acc > best_acc:
            paddle.save(model.state_dict(), "./model_paddle_best.pdparams")
            best_acc = acc
        print(f"第{t + 1}/5次训练\t准确率：{acc}", file=open("./MLP_paddle.log", 'a'))
    print(f"最高准确率：{best_acc}", file=open("./MLP_paddle.log", 'a'))



if __name__ == "__main__":
    main()
