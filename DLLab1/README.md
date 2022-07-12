# 深度学习 实验一

这个文档就是用来介绍代码目录的

- model.py：Pytorch版本的MLP结构、训练、测试代码
- model_paddle.py：百度PaddlePaddle版本的MLP相关代码，结构与Pytorch版本相同，这个就是存粹用来测试的
- MLP_torch.log：Pytorch版本训练日志
- MLP_paddle.log：Paddle版本训练日志，发现参数近乎相同，但paddle版本效果远差于Pytorch版本
- model_torch_best.pth：Pytorch版本训练好的参数，存的模型权值字典
- model_paddle_best.pdparams：Paddle版本训练好的参数，存的也是模型权值字典