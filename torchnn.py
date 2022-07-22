# https://www.aiworkbox.com/lessons/how-to-use-the-view-method-to-manage-tensor-shape-in-pytorch#lesson-code-section
import torch.nn as nn
class Convolutional(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super(Convolutional, self).__init__()
        self.layer1 = nn.Sequential()
        self.layer1.add_module("Conv1", nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=3, padding=1))
        self.layer1.add_module("Relu1", nn.ReLU(inplace=False))
        self.layer2 = nn.Sequential()
        self.layer2.add_module("Conv2", nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=2))
        self.layer2.add_module("Relu2", nn.ReLU(inplace=False))
        self_fully_connected = nn.Linear(32 * 16 * 16, num_classes)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(-1, 32 * 16 * 16)
        x = self.fully_connected(x)
        return x