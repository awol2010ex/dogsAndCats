import torch
import torch.nn as nn
from torch.autograd import Variable

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # (3,128,128)
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3,
                      stride=1, padding=1),
            # 想要con2d卷积出来的图片尺寸没有变化, padding=(kernel_size-1)/2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)

        )  # (16,64,64)
        self.conv2 = nn.Sequential(  # (16,64,64)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)

        )
        # (32,32,32)
        self.conv3 = nn.Sequential(  # (32,32,32)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)

        )
        # (64,16,16)
        self.conv4 = nn.Sequential(  # (64,16,16)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)

        )
        # (128,8,8)

        self.out=nn.Linear(128*8*8 ,  2)   #cat+dog=2

    def forward(self, x):

        x = self.conv1(x)
        x =nn.Dropout(0.05)(x)
        x = self.conv2(x)
        x = nn.Dropout(0.05)(x)
        x = self.conv3(x)
        x = nn.Dropout(0.05)(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)  # flat (batch_size, 128*8*8)
        output = self.out(x)
        return output