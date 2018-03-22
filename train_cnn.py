import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam
import torch.utils.data as Data
from torchvision.utils import make_grid, save_image
import torchvision
from model_cnn import CNN
torch.manual_seed(1)

EPOCH = 1000
BATCH_SIZE = 10
LR = 0.001
# 获取训练集dataset
# 使用 ImageFolder 加载图片数据集，并进行transform变形(转成 128*128图片)
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((128,128)),
    torchvision.transforms.ToTensor()
])
training_data= torchvision.datasets.ImageFolder("dataset/training_set" ,transform=transform)
#测试dataset显示图片
print(training_data[0][0].size())
#to_pil_image = torchvision.transforms.ToPILImage()

#img = to_pil_image(training_data[0][0])
#img.show()

# 将 Dataset 封装为 DataLoader
training_data_loader = Data.DataLoader(dataset=training_data, shuffle=True, batch_size=16)
#测试loader显示图片
#data_iter = iter(training_data_loader)
#images, labels = data_iter.next()  # tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
#img = make_grid(images, 4)  # 拼成4*4网格图片，且会转成３通道
#save_image(img, 'make_grid.png')
#img = to_pil_image(img)
#img.show()

cnn = CNN()
print(cnn)

#optimizer
optimizer = Adam(cnn.parameters(), lr=LR)

#loss_fun
loss_func = nn.CrossEntropyLoss()


#training loop
for epoch in range(EPOCH):

    for i, (x, y) in enumerate(training_data_loader):
        batch_x = Variable(x)
        batch_y = Variable(y)
        #输入训练数据
        output = cnn(batch_x)
        #计算误差
        loss = loss_func(output, batch_y)

        #清空上一次梯度
        optimizer.zero_grad()
        #误差反向传递
        loss.backward()

        #优化器参数更新
        optimizer.step()
    cnn.eval()
    print('Epoch[{}/{}], loss: {:.6f}'.format(epoch + 1, EPOCH, loss.data[0]))
    if loss.data[0]<0.001:
        torch.save(cnn, 'model/model.pkl')
        break;