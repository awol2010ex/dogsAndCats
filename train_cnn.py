import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam,SGD
import torch.utils.data as Data
from torchvision.utils import make_grid, save_image
import torchvision
from model_cnn import CNN
from torch.utils.data.sampler import SubsetRandomSampler
import random
#数据量每次随机10%取样
def SplitSampler(datalen, split=0.1):
    idx = list(range(datalen))
    random.shuffle(idx)
    split = int(datalen * split)
    train = SubsetRandomSampler(idx[split:])
    devel = SubsetRandomSampler(idx[:split])
    return train, devel


if __name__ == '__main__':
    torch.manual_seed(1)

    EPOCH = 1000
    BATCH_SIZE = 100
    LR = 0.001
    # 获取训练集dataset
    # 使用 ImageFolder 加载图片数据集，并进行transform变形(转成 128*128图片)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((128,128)),
        torchvision.transforms.ToTensor()
    ])
    

    #img = to_pil_image(training_data[0][0])
    #img.show()
    
    test_data= torchvision.datasets.ImageFolder("dataset/test_set" ,transform=transform)
    print(test_data.class_to_idx)
    # 将 Dataset 封装为 DataLoader
    test_data_loader = Data.DataLoader(dataset=test_data, shuffle=True, batch_size=1)
    
    cnn = CNN()
    print(cnn)
    
    #optimizer
    optimizer = Adam(cnn.parameters(), lr=LR)
    
    #loss_fun
    loss_func = nn.CrossEntropyLoss()

    training_data = torchvision.datasets.ImageFolder("dataset/training_set", transform=transform)
    # 测试dataset显示图片
    # print(training_data[0][0].size())
    # to_pil_image = torchvision.transforms.ToPILImage()
    # 将 Dataset 封装为 DataLoader
    # training_data_loader = Data.DataLoader(dataset=training_data, shuffle=True, batch_size=100,num_workers =10)
    # 测试loader显示图片
    # data_iter = iter(training_data_loader)
    # images, labels = data_iter.next()  # tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
    # img = make_grid(images, 4)  # 拼成4*4网格图片，且会转成３通道
    # save_image(img, 'make_grid.png')
    # img = to_pil_image(img)
    # img.show()

    #training loop
    for epoch in range(EPOCH):
        train, devel = SplitSampler(len(training_data))
        training_data_loader = Data.DataLoader(dataset=training_data, shuffle=False, batch_size=BATCH_SIZE, num_workers=10, sampler=devel)

        i=0
        training_len=len(training_data_loader)
        data_iter = iter(training_data_loader)
        print("training_len=",training_len)
        while i < training_len:
            #print('Epoch[{}/{}/{}]'.format(epoch + 1, EPOCH, i+1))
            x,y =data_iter.next()
            #print(x.size())
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
            print('Epoch[{}/{}/{}], loss: {:.6f}'.format(epoch + 1, EPOCH, (i+1)*BATCH_SIZE,loss.data[0]))
            i=i+1
        
        print('Epoch[{}/{}], loss: {:.6f}'.format(epoch + 1, EPOCH, loss.data[0]))
        cnn.eval()
        for i, (x, y) in enumerate(test_data_loader):
            batch_x = Variable(x)
            batch_y = Variable(y)
            # 输入训练数据
            output = cnn(batch_x)
            print('fact={},predict={}'.format(batch_y.data[0],torch.max(output, 1)[1].data.numpy().squeeze()))
    torch.save(cnn, 'model/model.pkl')
         