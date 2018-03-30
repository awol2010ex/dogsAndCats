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
import os
from tensorboardX import SummaryWriter
#数据量每次随机10%取样
def SplitSampler(datalen, split=0.1):
    idx = list(range(datalen))
    random.shuffle(idx)
    split = int(datalen * split)
    train = SubsetRandomSampler(idx[split:])
    devel = SubsetRandomSampler(idx[:split])
    return train, devel


if __name__ == '__main__':
    writer = SummaryWriter()
    torch.manual_seed(1)

    EPOCH = 1000
    BATCH_SIZE = 100
    LR = 1e-4
    # 获取训练集dataset
    # 使用 ImageFolder 加载图片数据集，并进行transform变形(转成 64*64图片)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((64,64)),
        torchvision.transforms.ToTensor()
    ])
    

    #img = to_pil_image(training_data[0][0])
    #img.show()
    
    test_data= torchvision.datasets.ImageFolder("dataset/test_set" ,transform=transform)
    print(test_data.class_to_idx)
    # 将 Dataset 封装为 DataLoader
    test_data_loader = Data.DataLoader(dataset=test_data, shuffle=True, batch_size=1)

    #加载已训练模型（或中断）
    if os.path.exists("model/model.pkl"):
        cnn= torch.load("model/model.pkl")
    else:
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
    step=0
    for epoch in range(EPOCH):
        train, devel = SplitSampler(len(training_data) ,split=0.2)
        #取20%采样
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
            # 写loss指标进图表
            writer.add_scalar('loss', loss.data[0], step)
            i=i+1
            step=step+1
        
        print('Epoch[{}/{}], loss: {:.6f}'.format(epoch + 1, EPOCH, loss.data[0]))


        for name, param in cnn.named_parameters():
            writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)


        #触发dropout防止过拟合
        cnn.eval()



        #使用测试训练集查看该轮训练情况
        for i, (x, y) in enumerate(test_data_loader):
            batch_x = Variable(x)
            batch_y = Variable(y)
            # 输入训练数据
            output = cnn(batch_x)



            print('fact={},predict={}'.format(batch_y.data[0],torch.max(output, 1)[1].data.numpy().squeeze()))
        # 保存训练图

        torch.onnx.export(cnn, batch_x, "model/model.proto", verbose=True)
        writer.add_graph_onnx("model/model.proto")

        torch.save(cnn, 'model/model.pkl')
         