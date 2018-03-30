import torch
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
from model_cnn import CNN
torch.manual_seed(1)

# 获取测试训练集dataset
# 使用 ImageFolder 加载图片数据集，并进行transform变形(转成 128*128图片)
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((64,64)),
    torchvision.transforms.ToTensor()
])
test_data= torchvision.datasets.ImageFolder("dataset/test_set" ,transform=transform)
print(test_data.class_to_idx)
# 将 Dataset 封装为 DataLoader
test_data_loader = Data.DataLoader(dataset=test_data, shuffle=True, batch_size=1)

#训练好的模型
cnn = torch.load('model/model.pkl')
print(cnn)

for i, (x, y) in enumerate(test_data_loader):
    batch_x = Variable(x)
    batch_y = Variable(y)
    # 输入训练数据
    output = cnn(batch_x)
    print('fact={},predict={}'.format(batch_y.data[0],torch.max(output, 1)[1].data.numpy().squeeze()))