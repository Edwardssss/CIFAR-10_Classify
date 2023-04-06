import module
import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import resnet18

def test_all(test_net, testloader, now_epoch):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = test_net(images.cuda())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.cuda()).sum().item()
    print(f'Accuracy of the epoch {now_epoch} network on the 10000 test images: %d %%' % (100 * correct / total))


def test_every_cate(test_net, testloader, now_epoch):
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():  # 不进行跟踪
        for data in testloader:  # 遍历训练集当中的数据(testloader是训练集)
            images, labels = data  # 获取图像和图像对应的标签
            outputs = test_net(images.cuda())  # 将图片传给神经网络去识别
            _, predicted = torch.max(outputs, 1)  # 返回输入张量所有元素的最大值(即得出神经网络的判断结果)
            c = (predicted == labels.cuda()).squeeze()  # 移除数组中维度为1的维度
            for i in range(4):
                label = labels[i]  # 读取正确的标签
                # 返回可遍历的(键, 值) 元组数组,即判断神经网络是否判断正确,并归类至相应的类别
                class_correct[label] += c[i].item()
                class_total[label] += 1  # 总样本数量计数+1
    for i in range(10):
        print(f'epoch {now_epoch} net Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))


if __name__ == "__main__":
    batch_size = 128
    total_epoch = 135

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=16)

    # 普通网络
    # net = module.Net().cuda()
    # ResNet18 + pretrained
    # net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1).cuda()
    # 网上抄的Resnet18
    net = resnet18.ResNet18().cuda()
    for epoch in range(5, total_epoch + 1, 5):
        path = f'./cifar_net_epoch_{epoch}.pth'
        net.load_state_dict(torch.load(path))

        test_all(test_net=net, testloader=test_loader, now_epoch=epoch)

        test_every_cate(test_net=net, testloader=test_loader, now_epoch=epoch)
