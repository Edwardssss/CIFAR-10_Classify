import torch
import torchvision
import torchvision.transforms as transforms
import module
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import resnet18


def my_train(my_net, train_loader, train_epoch, save_step: int, train_optimizer):
    for epoch in range(1, train_epoch + 1):  # loop over the dataset multiple times
        print(f"epoch {epoch} begins!")
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            # zero the parameter gradients
            # 当使用StepLR时没有这个
            train_optimizer.zero_grad()
            # forward + backward + optimize
            outputs = my_net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            train_optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        if epoch % save_step == 0:
            path = f'./cifar_net_epoch_{epoch}.pth'
            torch.save(net.state_dict(), path)
            print(f"Epoch {epoch} pth has been saved successfully!")
    print('Finished Training')


if __name__ == "__main__":
    batch_size = 128
    num_workers = 16
    total_epoch = 135
    save_step = 5

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 普通网络
    # net = module.Net().cuda()
    # Resnet18
    # net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1).cuda()
    # 网上抄的Resnet18
    net = resnet18.ResNet18().cuda()

    criterion = nn.CrossEntropyLoss()  # 设置交叉熵
    # SGD
    # optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
    # Adam
    optimizer = optim.Adam(net.parameters(), lr=1e-2, weight_decay=5e-4)
    my_train(my_net=net, train_loader=train_loader, train_epoch=total_epoch, save_step=save_step,
             train_optimizer=optimizer)
