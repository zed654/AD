import torch
import torch.nn.init
import torchvision

from torch.autograd import Variable
import torchvision.utils as utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import os

## torch.device("cuda:0")

# 파일의 디렉토리 경로를 얻기
parrent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

trainset = torchvision.datasets.CIFAR10(root='~/data', train=True, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=10)

testset = torchvision.datasets.CIFAR10(root='~/data', train=False, download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=10)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



##############################
########### Viewer ###########
##############################

import matplotlib.pylab as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

dataiter = iter(trainloader)
images, labels = next(dataiter)

imshow(torchvision.utils.make_grid(images))

print(''.join('%5s' % classes[labels[j]] for j in range(4)))

#####################################
# Define Couvolution Neural Network #
#####################################

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


# Python은 Class에서 self라는 방법을 사용.
#   클래스의 입력이 self로 대체된다고 생각하면 됨.
class Net(nn.Module):                       # 얘는 nn 클래스에 Module 클래스 (이중 클래스)를 상속받은 것.
    def __init__(self):                     # 이 self는 nn.Module을 뜻함.
        super(Net, self).__init__()         # super()는 상속에 대한 오버라이딩을 무시하겠다는 뜻
                                            #   Super()와 Super(Net, self)는 같은 뜻 이다.
                                            #   즉, Net이 상속받은(nn.Module로부터) 인스턴스의 메소드(여기선 __init__)를 사용하겠다는 뜻

        self.conv1 = nn.Conv2d(3, 24, 5)    # 이건 nn.Module.conv1을 뜻 함.
        self.b1 = nn.BatchNorm2d(24)        # 이건 nn.Module.b1을 뜻 함.
        self.pool = nn.MaxPool2d(2, 2)      # 이건 nn.Module.pool을 뜻 함.

        self.conv2 = nn.Conv2d(24, 64, 5)   # 이건 nn.Module.conv2을 뜻 함.
        self.b2 = nn.BatchNorm2d(64)        # 이건 nn.Module.b2을 뜻 함.

        self.fc1 = nn.Linear(64 * 5 * 5, 240)   # 이건 nn.Module.fc1을 뜻 함.
        self.fc2 = nn.Linear(240, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.b1((self.conv1(x)))))
        x = self.pool(F.relu(self.b2(self.conv2(x))))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Set device to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Move the network to the GPU
net = Net().to(device)



#############################
########### optim ###########
#############################

import torch.optim as optim

def backpropagation():
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())    # learning_rate는 디폴트값 0.01을 갖는다.

    for epoch in range(10):
        running_loss = 0.0                          # loss 값
        for i, data in enumerate(trainloader, 0):

            inputs, labels = data

            # Move inputs and labels to the GPU
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()                   # Weight의 gradient 초기화

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()                         # Gradient 계산
            optimizer.step()

            running_loss += loss.item()            # loss값 갱신

            if i % 128 == 127:  # print every 2000 mini-batches
                print('[%d, %5d] loss : %.3f' % (epoch + 1, i + 1, running_loss / 128))
                running_loss = 0.0

        correct = 0
        total = 0
        with torch.no_grad(): # 추론단계로, 미분 비활성화
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)   # Net() 의 forward() 함수 호출. nn.Module에서 forward() 를 호출하게 구성되어있음
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy 1000 test images : %d %%' % (100 * correct / total))

    print('Finished Training')

    torch.save(net.state_dict(), parrent_dir + '/result/CNN.pth')

def inference_only():
    # 저장된 모델 불러오기
    net.load_state_dict(torch.load(parrent_dir + '/result/CNN.pth'))
    net.eval()  # 모델을 평가 모드로 설정 (학습 모드가 아닌)

    # 정확도 계산을 위한 변수 초기화
    correct = 0  # 정답을 맞춘 개수
    total = 0    # 전체 테스트 데이터 개수
    
    # 추론 시에는 gradient 계산이 필요없으므로 비활성화
    with torch.no_grad():
        for data in testloader:
            # 입력 이미지와 정답 레이블 가져오기
            images, labels = data
            # GPU로 데이터 이동
            images, labels = images.to(device), labels.to(device)
            
            # 모델로 예측 수행
            outputs = net(images)
            # 가장 높은 확률을 가진 클래스 선택
            _, predicted = torch.max(outputs.data, 1)
            
            # 정확도 계산
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 전체 테스트 셋에 대한 정확도 출력
    print('전체 테스트 데이터 정확도: %d %%' % (100 * correct / total))

    # 실제 예측 결과 샘플 보여주기
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)
    
    # 입력 이미지 보여주기
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Set the figure title
    plt.figure(num='입력 이미지')
    
    # CPU로 이미지 데이터 이동 후 numpy 배열로 변환
    img = images[0].cpu().numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)))  # CHW -> HWC
    plt.title('입력 이미지')
    plt.show()
    
    # 예측 수행 및 결과 출력
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    
    print('실제 레이블:', classes[labels[0].item()])
    print('예측 레이블:', classes[predicted[0].item()])
    print('처음 4개 이미지에 대한 예측 결과:', ' '.join('%5s' % classes[predicted[j].item()]
                                for j in range(4)))


if __name__ == '__main__':
    #backpropagation()
    inference_only()