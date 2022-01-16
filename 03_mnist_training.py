from time import time
from matplotlib.pyplot import yticks
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pandas

# 어떠한 신경망을 만들든 항상 파이토치의 torch.nn을 상속받아 클래스를 만들어야 한다.

# 1. 클래스 생성
# nn.Module 상속 받음
class Classifire(nn.Module):

    def __init__(self):
        # 부모 클래스 초기화
        super().__init__()

        # 신경망 레이어 정의
        self.model = nn.Sequential(
            nn.Linear(784, 200), # 784개 노드로부터 200개 노드까지의 완전 연결
            nn.Sigmoid(), # 활성화 함수로 이전 레이어부터의 출력 여기는 200개 노드 적용
            nn.Linear(200,10), # 200개 노드를 다시 10 노드로 연결
            nn.Sigmoid() # 10개 노드의 출력이 적용 되며 최종 출력
        )

        # 손실 함수 생성
        self.loss_function = nn.MSELoss()

        # 단순 SGD 옴티마이저 설정
        self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)

         # 변수 초기화
        self.counter = 0
        self.progress = []

        pass

    def forward(self, inputs):
        # 모델 실행
        return self.model(inputs)

    def train(self, inputs, targets):
        # 신경망 출력 계산
        outputs = self.forward(inputs)

        # 손실 계산
        loss = self.loss_function(outputs, targets)


        # 카운터를 증가시키고 10회마다 오차 저장
        self.counter += 1
        if (self.counter % 10 == 0):
            self.progress.append(loss.item())
            pass

        if (self.counter % 10000 == 0):
            print("counter = ", self.counter)
            pass

 
    def plot_progress(self):
        df = pandas.DataFrame(self.progress, columns=['loss']) # 차트를 쉽게 나타내기 위해 손실값을 저장해둔 리스트를 데이터프레임으로 변환
        df.plot(ylim=(0, 1.0), figsize=(16,8), alpha=0.1, marker='.', grid=True, yticks=(0.25 , 0.5)) # 스타일과 디자인 지정
        pass

    pass




class MnistDataset(Dataset):

    def __init__(self, csv_file):
        self.data_df = pandas.read_csv(csv_file, header=None)
        pass

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        # 이미지 목표(레이블)
        label = self.data_df.iloc[index, 0]
        target = torch.zeros((10))
        target[label] = 1.0

        # 0 - 255 의 이미지를 0-1로 정규호
        image_values = torch.FloatTensor(self.data_df.iloc[index, 1:].values)

        # 레이블, 이미지 데이터 텐서, 목표 텐서 반호나
        return label, image_values, target

    def plot_image(self, index):
        img = self.data_df.iloc[index, 1:].values.reshape(28, 28)
        plt.title("label = " + str(self.data_df.iloc[index,0]))
        plt.imshow(img, interpolation='none', cmap='Blues')
        plt.show()

    pass

# 데이터셋 
mnist_dataset = MnistDataset('./02_myo_gan/mnist_train.csv')

# 9번째 데이터 불러오기
# mnist_dataset.plot_image(9)

print('========================================================')

# 신경망 생성

C = Classifire()

# Mnist 데이터에 대해 훈련 진행
epochs = 3

for i in range(epochs):
    print('training epoch', i+1, "of", epochs)
    for label, image_data_tensor, target_tensor in mnist_dataset:
        C.train(image_data_tensor, target_tensor)
        pass
    pass