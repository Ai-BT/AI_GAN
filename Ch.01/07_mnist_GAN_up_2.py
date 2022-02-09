# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas
import numpy
import random
import matplotlib.pyplot as plt

# 1. Mnist 데이터 로드
class MnistDataset(Dataset):
    
    def __init__(self, csv_file):
        self.data_df = pandas.read_csv(csv_file, header=None)
        pass
    
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, index):
        # 이미지 목표(레이블)
        label = self.data_df.iloc[index,0]
        target = torch.zeros((10))
        target[label] = 1.0
        
        # 0-255의 이미지를 0-1로 정규화
        image_values = torch.FloatTensor(self.data_df.iloc[index,1:].values) / 255.0
        
        # 레이블, 이미지 데이터 텐서, 목표 텐서 반환
        return label, image_values, target
    
    def plot_image(self, index):
        img = self.data_df.iloc[index,1:].values.reshape(28,28)
        plt.title("label = " + str(self.data_df.iloc[index,0]))
        plt.imshow(img, interpolation='none', cmap='Blues')
        pass
    
    pass

# data 확인
mnist_dataset = MnistDataset('02_myo_gan/mnist_train.csv')
# mnist_dataset.plot_image(17)

# %%
# 2. 판별기 만들기
class Discriminator(nn.Module):
    def __init__(self):
        # 파이토치 부모 클래스 초기화
        super().__init__()

        # 신경망 레이어 정의
        self.model = nn.Sequential(
            nn.Linear(784,200),
            nn.LeakyReLU(0.02),
            nn.LayerNorm(200),
            nn.Linear(200,1),
            nn.Sigmoid()
        )

        # 손실함수 설정
        self.loss_function = nn.MSELoss()

        # SGD 옵티마이저 설정
        self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)

        # 진행 측정을 위한 변수 초기화
        self.counter = 0
        self.progress = []

        pass

    # 모델
    def forward(self, inputs):
        # 모델 실행
        return self.model(inputs)

    # 훈련
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
            print("counter D = ", self.counter)
            pass

        # 기울기를 초기화하고 역전파 후 가중치 갱신
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        pass

    def plot_progress(self):
        df = pandas.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=(0, 1.0), figsize=(16,8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5))
        pass


# 판별기 테스트
def generate_random(size):
    random_data = torch.rand(size)
    return random_data

def generate_random_seed(size):
    random_data = torch.randn(size)
    return random_data

# 판별기 생성
D = Discriminator()

# 판별기 테스트
for label, image_data_tensor, target_tensor in mnist_dataset:
    # 실제 데이터
    D.train(image_data_tensor, torch.FloatTensor([1.0]))
    # 생성된 데이터
    D.train(generate_random(784), torch.FloatTensor([0.0]))
    pass

D.plot_progress()


# 판별기 학습??
for i in range(4):
    image_data_tensor = mnist_dataset[random.randint(0,60000)][1]
    print(D.forward(image_data_tensor).item())
    pass

for i in range(4):
    print(D.forward(generate_random(784)).item())
    pass


# %%
# 3. 생성기 만들기
class Generator(nn.Module):
    def __init__(self):
        # 파이토치 부모 클래스 초기화
        super().__init__()

        # 신경망 레이어 정의
        self.model = nn.Sequential(
            nn.Linear(100,200),
            nn.LeakyReLU(0.02),
            nn.LayerNorm(200),
            nn.Linear(200,784),
            nn.Sigmoid()
        )

        # SGD 옵티마이저 설정
        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.0001)

        # 진행 측정을 위한 변수 초기화
        self.counter = 0
        self.progress = []

        pass

    # 모델
    def forward(self, inputs):
        # 모델 실행
        return self.model(inputs)

    # 훈련
    def train(self, D, inputs, targets):
        # 신경망 출력 계산
        g_outputs = self.forward(inputs)

        # 판별기로 전달
        d_output = D.forward(g_outputs)

        # 오차 계싼
        loss = D.loss_function(d_output, targets)

        # 카운터를 증가시키고 10회마다 오차 저장
        self.counter += 1
        if (self.counter % 10 == 0):
            self.progress.append(loss.item())
            pass

        if (self.counter % 10000 == 0):
            print("counter G = ", self.counter)
            pass

        # 기울기를 초기화하고 역전파 후 가중치 갤신
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

    def plot_progress(self):
        df = pandas.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=(0), figsize=(16,8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5, 1.0, 5.0))
        pass

    pass

# 생성기 생성
G = Generator()

output = G.forward(generate_random_seed(100))
img = output.detach().numpy().reshape(28,28)
plt.imshow(img, interpolation='none', cmap='Blues')

# %%
# 4. 판별기 및 생성기 생성
D = Discriminator()
G = Generator()

# 판별기와 생성기 훈련
i = 0
epochs = 4
for epoch in range(epochs):
    print('epoch = ', i =+ 1)
    for label, image_data_tensor, target_tensor in mnist_dataset:

        # 참일 경우 판별기 훈련
        D.train(image_data_tensor, torch.FloatTensor([1.0]))
        
        # 거짓일 경우 판별기 훈련
        # G의 기울기가 계산되지 않도록 detach() 함수를 이용
        D.train(G.forward(generate_random_seed(100)).detach(), torch.FloatTensor([0.0]))
        
        # 생성기 훈련
        G.train(D, generate_random_seed(100), torch.FloatTensor([1.0]))

        pass
    pass


# %%
# 5. 판별기 오차 플롯

D.plot_progress()

# %%

# 6. 생성기 오차 플롯

G.plot_progress()

# %%

# 7. 훈련된 생성기로부터 몇개의 출력을 플롯

# plot a 3 column, 2 row array of generated images
f, axarr = plt.subplots(2,3, figsize=(16,8))
for i in range(2):
    for j in range(3):
        output = G.forward(generate_random_seed(100))
        img = output.detach().numpy().reshape(28,28)
        axarr[i,j].imshow(img, interpolation='none', cmap='Blues')
        pass
    pass

# %%
