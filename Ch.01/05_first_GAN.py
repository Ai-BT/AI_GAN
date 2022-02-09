# %%

import torch
import torch.nn as nn

import pandas
import matplotlib.pyplot as plt
import random
import numpy

# 1010 패턴 GAN

# 실제 데이터 소스
def generate_real():
    real_data = torch.FloatTensor(
        [random.uniform(0.8, 1.0), # 0.8 ~ 1.0 사이의 임의의 값
        random.uniform(0.0, 0.2),
        random.uniform(0.8, 1.0),
        random.uniform(0.0, 0.2)]
        )
    return real_data

# gr = generate_real()
# print(gr)


# 판별기 테스트
def generate_random(size):
    random_data = torch.rand(size)
    return random_data


# 판별기 
class Discriminator(nn.Module):
    def __init__(self):
        # 파이토치 부모 클래스 초기화
        super().__init__()

        # 신경망 레이어 정의
        self.model = nn.Sequential(
            nn.Linear(4,3),
            nn.Sigmoid(),
            nn.Linear(3,1),
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
            print("counter = ", self.counter)
            pass

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        pass

    def plot_progress(self):
        df = pandas.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=(0, 1.0), figsize=(16,8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5))
        pass



D = Discriminator()

# 판별기 테스트
for i in range(10000):
    # 실제 데이터
    D.train(generate_real(), torch.FloatTensor([1.0]))
    # 생성된 데이터
    D.train(generate_random(4), torch.FloatTensor([0.0]))
    pass

D.plot_progress()
# %%

print( D.forward( generate_real() ).item() )
print( D.forward( generate_random(4) ).item() )

# %%

# 생성기
class Generator(nn.Module):
    def __init__(self):
        # 파이토치 부모 클래스 초기화
        super().__init__()

        # 신경망 레이어 정의
        self.model = nn.Sequential(
            nn.Linear(1,3),
            nn.Sigmoid(),
            nn.Linear(3,4),
            nn.Sigmoid()
        )

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
            print("counter = ", self.counter)
            pass

        # 기울기를 초기화하고 역전파 후 가중치 갤신
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

    def plot_progress(self):
        df = pandas.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=(0, 1.0), figsize=(16,8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5))
        pass

    pass

# 생성기 결과 확인
G = Generator()
G.forward(torch.FloatTensor([0.5]))

# %%

# GAN 훈련하기

# 판별기 및 생성기 생성
D = Discriminator()
G = Generator()

image_list = []

# 판별기와 생성기 훈련
for i in range(10000):

    # 1단계 : True에 대한 판별기 훈련
    D.train(generate_real(), torch.FloatTensor([1.0]))

    # 2단계: 거짓에 대하 판별기 훈련
    # G의 기울기가 계산되지 않도록 detach() 함수를 이용
    D.train(G.forward(torch.FloatTensor([0.5])).detach(), torch.FloatTensor([0.0]))

    # 3단계: 생성기 훈련
    G.train(D, torch.FloatTensor([0.5]), torch.FloatTensor([1.0]))

    # add image to list every 1000
    if (i % 1000 == 0):
      image_list.append( G.forward(torch.FloatTensor([0.5])).detach().numpy() )


    pass


# %%
D.plot_progress()

# %%
G.plot_progress()

# %%

plt.figure(figsize = (16,8))
plt.imshow(numpy.array(image_list).T, interpolation='none', cmap='Blues')

# %%
