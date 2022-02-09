import numpy as np
import torch
import time

# 1. CPU 텐서로 생성
x = torch.tensor(3.5)
y = torch.FloatTensor([3.5])
# print(y.type())

# 2. GPU 텐서로 생성
# CUDA 사용 확인 코드
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    print('using cuda: ', torch.cuda.get_device_name(0))
    pass

# 기본값을 CUDA 로 설정, 가징하지 않으면 CPU 사용
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

a = torch.cuda.FloatTensor([3.5])
b = a * a
print(b)


