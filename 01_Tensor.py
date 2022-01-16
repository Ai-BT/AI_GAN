import torch

# 1. 변수 저장
x = torch.tensor(3.5)
print(x)

y = x + 3
print(y)


# %%
import torch

# 2. 자동 기울기 계산
x = torch.tensor(3.5, requires_grad=True)
print(x)

y = (x-1) * (x-2) * (x-3)
print(y)

# y 미분으로 기울기 계산하기
# 3x^2 = 12x + 11
# backward는 기울기에 대해 숫자로 계산하여 텐서 x에 저장
# x = 3.5 이므로, 3 * (3.5*3.5) - 12 * (3.5) + 11 = 5.75
y.backward()

# x 가 3.5 일때 기울기 값
# what is gradient at x = 3.5 ?
x.grad # 5.75



# %%

# 정방향
x = torch.tensor(3.5, requires_grad=True)
y = x * x 
z = 2 * y +3

# z 기울기 계산 
# dz/dx = dz/dy * dy/dx
# dz/dx = 2 * 2x
# dz/dx = 4x
# = 14

# x = 3.5 일때 z 에 대한 기울기를 숫자로 계산하여 텐서 x에 저장
z.backward()

# x = 3.5 일때 z 의 기울기 계산
x.grad


# %%

# 뉴런이 5개 연결된 경우

a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

x = 2*a + 3*b
y = 5*a*a + 3*b*b*b
z = 2*x + 3*y

# a 가 2 일때, z 의 변화율
# dz/da = dz/dy*dy/da + dz/dx*dx/da

# dz/dy = 3
# dz/dx = 2

# dy/da = 10a
# dx/da = 2

# 30a + 4
# a 가 2 일때, z 는 64

z.backward()
a.grad


# p.40
# %%
