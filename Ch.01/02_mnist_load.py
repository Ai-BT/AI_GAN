from re import I
import pandas
import matplotlib.pyplot as plt

# mnsit 파일은 아래에서 다운로드
# 훈련용 - pjreddie.com/media/files/mnist_train.csv
# 테스트용 - pjreddie.com/media/files/mnist_test.csv

df = pandas.read_csv("./02_myo_gan/mnist_train.csv", header=None)

head = df.head() # 데이터셋의 처음 다섯 행 출력, 각 행은 785개 이고 첫번째는 정답
info = df.info() # 데이터 프레임 안에 정보

# 1. 데이터 시각화
row = 6 # 각 데이터 위치
data = df.iloc[row] # row 번째 행을 data 변수에 할당

# 첫 번째 값은 레이블
label = data[0] # 정답이 0 번째 있으니, label에 data의 정답 할당

# 이미지 데이터는 나머지 784개의 값
img = data[1:].values.reshape(28,28)
plt.title("label = " + str(label))
plt.imshow(img, interpolation='none', cmap='Blues')
plt.show()

