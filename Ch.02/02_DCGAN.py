
# %%

import os
from glob import glob
import time
import random
import IPython.display as display
import matplotlib.pyplot as plt # Matplotlib is used to generate plots of data.
import matplotlib.image as mpimg
import PIL
from PIL import Image
import numpy as np # Numpy is an efficient linear algebra library.
import tensorflow as tf
from tensorflow.keras import layers

# 1. 데이터 확인

# 경로 설정
EXPERIMENT_ID = "train_3"
MODEL_SAVE_PATH = os.path.join("C:/Users/the35/Documents/Z. etc/GAN/lecture hands on lab/vanilla_gan_DCGAN/results", EXPERIMENT_ID)
if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)
CHECKPOINT_DIR = os.path.join(MODEL_SAVE_PATH, 'training_checkpoints')

# 데이터 Path
DATA_PATH = "C:/Users/the35/Documents/Z. etc/GAN/lecture hands on lab/datasets/cars/cars_images/"

# Model parameters
BATCH_SIZE = 64
EPOCHS = 6000
LATENT_DEPTH = 100
IMAGE_SHAPE = [100,100]
NB_CHANNELS = 3
LR = 1e-4

seed = random.seed(30)

# 이미지 count
image_count = len(list(glob(str( DATA_PATH + '*.jpg'))))
cars_images_path = list(glob(str(DATA_PATH + '*.jpg')))
# print(cars_images_path)

# 이미지 open (2개만)
for image_path in cars_images_path[:2]:
    display.display(Image.open(str(image_path)))

# %%

# 1-1. Car 브랜드명 추출
images_name = [i.split(DATA_PATH) for i in cars_images_path]
images_name = sum(images_name,[])
cars_model = [i.split('\\')[1].split('_')[0] for i in images_name]

def unique(list1): 
    list_set = set(list1) 
    unique_list = (list(list_set)) 
    return unique_list

unique_cars = unique(cars_model)
unique_cars

# %%

# 1-2. Car 각 브랜드 count
plt.figure(figsize=(20,10))
plt.hist(cars_model, color = "blue", lw=0, alpha=0.7)
plt.ylabel('images number')
plt.xlabel('car model')
plt.show()

# %%

# 1-3. 이미지 최대 최소 크기 확인
image_size = []
for filename in cars_images_path:
    im=Image.open(filename)
    im =im.size
    image_size.append(im)
print(max(image_size))
print(min(image_size))


# %%

# 1-4. 특정 이미지 오픈
image = mpimg.imread(cars_images_path[20])

plt.axis("off")
plt.imshow(image)
# %%

# 1-5. 채널별로 이미지 분류
# Isolate RGB channels
r = image[:,:,0]
g = image[:,:,1]
b = image[:,:,2]

# Visualize the individual color channels
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))
ax1.set_title('R channel') # red
ax1.imshow(r, cmap='gray')
ax2.set_title('G channel') # green
ax2.imshow(g, cmap='gray')
ax3.set_title('B channel') # blue
ax3.imshow(b, cmap='gray')
# %%

# 2. 전처리

# 이미지 정규화
# [-1,1] 사용하는 경우 활성화 함수는 tanh
# [0,1] 사용하는 경우 활성화 함수는 sigmoid
# 이번 코드는 [0,1] 선택했으므로, 마지막 레이어에서 시그모이드 사용

# 2-1. Data loader
# tensorflow 사용
@tf.function # tf1.x 스타일로 해당 함수내의 로직이 동작 (실행속도 약간 상승)
def preprocessing_data(path): 
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMAGE_SHAPE[0],IMAGE_SHAPE[1]])
    image = image / 255.0
    return image

def dataloader(paths): 
    dataset = tf.data.Dataset.from_tensor_slices(paths)

    # 훈련세트가 독립적이고 동일하게 분산되도록 셔플을 호출
    # 버퍼 크기를 지정해야하며 충분히 크게 만드는 것이 중요
    # 그렇지 않으면 셔플링이 매우 효율적이지 않다
    dataset = dataset.shuffle(10* BATCH_SIZE)
    
    dataset = dataset.map(preprocessing_data)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(1)
    return dataset   

dataset = dataloader(cars_images_path)

for batch in dataset.take(1):
    for img in batch:
        img_np = img.numpy()
        plt.figure()
        plt.axis('off')
        plt.imshow((img_np-img_np.min())/(img_np.max()-img_np.min()))

# %%

# PIL 사용으로 data loader
image_list = []
for filename in cars_images_path:
    im=Image.open(filename)
    im =im.resize((IMAGE_SHAPE[0],IMAGE_SHAPE[1]))
    image_list.append(im)

images_processed = []
for image in image_list:
    gan_train_images = np.asarray(image)
    gan_train_images = gan_train_images.astype('float32') /255.0
    images_processed.append(gan_train_images)

dataset = tf.data.Dataset.from_tensor_slices(images_processed).shuffle(10* BATCH_SIZE).batch(BATCH_SIZE)
for batch in dataset.take(1):
    for img in batch:
        img_np = img.numpy()
        plt.figure()
        plt.axis('off')
        plt.imshow((img_np-img_np.min())/(img_np.max()-img_np.min()))

# %%

# 3. Modeling

# 3-1. Generator
# Dense (Fully Connected)
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(25*25*128, use_bias=False, input_shape=(100,))) # 25*25*128 = 80,000
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Reshape((25, 25, 128)))
    assert model.output_shape == (None, 25, 25, 128) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 25, 25, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 50, 50, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    
    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='sigmoid'))
    assert model.output_shape == (None, 100, 100, 3)
    model.summary()
    return model

generator = make_generator_model()
noise = tf.random.normal([1, LATENT_DEPTH])
generated_image = generator(noise, training=True)

plt.imshow(generated_image[0, :, :, :], cmap='gray')
# %%

# ** 이해 예제 코드 **

# 미리 만들어진 계층을 가져와서 설정하고 쌓아가는 방식으로 아주 쉽게 구축이 가능
# 부에 순전파 역전파에 대한 것도 객체지향적으로 잘 구현되어 있고, 시퀀셜 객체 자체의 학습, 
# 검증 메소드 등을 통해 신경망을 말 그대로 객체로 다룰수 있습니다.

model = tf.keras.Sequential()
# 64개의 유닛을 가진 완전 연결 층을 모델에 추가합니다:
model.add(layers.Dense(64, activation='relu')) # Dense(=FC)
# 또 하나를 추가합니다:
model.add(layers.Dense(64, activation='relu'))
# 10개의 출력 유닛을 가진 소프트맥스 층을 추가합니다:
model.add(layers.Dense(10, activation='softmax'))

# Sequential 안에 계층을 생성하여 넣어주면, 
# Sequential이 각 계층의 객체 데이터를 해석하여 학습, 검증, 예측의 기능을 실행시키는 것입니다.


# - 레이어 예시

# 시그모이드 활성화 층을 만듭니다:
layers.Dense(64, activation='sigmoid')
# 또는 다음도 가능합니다:
layers.Dense(64, activation=tf.keras.activations.sigmoid)

# 커널 행렬에 L1 규제가 적용된 선형 활성화 층. 하이퍼파라미터 0.01은 규제의 양을 조절합니다:
layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l1(0.01))

# 절편 벡터에 L2 규제가 적용된 선형 활성화 층. 하이퍼파라미터 0.01은 규제의 양을 조절합니다:
layers.Dense(64, bias_regularizer=tf.keras.regularizers.l2(0.01))

# 커널을 랜덤한 직교 행렬로 초기화한 선형 활성화 층:
layers.Dense(64, kernel_initializer='orthogonal')

# 절편 벡터를 상수 2.0으로 설정한 선형 활성화 층:
layers.Dense(64, bias_initializer=tf.keras.initializers.Constant(2.0))

# %%

# assert - 가정 설정문
# 뒤의 조건이 True가 아니면 AssertError를 발생한다.

lists = [1, 3, 6, 3, 8, 7, 13, 23, 13, 2, 3.14, 2, 3, 7]

def test(t):
    assert type(t) is int, '정수 아닌 값이 있네'

for i in lists:
    test(i)


# %%
# 3-2. Discriminator
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[100, 100, 3]))
    model.add(layers.ReLU(0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.ReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    model.summary()
    return model

discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print (decision)

# %%
generator_optimizer = tf.keras.optimizers.Adam(lr=LR)
discriminator_optimizer = tf.keras.optimizers.Adam(lr=LR) 

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# %%

# 4. Loss and Optimization

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
# %%

# Experimnent
def summary(name_data_dict,
            step=None,
            types=['mean', 'std', 'max', 'min', 'sparsity', 'histogram'],
            historgram_buckets=None,
            name='summary'):
    """Summary.
    Examples
    --------
    >>> summary({'a': data_a, 'b': data_b})
    """
    def _summary(name, data):
        if data.shape == ():
            tf.summary.scalar(name, data, step=step)
        else:
            if 'mean' in types:
                tf.summary.scalar(name + '-mean', tf.math.reduce_mean(data), step=step)
            if 'std' in types:
                tf.summary.scalar(name + '-std', tf.math.reduce_std(data), step=step)
            if 'max' in types:
                tf.summary.scalar(name + '-max', tf.math.reduce_max(data), step=step)
            if 'min' in types:
                tf.summary.scalar(name + '-min', tf.math.reduce_min(data), step=step)
            if 'sparsity' in types:
                tf.summary.scalar(name + '-sparsity', tf.math.zero_fraction(data), step=step)
            if 'histogram' in types:
                tf.summary.histogram(name, data, step=step, buckets=historgram_buckets)

    with tf.name_scope(name):
        for name, data in name_data_dict.items():
            _summary(name, data)

train_summary_writer = tf.summary.create_file_writer(os.path.join(MODEL_SAVE_PATH, 'summaries', 'train'))
checkpoint_prefix = os.path.join(CHECKPOINT_DIR, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# %%

# save_image
def generate_and_save_images(model, epoch):
    
    plt.figure(figsize=(15,10))

    for i in range(4):
        noise = tf.random.normal([1,100])
        images = model(noise, training=False)
        
        image = images[0, :, :, :]
        image = np.reshape(image, [100, 100, 3])

        plt.subplot(1, 4, i+1)
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.title("Randomly Generated Images")

    plt.tight_layout()  
    plt.savefig(os.path.join(MODEL_SAVE_PATH,'image_at_epoch_{:02d}.png'.format(epoch)))
    plt.show()
# %%

# model_training
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, LATENT_DEPTH])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return {'Generator loss': gen_loss,
            'Discriminator loss': disc_loss}

# %%
def train(dataset, epochs):
    with train_summary_writer.as_default():
        with tf.summary.record_if(True):
            for epoch in range(epochs):
                start = time.time()

                for image_batch in dataset:
                    loss_dict = train_step(image_batch)
                summary(loss_dict, step=generator_optimizer.iterations, name='losses')
                # Save the model every 15 epochs
                if (epoch + 1) % 15 == 0:
                    checkpoint.save(file_prefix = checkpoint_prefix)
                    display.clear_output(wait=True)
                    generate_and_save_images(generator,
                                             epoch + 1)
                print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))


    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator, epochs)
# %%

# 실행
train(dataset, EPOCHS)