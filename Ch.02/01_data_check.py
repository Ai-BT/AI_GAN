
# %%

import os
from glob import glob
# from glob import glob
import time
import random

import IPython.display as display
import matplotlib.pyplot as plt # Matplotlib is used to generate plots of data.
import matplotlib.image as mpimg
import PIL
from PIL import Image
# import imageio
import numpy as np # Numpy is an efficient linear algebra library.

import tensorflow as tf
from tensorflow.keras import layers


# from google.colab import drive
# drive.mount('/content/drive')

# EXPERIMENT_ID = "train_3"
# MODEL_SAVE_PATH = os.path.join("/content/drive/My Drive/lecture hands on lab/vanilla_gan_DCGAN/results", EXPERIMENT_ID)
# if not os.path.exists(MODEL_SAVE_PATH):
#     os.makedirs(MODEL_SAVE_PATH)
# CHECKPOINT_DIR = os.path.join(MODEL_SAVE_PATH, 'training_checkpoints')

# Data path
# GAN\lecture hands on lab\datasets\cars\cars_images
# C:\Users\the35\Documents\Z. etc\GAN\lecture hands on lab\datasets\cars\cars_images
DATA_PATH = "C:/Users/the35/Documents/Z. etc/GAN/lecture hands on lab/datasets/cars/cars_images/"

# Model parameters
BATCH_SIZE = 64
EPOCHS = 6000
LATENT_DEPTH = 100
IMAGE_SHAPE = [100,100]
NB_CHANNELS = 3
LR = 1e-4



seed = random.seed(30)



image_count = len(os.listdir(DATA_PATH))

# cars_images_path = os.listdir(DATA_PATH)

image_count = len(list(glob(str( DATA_PATH + '*.jpg'))))

cars_images_path = list(glob(str(DATA_PATH + '*.jpg')))

print(cars_images_path)

for image_path in cars_images_path[:2]:
    display.display(Image.open(str(image_path)))

# for image_path in cars_images_path[:2]:
#     display.display(Image.open(DATA_PATH + str(image_path)))
# %%

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

plt.figure(figsize=(20,10))
plt.hist(cars_model, color = "blue", lw=0, alpha=0.7)
plt.ylabel('images number')
plt.xlabel('car model')
plt.show()

# %%

image_size = []
for filename in cars_images_path:
    im=Image.open(filename)
    im =im.size
    image_size.append(im)
print(max(image_size))
print(min(image_size))


# %%
image = mpimg.imread(cars_images_path[20])

plt.axis("off")
plt.imshow(image)
# %%

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

@tf.function
def preprocessing_data(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMAGE_SHAPE[0],IMAGE_SHAPE[1]])
    image = image / 255.0
    return image

def dataloader(paths):
    dataset = tf.data.Dataset.from_tensor_slices(paths)
    dataset = dataset.shuffle(10* BATCH_SIZE)
    dataset = dataset.map(preprocessing_data)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(1)
    # print(dataset)
    return dataset   

dataset = dataloader(cars_images_path)


for batch in dataset.take(1):
    for img in batch:
        img_np = img.numpy()
        plt.figure()
        plt.axis('off')
        plt.imshow((img_np-img_np.min())/(img_np.max()-img_np.min()))

# %%
