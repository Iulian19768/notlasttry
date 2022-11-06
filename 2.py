import os
from sys import modules
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import cv2 as cv
import matplotlib.pyplot as plt
image_size = (155, 135)
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "math2",
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "math2",
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

class_names = train_ds.class_names
AUTOTUNE = tf.data.AUTOTUNE
model = tf.keras.models.load_model('handwritten.model')
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
num_classes = 10
img= cv.imread('resized6.png')
img= cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.imshow(img, cmap=plt.cm.binary)

prediction= model.predict(np.array([img])/255)
index= np.argmax(prediction)
print(f'Prediction is {class_names[index]}')
