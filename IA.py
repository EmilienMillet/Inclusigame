# Librairies

import numpy as np
import pandas as pd
import os
import tensorflow as tf
import matplotlib.pyplot as plt

# Adresse

list_adress_test = []
list_adress_train = []
list_adress_valid = []
for dirname, _, filenames in os.walk('/kaggle/input/cards-image-datasetclassification/test'):
    for filename in filenames:
        list_adress_test.append(os.path.join(dirname, filename))
for dirname, _, filenames in os.walk('/kaggle/input/cards-image-datasetclassification/train'):
    for filename in filenames:
        list_adress_train.append(os.path.join(dirname, filename))
for dirname, _, filenames in os.walk('/kaggle/input/cards-image-datasetclassification/valid'):
    for filename in filenames:
        list_adress_valid.append(os.path.join(dirname, filename))
print(list_adress_valid[0])

# Data test

test = tf.keras.utils.image_dataset_from_directory('/kaggle/input/cards-image-datasetclassification/test')
valid = tf.keras.utils.image_dataset_from_directory('/kaggle/input/cards-image-datasetclassification/valid')
data = tf.keras.utils.image_dataset_from_directory('/kaggle/input/cards-image-datasetclassification/train')
data_iterator_test = test.as_numpy_iterator() #Voir ce que c'est
data_iterator_valid = valid.as_numpy_iterator()
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()
batch_valid = data_iterator_valid.next()
batch_test = data_iterator_test.next()

# Visu

fig, ax = plt.subplots(ncols = 8, figsize = (20,20))
for idx, img in enumerate(batch[0][:8]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])
    
    
import matplotlib.image as mpimg
target_dir = "/kaggle/input/cards-image-datasetclassification/train"
random_image = "ace of clubs/002.jpg"
random_image_path = target_dir + "/" + random_image

print(random_image_path)
# Read in the random image
img = mpimg.imread(random_image_path)
plt.imshow(img)




data_augmentation = tf.keras.Sequential(
  [
    tf.keras.layers.RandomFlip("horizontal",
                      input_shape=(256,
                                  256,
                                  3)),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
  ]
)




plt.figure(figsize=(10, 10))
for images, _ in data.take(1):
  for i in range(9):
    augmented_images = data_augmentation(images)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_images[0].numpy().astype("uint8"))
    plt.axis("off")
    
    
data=data.map(lambda x,y: (x/255,y))
test=test.map(lambda x,y: (x/255,y))
valid=valid.map(lambda x,y: (x/255,y))

#Deep learning model 

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()

model.add(data_augmentation)
model.add(Conv2D(16, (3,3), 1,activation= 'relu',input_shape=(256,256,3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3,3), 1,activation= 'relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3,3), 1,activation= 'relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(53, activation='sigmoid'))

model.compile('adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])

model.summary()



logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logdir)

hist = model.fit(data, epochs=20, validation_data = valid, callbacks = [tensorboard_callback])



acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']

loss = hist.history['loss']
val_loss = hist.history['val_loss']

epochs_range = range(20)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

    
    
