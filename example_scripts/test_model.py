import keras
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum, auto
from typing import List
from keras.utils.vis_utils import plot_model
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import scale
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras import regularizers

from keras.datasets import cifar10

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
(x_train, y_train), (x_test, y_test) = (x_train[0:500], y_train[0:500]), (x_test[0:50], y_test[0:50])
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

num_classes = 10

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
validation_split = 0.15
#training
batch_size = 256
training_epochs = 200

#data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    )
datagen.fit(x_train)

print("Using data augmentation.")

compiler = {"loss":"categorical_crossentropy", "optimizer":keras.optimizers.RMSprop(), "metrics":["accuracy"]}

es = EarlyStopping(monitor='val_acc', mode='min', verbose=1, patience=10)
mc = ModelCheckpoint('best_model_checkpoint.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
csv_logger = CSVLogger('training.log')

model = keras.models.load_model("./runs/CIFAR-10_5_17gens/models/best_generation_16.h5")

custom_fit_args = {"generator": datagen.flow(x_train, y_train, batch_size=batch_size),
"steps_per_epoch": x_train.shape[0] // batch_size,
"epochs": training_epochs,
"verbose": 1,
"validation_data": (x_test,y_test),
"callbacks": [es, csv_logger]#, mc]
}
history = model.fit_generator(**custom_fit_args)

#history = model.fit(x_train, y_train, epochs=training_epochs, validation_split=validation_split, batch_size=batch_size, callbacks=[csv_logger])#, callbacks=[es, csv_logger])

N = training_epochs

# summarize history for accuracy
plt.style.use("ggplot")
plt.plot(history.history['acc'], label="train_acc")
plt.plot(history.history['val_acc'], label="val_acc")
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.tight_layout()
plt.savefig(f'./history_acc', show_shapes=True, show_layer_names=True)
plt.clf()

# summarize history for loss
plt.style.use("ggplot")
plt.plot(history.history['loss'], label="train_loss")
plt.plot(history.history['val_loss'], label="val_loss")
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.tight_layout()
plt.savefig(f'./history_loss', show_shapes=True, show_layer_names=True)
plt.clf()

scores = model.evaluate(x_test, y_test, verbose=1)
print(scores)