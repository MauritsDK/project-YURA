from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import tensorflow as tf

# dimensions of our images.
img_width, img_height = 32, 32

AUTOTUNE=tf.data.experimental.AUTOTUNE


train_data_dir = 'split_Big_small/train'
validation_data_dir = 'split_Big_small/val'
epochs = 20
batch_size = 128

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
print(K.image_data_format())
model = Sequential()
##1
print("1")
model.add(Conv2D(32, (3, 3), strides=1, padding="same", input_shape=input_shape))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=3, strides=2, padding="same"))
##2
print("2")
model.add(Conv2D(64, (3, 3), strides=1, padding="same"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=3, strides=2, padding="same"))
##3
print("3")
model.add(Conv2D(128, (3, 3), strides=1, padding="same"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=3, strides=2, padding="same"))
##4
model.add(Conv2D(256, (3, 3), strides=1, padding="same"))
model.add(Conv2D(256, (3, 3), strides=1, padding="same"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=3, strides=2, padding="same"))
##5
model.add(Conv2D(512, (3, 3), strides=1, padding="same"))
model.add(Conv2D(512, (3, 3), strides=1, padding="same"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=3, strides=2, padding="same"))
##6
model.add(AveragePooling2D(pool_size=1, strides=1, padding="valid"))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(3, activation="linear"))
print(model.summary())

model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'],
              optimizer='Adam')

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(32,32),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(32,32),
    batch_size=batch_size,
    class_mode='categorical')

model.fit_generator(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=128 // batch_size,
    steps_per_epoch=25)
#steps_per_epoch=nb_train_samples //batch_size,

model.save_weights('model_weights.h5')
