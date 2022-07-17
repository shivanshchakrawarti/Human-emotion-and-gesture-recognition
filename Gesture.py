import keras
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Dropout, BatchNormalization, Dense, Flatten
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import os

num_classes = 4

Img_Height = 200
Img_width = 200

batch_size = 128

train_dir = "train1"
validation_dir = "validation1"

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=30,
                                   shear_range=0.3,
                                   zoom_range=0.3,
                                   width_shift_range=0.4,
                                   height_shift_range=0.4,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(Img_Height, Img_width),
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=True)

validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                             target_size=(Img_Height, Img_width),
                                                             batch_size=batch_size,
                                                             class_mode='categorical',
                                                             shuffle=True)

VGG16_MODEL = VGG16(input_shape=(Img_width, Img_Height, 3), include_top=False, weights='imagenet')

for layers in VGG16_MODEL.layers: 
    layers.trainable=False

for layers in VGG16_MODEL.layers:
    print(layers.trainable)


# Input layer
input_layer = VGG16_MODEL.output

# Convolutional Layer
Conv1 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='valid',
               data_format='channels_last', activation='relu', 
               kernel_initializer=keras.initializers.he_normal(seed=0), 
               name='Conv1')(input_layer)

# MaxPool Layer
Pool1 = MaxPool2D(pool_size=(2,2),strides=(2,2),padding='valid', 
                  data_format='channels_last',name='Pool1')(Conv1)

# Flatten
flatten = Flatten(data_format='channels_last',name='Flatten')(Pool1)

# Fully Connected layer-1
FC1 = Dense(units=30, activation='relu', 
            kernel_initializer=keras.initializers.glorot_normal(seed=32), 
            name='FC1')(flatten)

# Fully Connected layer-2
FC2 = Dense(units=30, activation='relu', 
            kernel_initializer=keras.initializers.glorot_normal(seed=33),
            name='FC2')(FC1)

# Output layer
Out = Dense(units=num_classes, activation='softmax', 
            kernel_initializer=keras.initializers.glorot_normal(seed=3), 
            name='Output')(FC2)

model1 = Model(inputs=VGG16_MODEL.input,outputs=Out)


from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import TensorBoard

checkpoint = ModelCheckpoint("gesturenew.h5", monitor='accuracy', verbose=1,
    save_best_only=True, mode='auto', period=1)

reduce = ReduceLROnPlateau(monitor='accuracy', factor=0.2, patience=5, min_lr=0.00001, verbose = 1)

logdir='logsgesture'
tensorboard_Visualization = TensorBoard(log_dir=logdir, histogram_freq=True)


train_samples = 9600
validation_samples = 2400

epochs = 50

batch_size = 128

model1.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy']
              )
              
model1.fit(train_generator,
           steps_per_epoch = train_samples//batch_size,
           epochs = epochs,
           callbacks = [checkpoint, reduce, tensorboard_Visualization],
           validation_data = validation_generator,
           validation_steps = validation_samples//batch_size)
