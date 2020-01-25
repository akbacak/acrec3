import os
import glob
import keras
#from keras_video import VideoFrameGenerator# use sub directories names as classes
from generator import *
#from utils import *
from keras.applications import InceptionV3
from keras import layers
from keras.layers.recurrent import LSTM
from keras.models import Input,Model,InputLayer
from keras.layers import Dense, Dropout, TimeDistributed, Bidirectional, BatchNormalization
import tensorflow as tf
from keras import optimizers


classes = [i.split(os.path.sep)[1] for i in glob.glob('videos/*')]
classes.sort()# some global params
SIZE = (112, 112)
CHANNELS = 3
NBFRAME = 16
BS = 8         # pattern to get videos and classes
glob_pattern='videos/{classname}/*'# for data augmentation
data_aug = keras.preprocessing.image.ImageDataGenerator(
    zoom_range=.1,
    horizontal_flip=True,
    rotation_range=8,
    width_shift_range=.2,
    height_shift_range=.2)# Create video frame generator
train = VideoFrameGenerator(
    classes=classes, 
    glob_pattern=glob_pattern,
    nb_frames=NBFRAME,
    split=.33, 
    shuffle=True,
    batch_size=BS,
    target_shape=SIZE,
    nb_channel=CHANNELS,
    transformation=data_aug,
    use_frame_cache=False)

valid = train.get_validation_generator()

#import keras_video.utils
#keras_video.utils.show_sample(train)
#show_sample(train)

num_classes = 51

def get_model(shape=(NBFRAME, 112, 112, 3), nbout=3):
    # Define model

    video = keras.Input(shape = (NBFRAME,112,112,3), name='video')
    cnn = InceptionV3(weights='imagenet',include_top=False, pooling='avg')
    cnn.trainable =False

    frame_features = layers.TimeDistributed(cnn)(video)
    blstm_1 = Bidirectional(LSTM(512, dropout=0.1, recurrent_dropout=0.5, return_sequences= True))(frame_features)
    blstm_2 = Bidirectional(LSTM(512, dropout=0.1, recurrent_dropout=0.5, return_sequences= False))(blstm_1)
    Dense_2   = Dense(256, activation = 'sigmoid' )(blstm_2)
    batchNorm = BatchNormalization()(Dense_2)
    enver   = Dense(128, activation = 'sigmoid')(batchNorm)
    batchNorm2= BatchNormalization()(enver)
    Dense_3   = Dense(num_classes, activation='sigmoid')(batchNorm2)
    model = keras.models.Model(input = video , output = Dense_3)

    model.summary()
    #plot_model(model, show_shapes=True,
    #           to_file='model.png')

    from keras.optimizers import SGD
    sgd = SGD(lr=0.002, decay = 1e-5, momentum=0.9, nesterov=True)

    model.compile(loss = 'categorical_crossentropy',  optimizer=sgd, metrics=['acc'])
    return model


INSHAPE=(NBFRAME,) + SIZE + (CHANNELS,) # (5, 112, 112, 3)
model = get_model(INSHAPE, len(classes))

optimizer = keras.optimizers.Adam(0.001)

callbacks = [
    keras.callbacks.ReduceLROnPlateau(verbose=1),
    keras.callbacks.ModelCheckpoint(
        'chkp/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
        verbose=1),
]

EPOCHS=120 
hist = model.fit_generator(
    train,
    validation_data=valid,
    verbose=1,
    epochs=EPOCHS,
    callbacks=callbacks
)
