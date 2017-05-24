import load_data

from keras.models import Sequential, Model
from keras.layers import Cropping2D, Lambda, Convolution2D, Flatten, Dense, Dropout, MaxPooling2D, Activation, Conv2D
from keras.optimizers import Adam

BATCH_SIZE = 128
LEARNING_RATE = 0.0001
EPOCH = 10

# refer to the paper "End to End Learning for Self-Driving Cars"
model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(90, 320, 3)))
model.add(Conv2D(3, 5, 5, subsample=(2, 2)))
model.add(MaxPooling2D())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Conv2D(24, 5, 5, subsample=(2, 2)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Conv2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Conv2D(48, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.summary()

model.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE))

# add data here to train the model
train_samples, validation_samples = load_data.load_csv_records(
    ['../data0/driving_log.csv', '../data1/driving_log.csv', '../data2/driving_log.csv'])
# dataset provided by udacity: (['../data0/driving_log.csv'])
# dataset provided by keyboard: (['../data0/driving_log.csv', '../data1/driving_log.csv', '../data2/driving_log.csv'])
train_generator = load_data.generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = load_data.generator(validation_samples, batch_size=BATCH_SIZE)

model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator,
                    nb_val_samples=len(validation_samples), nb_epoch=EPOCH)

model.save('model.h5')
