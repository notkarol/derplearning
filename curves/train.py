import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import argparse
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import model_from_yaml
from keras.layers.normalization import BatchNormalization

def create_model(input_shape, n_output):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape, data_format='channels_first'))
    model.add(BatchNormalization())    
    model.add(Activation('elu'))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(BatchNormalization())    
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())    
    model.add(Activation('elu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())    
    model.add(Activation('elu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('elu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_output))
    return model

def main():
    
    parser = argparse.ArgumentParser(description='PyTorch Bezier Curve Predictor')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--bs', type=int, default=32, metavar='N',
                        help='batch size (default: 32)')
    parser.add_argument('--gpu', type=int, default=0, help='index of GPU to use')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    # Load Data
    X_train = np.load('X_train.npy')
    X_val = np.load('X_val.npy')
    y_train = np.load('y_train.npy')
    y_val = np.load('y_val.npy')

    print(np.min(X_train), np.max(X_train))
    print(np.min(X_val), np.max(X_val))
    print(np.min(y_train), np.max(y_train))
    print(np.min(y_val), np.max(y_val))
    
    # initiate RMSprop optimizer
    model = create_model(X_train.shape[1:], y_train.shape[1])
    opt = keras.optimizers.sgd(lr=args.lr, momentum=args.momentum)
    model.compile(loss='mean_squared_error', optimizer=opt)
    model.summary()
    
    model.fit(X_train, y_train, batch_size=args.bs, epochs=args.epochs,
              validation_data=(X_val, y_val), shuffle=True)

    # serialize model to YAML
    model_yaml = model.to_yaml()
    with open("model.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")


if __name__ == "__main__":
    main()

