import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import argparse
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, SeparableConv2D
from keras.models import model_from_yaml
from keras.layers.normalization import BatchNormalization

def create_model(input_shape, n_output, n_blocks=4):
    model = Sequential()
    model.add(Conv2D(96, (5, 5), padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=2))

    for i in range(n_blocks):
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('elu'))
        model.add(MaxPooling2D(pool_size=2))

    model.add(Flatten())
    model.add(Dense(100))
    model.add(Activation('elu'))
    
    model.add(Dense(n_output))
    
    return model

def main():
    
    parser = argparse.ArgumentParser(description='PyTorch Bezier Curve Predictor')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument("--opt", type=str, default='adam',
                        help='optimizers (sgd, adam)')
    parser.add_argument('--epochs', type=int, default=32, metavar='N',
                        help='number of epochs to train (default: 32)')
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

    print(X_train.shape, y_train.shape)
    
    # initiate RMSprop optimizer
    model = create_model(X_train.shape[1:], y_train.shape[1])
    if args.opt == 'adam':
        print("Using adam")
        opt = keras.optimizers.adam(lr=args.lr)
    else:
        print("Using sgd")
        opt = keras.optimizers.sgd(lr=args.lr, momentum=args.momentum)
        
    model.compile(loss='mean_squared_error', optimizer=opt)
    model.summary()

    for i in range(args.epochs):
        # Update learning rate
        lr = args.lr / (i + 1)
        model.optimizer.lr.assign(lr)
        print("Setting learning rate to", lr)
        
        # Fit model
        model.fit(X_train, y_train, batch_size=args.bs, epochs=1,
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

