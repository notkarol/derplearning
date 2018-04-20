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
from keras.layers.merge import concatenate, add
from keras.layers.normalization import BatchNormalization
from roadgen3d import Roadgen

import yaml
with open("config/line_model.yaml", 'r') as yamlfile:
    cfg = yaml.load(yamlfile)

'''
When run file builds and trains a model saving it to the same directory as the file.

Function list:
  create_model
'''

# defines the structure of the model to be trained
''' Note, I cut the number of blocks from 4 to 2 to deal with an out of bounds error
     on the converging layers when using 96x16 dim. current dim is 192x32 '''
def create_model(input_shape, n_output, n_blocks=2):
    model = Sequential()
    model.add(Conv2D(96, (5, 5), padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=2))

    for i in range(n_blocks):
        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('elu'))
        model.add(Conv2D(32, (3, 3), padding='same'))
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
    parser.add_argument('--sets', type=int, default=30,
            help='number of batches to train on')
    parser.add_argument('--gpu', type=int, default=0, help='index of GPU to use')
    #parser directory assistance
    parser.add_argument('--train_data', default=cfg['dir']['train_data'], 
            help='training data source directory(default: %s)' % cfg['dir']['train_data'])
    parser.add_argument('--val_data', default=cfg['dir']['val_data'], 
            help='validation data source directory(default: %s)' % cfg['dir']['val_data'])
    parser.add_argument('--model_dir', default=cfg['dir']['model'], 
            help='model files save directory(default: %s)' % cfg['dir']['model'])
    parser.add_argument('--model_name', default=cfg['dir']['model_name'], 
            help='line interpreter model name(default: %s)' % cfg['dir']['model_name'])
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    #invoke the roagen class for data pipe management fucntions:
    pipe = Roadgen(cfg)
    
    # Load a data sameple:
    X_val = pipe.normalize(pipe.batch_loader(args.val_data, 0) )
    y_val = pipe.label_norm(np.load('%s/y_%03i.npy' % (args.val_data, 0) ) )
    print(X_val.shape, y_val.shape)

    
    # initiate RMSprop optimizer
    model = create_model(X_val.shape[1:], y_val.shape[1])
    if args.opt == 'adam':
        print("Using adam")
        opt = keras.optimizers.adam(lr=args.lr)
    else:
        print("Using sgd")
        opt = keras.optimizers.sgd(lr=args.lr, momentum=args.momentum)
        
    model.compile(loss='mean_squared_error', optimizer=opt)
    model.summary()

    #file management stuff
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    for i in range(args.epochs):
        # Update learning rate
        lr = args.lr / (i + 1)
        model.optimizer.lr.assign(lr)
        print("Setting learning rate to", lr)
        
        for j in range(args.sets):
            print("Epoch %i/%i - Training set %03i" % (i+1,args.epochs, j) )
            # Load Data
            X_train = pipe.normalize(pipe.batch_loader(args.train_data, j) )
            #X_val = pipe.batch_loader(args.train_data, 0)
            y_train = pipe.label_norm(np.load('%s/y_%03i.npy' % (args.train_data, j) ) )
            #y_val = np.load('%s/y_%03i.npy' % (args.val_data, 0))
            # Fit model
            model.fit(X_train, y_train, batch_size=args.bs, epochs=1,
                      validation_data=(X_val, y_val), shuffle=True)
        
        # serialize model to YAML
        model_yaml = model.to_yaml()
        with open("%s/%s.yaml" % (args.model_dir, args.model_name), "w") as yaml_file:
            yaml_file.write(model_yaml)
        # serialize weights to HDF5
        model.save_weights("%s/%s.h5" % (args.model_dir, args.model_name))
        print("Saved model to disk")


if __name__ == "__main__":
    main()

