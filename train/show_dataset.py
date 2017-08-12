#!/usr/bin/env python3

import cv2
import numpy as np
import os
import sys
import tensorflow as tf
import pickle
    
def main():
    train_path = sys.argv[1]
    train_name = os.path.basename(train_path).split(".")[0]

    # Load data
    with open(train_path, 'rb') as f:
        train_x, train_y = pickle.load(f)
    print(np.mean(train_y, axis=0), np.std(train_y, axis=0))
    
    # Store png files
    if not os.path.exists(train_name):
        os.makedirs("../data/%s" % train_name)
    for i, (x, y) in enumerate(zip(train_x, train_y)):
        cv2.imwrite("../data/%s/%s_%06i_%.3f_%.3f.png" % (train_name, train_name, i, y[0], y[1]), x)
        sys.stdout.write("%.3f\r" % (100.0 * i / len(train_x)))
    print("Done")
    print(train_x.shape, train_y.shape)

    
if __name__ == "__main__":
    main()
