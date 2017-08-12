#!/usr/bin/env python3

import cv2
import numpy as np
import os
import pickle
import sys
import srperm as srp

def main():
    source_size = (640, 480)
    crop_size = (640, 320)
    crop_x = (source_size[0] - crop_size[0]) // 2
    crop_y = (source_size[1] - crop_size[1]) // 2
    target_size = (80, 40)

    # Output labels
    thumbs_out = []
    labels_out = []
    
    name = sys.argv[1]
    folders = sys.argv[2:]
    for folder in folders:

        # Prepare timestamps and labels
        timestamps = []
        labels = []
        csv_path = os.path.join(folder, 'video.csv')
        with open(csv_path) as f:
            headers = f.readline()
            for row in f:
                timestamp, speed, nn_speed, steer, nn_steer = row.split(",")
                timestamps.append([float(timestamp)])
                labels.append((float(speed), float(steer)))
        timestamps = np.array(timestamps, dtype=np.double)
        labels = np.array(labels, dtype=np.float32)

        # Prepare video frames by extracting the patch and thumbnail for training
        video_path = os.path.join(folder, 'video.mp4')
        video_cap = cv2.VideoCapture(video_path)
        counter = 0
        while video_cap.isOpened() and counter < len(labels):

            # Get the frame
            ret, frame = video_cap.read()
            if not ret: break

            #shift values
            drot = 0
            mshift = 0

            #permutate frame
            pframe = srp.shiftimg(frame, drot, mshift)

            # Prepare patch
            patch = pframe[crop_x : crop_x + crop_size[0], crop_y : crop_y + crop_size[1], :]
            thumb = cv2.resize(patch, target_size)

            # Prepare label, shape, thumb            
            labels_out.append( [labels[counter][0],srp.shiftsteer(labels[counter][1],drot,mshift) ] )
            thumbs_out.append(thumb)

            counter += 1

        # Clean up video capture
        video_cap.release()

    thumbs_out = np.array(thumbs_out, dtype=np.uint8)
    labels_out = np.array(labels_out, dtype=np.float32)

    # store pickles
    print("Storing ", thumbs_out.shape, labels_out.shape)
    with open("%s.pkl" % name, 'wb') as f:
        pickle.dump((thumbs_out, labels_out), f, protocol=pickle.HIGHEST_PROTOCOL)
        
if __name__ == "__main__":
    main()
