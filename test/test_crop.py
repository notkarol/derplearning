#!/usr/bin/python3

import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
import sys
import derp.util
import derp.imagemanip

def prepare_thumb(frame, bbox):
    patch = frame[bbox.y : bbox.y + bbox.h, bbox.x : bbox.x + bbox.w]
    thumb = cv2.resize(patch, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_AREA)
    return thumb


def main(args):

    # Prepare frame
    frame = np.array(PIL.Image.open("../test/100deg.jpg"))
    frame_above = frame[:80]
    frame_center = frame[10:90]
    frame_below = frame[20:]

    # Prepare configs for cutting out patches
    source_above_config = {'hfov': 50, 'vfov': 40, 'yaw': 0, 'pitch': 5,
                           'width': 100, 'height': 80}
    source_center_config = {'hfov': 50, 'vfov': 40, 'yaw': 0, 'pitch': 0,
                            'width': 100, 'height': 80}
    source_below_config = {'hfov': 50, 'vfov': 40, 'yaw': 0, 'pitch': -5,
                           'width': 100, 'height': 80}
    target_config = {'hfov': 10, 'vfov': 10, 'yaw': 0, 'pitch': 0,
                     'width': 100, 'height': 100, 'vcenter': -10, 'hcenter': -10}

    # Plot cutouts
    fig, axs = plt.subplots(3, 2, figsize=(10, 10))
    for (ax1, ax2), config, frame in zip(axs, (source_above_config,
                                               source_center_config,
                                               source_below_config),
                                         (frame_above, frame_center, frame_below)):
        bbox = derp.imagemanip.get_patch_bbox(target_config, config)
        print(bbox)
        thumb = prepare_thumb(frame, bbox)
        ax1.imshow(frame)
        ax1.plot((bbox.x, bbox.x), (bbox.y, bbox.y + bbox.h - 1), 'm')
        ax1.plot((bbox.x + bbox.w - 1, bbox.x + bbox.w - 1), (bbox.y, bbox.y + bbox.h - 1), 'm')
        ax1.plot((bbox.x, bbox.x + bbox.w - 1), (bbox.y, bbox.y), 'm')
        ax1.plot((bbox.x, bbox.x + bbox.w - 1), (bbox.y + bbox.h - 1, bbox.y + bbox.h - 1), 'm')
        if thumb is not None:
            ax2.imshow(thumb)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()     
    sys.exit(main(args))
