#!/usr/bin/python3
import argparse
import derp.util
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal

def main(args):

    # Prepare path to and load state csv
    state_path = args.path if 'state.csv' in args.path else args.path + '/state.csv'
    timestamps, headers, states = derp.util.read_csv(state_path)

    # Prepare plot
    fig, axs = plt.subplots(3, 3, figsize=(16, 10))
    for header in ['speed', 'steer']:
        axs[0][0].plot(timestamps, states[:, headers.index(header)], label=header)
    for header in ['offset_speed', 'offset_steer', 'use_offset_speed', 'use_offset_steer']:
        axs[0][1].plot(timestamps, states[:, headers.index(header)], label=header)
    for header in ['temp']:
        axs[0][2].plot(timestamps, states[:, headers.index(header)], label=header)
    b, a = signal.butter(2, 0.1, output='ba')
    for header in ['accel_x', 'accel_y', 'accel_z']:
        axs[1][0].plot(timestamps, signal.filtfilt(b, a, states[:, headers.index(header)]),
                       label=header)
    axs[1][1] = plt.subplot(335, projection='polar')
    for header in ['euler_h', 'euler_r', 'euler_p']:
        axs[1][1].plot(states[:, headers.index(header)] / 180 * np.pi,
                       timestamps - timestamps.min(), label=header)
    for header in ['quaternion_w', 'quaternion_x', 'quaternion_y', 'quaternion_z']:
        axs[1][2].plot(timestamps, states[:, headers.index(header)], label=header)
    for header in ['gyro_x', 'gyro_y', 'gyro_z']:
        axs[2][0].plot(timestamps, signal.filtfilt(b, a, states[:, headers.index(header)]),
                       label=header)
    for header in ['gravity_x', 'gravity_y', 'gravity_z']:
        axs[2][1].plot(timestamps, states[:, headers.index(header)], label=header)
    for header in ['magneto_x', 'magneto_y', 'magneto_z']:
        axs[2][2].plot(timestamps, states[:, headers.index(header)], label=header)        

    # Clean up axes
    for ax_y in axs:
        for ax in ax_y:
            ax.legend()
            ax.grid(True)

    # Save sensors data
    plt.savefig('sensors.png', bbox_inches='tight', dpi=160)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help="path to data")
    args = parser.parse_args()
    main(args)
