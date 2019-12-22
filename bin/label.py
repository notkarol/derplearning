import numpy as np
import cv2
import os
from pathlib import Path
import sys
import time
import yaml
import argparse
import derp.util
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.interpolate import interp1d
from skimage.draw import line_aa
import derp.util as util


class Labeler:
    def __init__(self, folder, scale=1):
        self.scale = scale
        self.folder = folder
        self.config_path = self.folder / 'config.yaml'
        self.config = util.load_config(self.config_path)
        self.marker_colors = [(128, 128, 128), (0, 255, 0), (255, 128, 0), (255, 0, 0)]
        self.topics = {}
        for topic in derp.util.TOPICS:
            if not derp.util.topic_exists(self.folder, topic):
                continue
            topic_fd = derp.util.topic_file_reader(self.folder, topic)
            self.topics[topic] = [msg for msg in derp.util.TOPICS[topic].read_multiple(topic_fd)]
            topic_fd.close()
            print(topic, len(self.topics[topic]), sep='\t')
        self.n_frames = len(self.topics['camera'])
        control_times = [msg.timestampPublished for msg in self.topics['control']]
        self.camera_times = [msg.timestampPublished for msg in self.topics['camera']]
        self.speeds = derp.util.interpolate(self.camera_times, control_times,
                                            [msg.speed for msg in self.topics['control']])
        self.steers = derp.util.interpolate(self.camera_times, control_times,
                                            [msg.steer for msg in self.topics['control']])
        b, a = signal.butter(3, 0.05, output='ba')
        self.steers_butter = signal.filtfilt(b, a, self.steers)
        self.frame_id = 0
        self.seek(self.frame_id)
        self.frame_height = self.frame.shape[0]
        self.frame_width = self.frame.shape[1]
        self.bar_half_height = 150
        self.frame_width_index = np.arange(self.frame_width)
        self.window_shape = list(self.frame.shape)
        self.window_shape[0] += self.bar_half_height * 2 + 2
        self.window = np.zeros(self.window_shape, dtype=np.uint8)
        self.state_x = np.linspace(0, 1, self.n_frames)
        self.window_x = np.linspace(0, 1, self.frame_width)
        self.paused = True
        self.show = False
        self.marker = 'unknown'
        self.label_bar = np.ones((self.frame_width, 3), dtype=np.uint8) * (128, 128, 128)
        if 'label' in self.topics and len(self.topics['label']) >= self.n_frames:
            self.labels = [str(msg.quality) for msg in self.topics['label']]
        else:
            self.labels = ['unknown' for _ in range(self.n_frames)]
        for i in range(self.n_frames):
            self.update_label(i, i, self.labels[i])

    def __del__(self):
        cv2.destroyAllWindows()

    def update_label(self, first_index, last_index, marker):
        if marker == 'unknown':
            return False
        for index in range(first_index, last_index + 1):
            self.labels[index] = marker
        beg_pos, end_pos = self.frame_pos(first_index), self.frame_pos(last_index)
        self.label_bar[beg_pos : end_pos + 1] = self.marker_colors[derp.util.TOPICS['label'].QualityEnum.__dict__[marker]]

    def seek(self, frame_id=None):
        if frame_id is None:
            frame_id = self.frame_id + 1
        if frame_id < 0:
            frame_id = 0
            self.paused = True
        if frame_id >= self.n_frames:
            frame_id = self.n_frames - 1
            self.paused = True
        jpg_arr = np.frombuffer(self.topics['camera'][self.frame_id].jpg, np.uint8)
        frame = cv2.imdecode(jpg_arr, cv2.IMREAD_COLOR)
        self.frame = cv2.resize(
            frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA
        )
        self.frame_id = frame_id
        return True

    def draw_bar_time_marker(self):
        self.window[self.frame_height :, self.frame_pos(self.frame_id), :] = self.marker_colors[derp.util.TOPICS['label'].QualityEnum.__dict__[self.marker]]

    def draw_bar_blank(self):
        self.window[self.frame_height :, :, :] = 0

    def draw_bar_status(self):
        self.window[self.frame_height : self.frame_height + int(self.bar_half_height // 10),
                    self.frame_width_index, :] = self.label_bar

    def draw_bar_zeroline(self):
        self.window[self.frame_height + self.bar_half_height,
                    self.frame_width_index, :] = (192, 192, 192)

    def draw_horizon_bar(self):
        percent = self.config['camera']['pitch'] / self.config['camera']['vfov'] + 0.5
        self.window[int(self.frame_height * percent), :, :] = (255, 0, 255)

    def draw_graph(self, data_vector, color):
        d_interpld = interp1d(self.state_x, data_vector)
        data_bar = np.array([-d_interpld(x) * self.bar_half_height + 0.5 for x in self.window_x], dtype=np.int)
        data_jump_locs = []
        for loc in np.where(abs(data_bar[:-1] - data_bar[1:]) >= 2)[0]:
            rr, cc, val = line_aa(
                data_bar[loc] + self.frame_height + self.bar_half_height,
                loc,
                data_bar[loc + 1] + self.frame_height + self.bar_half_height,
                loc + 1,
            )
            data_jump_locs.append((rr, cc))
        self.window[data_bar + self.frame_height + self.bar_half_height, self.frame_width_index, :] = color
        for rr, cc in data_jump_locs:
            self.window[rr, cc, :] = color

    def display(self):
        self.window[: self.frame.shape[0], :, :] = self.frame
        self.draw_horizon_bar()
        self.draw_bar_blank()
        self.draw_bar_time_marker()
        self.draw_bar_zeroline()
        self.draw_bar_status()
        self.draw_graph(data_vector=self.speeds, color=(255,255,0))
        self.draw_graph(data_vector=self.steers, color=(0,255,0))
        self.draw_graph(data_vector=self.steers_butter, color=(255,255,255))
        cv2.imshow('Labeler %s' % self.folder, self.window)

    def save_labels(self):
        with derp.util.topic_file_writer(self.folder, 'label') as label_fd:
            for label_i, label in enumerate(self.labels):
                msg = derp.util.TOPICS['label'].new_message(
                    timestampCreated=derp.util.get_timestamp(),
                    timestampPublished=self.topics['camera'][label_i].timestampPublished,
                    timestampWritten=derp.util.get_timestamp(),
                    quality=label)
                msg.write(label_fd)
        print('Saved labels in', self.folder)

    def handle_keyboard_input(self):
        key = cv2.waitKey(10) & 0xFF
        if key == 27: return False # ESC
        elif key == ord(' '): self.paused = not self.paused
        elif key == ord('g'): self.marker = 'good'
        elif key == ord('r'): self.marker = 'risky'
        elif key == ord('t'): self.marker = 'trash'
        elif key == ord('c'): self.marker = 'unknown'
        elif key == ord('s'): self.save_labels()
        elif key == 82: self.seek(self.frame_id + 10) # up
        elif key == 84: self.seek(self.frame_id - 10) # down
        elif key == 81: self.seek(self.frame_id - 1) # left
        elif key == 83: self.seek(self.frame_id + 1) # right
        elif key == 85: self.config['camera']['pitch'] -= 0.1 # page up
        elif key == 86: self.config['camera']['pitch'] += 0.1 # page down
        elif ord('1') <= key <= ord('6'): self.seek(int(self.n_frames * (key - ord('0') - 1) / 5))
        elif key != 255: print('Unknown key press: [%s]' % key)
        self.show = True
        return True

    def frame_pos(self, frame_id):
        return min(self.frame_width - 1, int(frame_id / self.n_frames * self.frame_width))

    def run(self):
        self.display()
        while True:
            if not self.paused:
                self.update_label(self.frame_id, self.frame_id, self.marker)
                self.show = self.seek()
            if self.show:
                self.display()
                self.show = False
            if not self.handle_keyboard_input():
                break


if __name__ == '__main__':
    print('Arrow keys to navigate, ` through 0 to teleport, s to save, ESC to quit')
    print('To assign a label state as you play: g=good, r=risky, t=trash, and c to clear marker')
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=Path, help='recording path location')
    parser.add_argument('--scale', type=float, default=1.0, help='frame rescale ratio')
    args = parser.parse_args()
    labeler = Labeler(folder=args.path, scale=args.scale)
    labeler.run()
