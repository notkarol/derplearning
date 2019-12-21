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
        self.scale = scale  # Image Scale Factor
        self.folder = folder
        self.config_path = self.folder / "config.yaml"
        self.config = util.load_config(self.config_path)

        self.red = np.array([0, 0, 255], dtype=np.uint8)
        self.green = np.array([32, 192, 32], dtype=np.uint8)
        self.blue = np.array([255, 128, 0], dtype=np.uint8)
        self.yellow = np.array([0, 255, 255], dtype=np.uint8)
        self.cyan = np.array([255, 255, 0], dtype=np.uint8)
        self.magenta = np.array([255, 0, 255], dtype=np.uint8)
        self.gray50 = np.array([128, 128, 128], dtype=np.uint8)
        self.gray75 = np.array([192, 192, 192], dtype=np.uint8)
        self.white = np.array([255, 255, 255], dtype=np.uint8)
        self.orange = np.array([255, 128, 0], dtype=np.uint8)
        self.marker_color = {
            "": self.gray50,
            "good": self.green,
            "risk": self.orange,
            "junk": self.red,
        }

        self.topics = {}
        for topic in derp.util.TOPICS:
            self.topics[topic] = []
            if derp.util.topic_exists(self.folder, topic):
                topic_fd = derp.util.topic_file_reader(self.folder, topic)
                self.topics[topic] = [msg for msg in derp.util.TOPICS[topic].read_multiple(topic_fd)]
                topic_fd.close()
            else:
                print("Topic %s not found in folder %s" % (topic, self.folder))
            print(topic, len(self.topics[topic]))
        self.n_frames = len(self.topics['camera'])
        if 'label' in self.topics:
            self.labels = [msg.qualiy for msg in self.topics['label']]
        else:
            self.labels = [derp.util.TOPICS['label'].QualityEnum.unknown
                           for _ in range(self.n_frames)]
        control_times = [msg.timestampPublished for msg in self.topics['control']]
        self.camera_times = [msg.timestampPublished for msg in self.topics['camera']]
        self.speeds = derp.util.interpolate(self.camera_times, control_times,
                                            [msg.speed for msg in self.topics['control']])
        self.steers = derp.util.interpolate(self.camera_times, control_times,
                                            [msg.steer for msg in self.topics['control']])
        b, a = signal.butter(3, 0.05, output="ba")
        self.steers_butter = signal.filtfilt(b, a, self.steers)
        self.frame_id = 0
        self.seek(self.frame_id)
        self.init_window()

    def __del__(self):
        cv2.destroyAllWindows()

    def legal_position(self, pos):
        return 0 <= pos < self.n_frames

    def update_label(self, id1, id2, marker):
        # Update the labels that are to be stored
        beg, end = min(id1, id2), max(id1, id2)

        # The update is applied inclusive to both ends of the id range
        for i in range(beg, end + 1):
            self.labels[i] = marker

        # Update the visual label bar
        beg_pos, end_pos = self.frame_pos(beg), self.frame_pos(end)
        self.label_bar[beg_pos : end_pos + 1] = self.marker_color[marker]

    def seek(self, frame_id=None):
        if frame_id is None:
            frame_id = self.frame_id + 1
        if not self.legal_position(frame_id):
            print("read failed illegal", frame_id)
            return False
        jpg_arr = np.frombuffer(self.topics['camera'][self.frame_id].jpg, np.uint8)
        frame = cv2.imdecode(jpg_arr, cv2.IMREAD_COLOR)
        self.frame = cv2.resize(
            frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA
        )
        self.frame_id = frame_id
        return True

    def draw_bar_timemarker(self):
        self.window[self.fh :, self.frame_pos(self.frame_id), :] = self.marker_color[self.marker]

    def draw_bar_blank(self):
        self.window[self.fh :, :, :] = 0

    def draw_bar_status(self):
        self.window[self.fh : self.fh + int(self.bhh // 10), self.fwi, :] = self.label_bar

    def draw_bar_zeroline(self):
        self.window[self.fh + self.bhh, self.fwi, :] = self.gray75

    def draw_horizon_bar(self):
        percent = self.config['camera']["pitch"] / self.config['camera']["vfov"] + 0.5
        self.window[int(self.fh * percent), :, :] = self.magenta

    def draw_graph(self, data_vector, color):
        # interpolate the data vector to fill in gaps
        d_interpld = interp1d(self.state_x, data_vector)

        # convert data vector to a data array the size of the window's x dimension
        data_bar = np.array([-d_interpld(x) * self.bhh + 0.5 for x in self.window_x], dtype=np.int)

        # All locations where we need to draw lines
        data_jump_locs = []

        for loc in np.where(abs(data_bar[:-1] - data_bar[1:]) >= 2)[0]:
            rr, cc, val = line_aa(
                data_bar[loc] + self.fh + self.bhh,
                loc,
                data_bar[loc + 1] + self.fh + self.bhh,
                loc + 1,
            )
            data_jump_locs.append((rr, cc))

        """ Draw the speed and steering lines on the bar below the video. 
        Takes about 1ms to run through the for loops"""
        self.window[data_bar + self.fh + self.bhh, self.fwi, :] = color
        for rr, cc in data_jump_locs:
            self.window[rr, cc, :] = color

    # This function updates the viewer calling all appropriate functions
    def display(self):

        # Update the pixels of the frame
        self.window[: self.frame.shape[0], :, :] = self.frame

        self.draw_horizon_bar()
        self.draw_bar_blank()
        self.draw_bar_timemarker()
        self.draw_bar_zeroline()
        self.draw_bar_status()
        self.draw_graph(data_vector=self.speeds, color=self.cyan)
        self.draw_graph(data_vector=self.steers, color=self.green)
        self.draw_graph(data_vector=self.steers_butter, color=self.white)

        if self.model:
            self.draw_graph(data_vector=self.m_speeds, color=self.blue)
            self.draw_graph(data_vector=self.m_steers, color=self.red)

        # Display the newly generated window
        cv2.imshow("Labeler %s" % self.folder, self.window)

    def save_labels(self):
        with open(self.labels_path, "w") as f:
            f.write("timestamp,status\n")
            for timestamp, label in zip(self.timestamps, self.labels):
                f.write("%.6f,%s\n" % (timestamp, label))
        with open(self.config_path, "w") as f:
            f.write(yaml.dump(self.config))
        print("Saved labels at ", self.labels_path)

    # Handles keyboard inputs during data labeling process.
    def handle_input(self):
        key = cv2.waitKey(10) & 0xFF
        if key == ord("q") or key == 27:  # escape
            return False
        elif key == ord("p") or key == ord(" "):
            self.paused = not self.paused
        elif key == ord("g"):
            self.marker = "good"
            self.show = True
        elif key == ord("r"):
            self.marker = "risk"
            self.show = True
        elif key == ord("t"):
            self.marker = "junk"
            self.show = True
        elif key == ord("c"):
            self.marker = ""
            self.show = True
        elif key == ord("s"):
            self.save_labels()
        elif key == 82:  # up
            self.seek(self.frame_id + 10)
        elif key == 84:  # down
            self.seek(self.frame_id - 10)
        elif key == 81:  # left
            self.seek(self.frame_id - 1)
        elif key == 83:  # right
            self.seek(self.frame_id + 1)
        elif key == 85:  # page up
            self.config['camera']["pitch"] -= 0.1
            self.show = True
        elif key == 86:  # page down
            self.config['camera']["pitch"] += 0.1
            self.show = True
        elif key == ord("`"):
            self.seek(0)
        elif ord("1") <= key <= ord("9"):
            self.seek(int(self.n_frames * (key - ord("0")) / 10))
        elif key == ord("0"):
            self.seek(self.n_frames - 1)
        elif key != 255:
            print("Unknown key press: [%s]" % key)
        return True

    def frame_pos(self, frame_id):
        return min(self.fw - 1, int(frame_id / self.n_frames * self.fw))

    def init_window(self):
        self.fh = self.frame.shape[0]
        self.fw = self.frame.shape[1]
        self.bhh = 150  # bar half height

        self.fwi = np.arange(self.fw)  # frame width index
        self.window_shape = list(self.frame.shape)
        self.window_shape[0] += self.bhh * 2 + 2
        self.window = np.zeros(self.window_shape, dtype=np.uint8)

        # state_x is a vector containing the indicies of the x coordinate of the state graph
        self.state_x = np.linspace(0, 1, self.n_frames)
        self.window_x = np.linspace(0, 1, self.fw)

        self.paused = True
        self.show = False
        self.marker = ""
        self.label_bar = np.ones((self.fw, 3), dtype=np.uint8) * self.gray50
        for i in range(self.n_frames):
            self.update_label(i, i, self.labels[i])

    # Create arrays of predicted speed and steering using the designated model
    def predict(self, config, model_path):
        # Initialize the model output data vectors:
        self.m_speeds = np.zeros(self.n_frames, dtype=float)
        self.m_steers = np.zeros(self.n_frames, dtype=float)

        # opens the video config file
        video_config = util.load_config("%s/%s" % (self.folder, "car.yaml"))
        bot = Inferer(
            video_config=video_config,
            model_config=config,
            folder=self.folder,
            model_path=model_path,
        )

        # Move the capture function to the start of the video

        for i in range(self.n_frames):
            frame = cv2.imdecode(self.topics['camera'][i].jpg,  cv2.IMREAD_COLOR)

            if not ret or frame is None:
                print("read failed frame", frame_id)
            if i % 2 == 0:
                results = bot.evaluate(frame, self.timestamps[i], config, model_path)
                self.m_speeds[i], self.m_steers[i], batch = results
            else:
                self.m_speeds[i] = self.m_speeds[i - 1]
                self.m_steers[i] = self.m_steers[i - 1]

    def run_labeler(self, config_path=None, model_path=None):

        # create driving predictions:
        self.model = None
        if model_path:
            self.model = model_path
            config = util.load_config(path=config_path)
            self.predict(config_path, model_path)

        # Start the display window
        self.display()

        # Loop through video
        while True:
            if not self.paused and self.frame_id < self.n_frames:
                self.update_label(self.frame_id, self.frame_id, self.marker)
                self.seek()
                self.show = True

            if self.show:
                self.display()
                self.show = False

            if not self.handle_input():
                break


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="recording path location")
    parser.add_argument("--scale", type=float, default=1.0, help="frame rescale ratio")
    parser.add_argument("--config", type=str, default=None, help="physical configuration")
    parser.add_argument("--infer", type=str, default=None, help="infer configuration")
    parser.add_argument(
        "--controls", type=bool, default=True, help="Opens raceday.md with label.py controls notes."
    )

    args = parser.parse_args()

    if args.controls:
        print(
            """## Navigation
You can maneuver through the tool through the arrow keys.
* Left Arrow: Move backward in time 1 frame
* Right Arrow: Move forward in time 1 frame
* Up Arrow: Move forward in time 1 second
* Down Arrow: Move backward in time 1 second
* `: Move to the beginning of the file.
* 1: Move to 10% into the file.
* 2: Move to 20% into the file.
* 3: Move to 30% into the file.
* 4: Move to 40% into the file.
* 5: Move to 50% into the file.
* 6: Move to 60% into the file.
* 7: Move to 70% into the file.
* 8: Move to 80% into the file.
* 9: Move to 90% into the file.
* 0: Move to 100% into the file.

## Modes
* g: good data that we should use for data
* r: risky data that we mark is interesting but probably don't want to train
* t: trash data that we wish to junk and not use again

You an also clear the marker so that when you maneuver through the video you don't update the mode at the time.
* c: clear marker

## Save the labels to a file:
* s: Save video

"""
        )

    labeler = Labeler(folder=args.path, scale=args.scale)
    labeler.run_labeler(args.config, args.infer)
