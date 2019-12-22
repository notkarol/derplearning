#!/usr/bin/env python3
import argparse
from pathlib import Path
import time
import cv2
import numpy as np
from skimage.draw import line_aa
import derp.util


class Labeler:
    def __init__(self, folder, scale=1):
        self.scale = scale
        self.folder = folder
        self.config_path = self.folder / "config.yaml"
        self.config = derp.util.load_config(self.config_path)
        self.marker_colors = [(128, 128, 128), (0, 0, 255), (0, 128, 255), (0, 255, 0)]
        self.topics = {}
        for topic in derp.util.TOPICS:
            if not derp.util.topic_exists(self.folder, topic):
                continue
            topic_fd = derp.util.topic_file_reader(self.folder, topic)
            self.topics[topic] = [
                msg for msg in derp.util.TOPICS[topic].read_multiple(topic_fd)
            ]
            topic_fd.close()
            print(topic, len(self.topics[topic]), sep="\t")
        self.n_frames = len(self.topics["camera"])
        control_times = [msg.timestampPublished for msg in self.topics["control"]]
        camera_times = [msg.timestampPublished for msg in self.topics["camera"]]
        speeds = [msg.speed for msg in self.topics["control"]]
        steers = [msg.steer for msg in self.topics["control"]]
        self.speeds = derp.util.latest_messages(camera_times, control_times, speeds)
        self.steers = derp.util.latest_messages(camera_times, control_times, steers)
        self.frame_id = 0
        self.seek(self.frame_id)
        self.f_h = self.frame.shape[0]
        self.f_w = self.frame.shape[1]
        self.bhh = 100
        self.fwi = np.arange(self.f_w)
        window_shape = list(self.frame.shape)
        window_shape[0] += self.bhh * 2 + 2
        self.window = np.zeros(window_shape, dtype=np.uint8)
        self.paused = True
        self.show = False
        self.marker = "unknown"
        self.label_bar = np.ones((self.f_w, 3), dtype=np.uint8) * (128, 128, 128,)
        if "label" in self.topics and len(self.topics["label"]) >= self.n_frames:
            self.labels = [str(msg.quality) for msg in self.topics["label"]]
        else:
            self.labels = ["unknown" for _ in range(self.n_frames)]
        for i, l in enumerate(self.labels):
            self.update_label(i, i, l)

    def __del__(self):
        cv2.destroyAllWindows()

    def update_label(self, first_index, last_index, marker="unknown"):
        if marker == "unknown":
            return False
        for index in range(first_index, last_index + 1):
            self.labels[index] = marker
        beg_pos = self.frame_pos(first_index)
        end_pos = self.frame_pos(last_index + (self.n_frames < len(self.label_bar)))
        bar_color = self.marker_colors[
            derp.util.TOPICS["label"].QualityEnum.__dict__[marker]
        ]
        self.label_bar[beg_pos : end_pos + 1] = bar_color

    def seek(self, frame_id=None):
        if frame_id is None:
            frame_id = self.frame_id + 1
        if frame_id < 0:
            frame_id = 0
            self.paused = True
        if frame_id >= self.n_frames:
            frame_id = self.n_frames - 1
            self.paused = True
        jpg_arr = np.frombuffer(self.topics["camera"][self.frame_id].jpg, np.uint8)
        frame = cv2.imdecode(jpg_arr, cv2.IMREAD_COLOR)
        self.frame = cv2.resize(
            frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA
        )
        self.frame_id = frame_id
        return True

    def draw_bar_time_marker(self):
        bar_color = self.marker_colors[
            derp.util.TOPICS["label"].QualityEnum.__dict__[self.marker]
        ]
        self.window[self.f_h :, self.frame_pos(self.frame_id), :] = bar_color

    def draw_bar_blank(self):
        self.window[self.f_h :, :, :] = 0

    def draw_bar_status(self):
        self.window[
            self.f_h : self.f_h + int(self.bhh // 10), self.fwi, :,
        ] = self.label_bar

    def draw_bar_zeroline(self):
        y_val = self.f_h + self.bhh
        self.window[y_val, self.fwi, :] = (192, 192, 192)

    def draw_horizon_bar(self):
        percent = self.config["camera"]["pitch"] / self.config["camera"]["vfov"] + 0.5
        self.window[int(self.f_h * percent), :, :] = (255, 0, 255)

    def draw_line(self, data_vector, color):
        bar = derp.util.interpolate(data_vector, self.f_w, self.bhh)
        data_jump_locs = []
        for loc in np.where(abs(bar[:-1] - bar[1:]) >= 2)[0]:
            rr, cc, _ = line_aa(
                bar[loc] + self.f_h + self.bhh,
                loc,
                bar[loc + 1] + self.f_h + self.bhh,
                loc + 1,
            )
            data_jump_locs.append((rr, cc))
        self.window[bar + self.f_h + self.bhh, self.fwi, :] = color
        for rr, cc in data_jump_locs:
            self.window[rr, cc, :] = color

    def display(self):
        self.window[: self.frame.shape[0], :, :] = self.frame
        self.draw_horizon_bar()
        self.draw_bar_blank()
        self.draw_bar_time_marker()
        self.draw_bar_zeroline()
        self.draw_bar_status()
        self.draw_line(data_vector=self.speeds, color=(255, 255, 0))
        self.draw_line(data_vector=self.steers, color=(0, 255, 0))
        cv2.imshow("Labeler %s" % self.folder, self.window)

    def save_labels(self):
        with derp.util.topic_file_writer(self.folder, "label") as label_fd:
            for label_i, label in enumerate(self.labels):
                msg = derp.util.TOPICS["label"].new_message(
                    timestampCreated=derp.util.get_timestamp(),
                    timestampPublished=self.topics["camera"][
                        label_i
                    ].timestampPublished,
                    timestampWritten=derp.util.get_timestamp(),
                    quality=label,
                )
                msg.write(label_fd)
        print("Saved labels in", self.folder)

    def handle_keyboard_input(self):
        key = cv2.waitKey(1) & 0xFF
        if key == 255:
            return True
        if key == 27:
            return False  # ESC
        elif key == ord(" "):
            self.paused = not self.paused
        elif key == ord("g"):
            self.marker = "good"
        elif key == ord("r"):
            self.marker = "risky"
        elif key == ord("t"):
            self.marker = "trash"
        elif key == ord("c"):
            self.marker = "unknown"
        elif key == ord("s"):
            self.save_labels()
        elif key == 82:
            self.seek(self.frame_id + 10)  # up
        elif key == 84:
            self.seek(self.frame_id - 10)  # down
        elif key == 81:
            self.seek(self.frame_id - 1)  # left
        elif key == 83:
            self.seek(self.frame_id + 1)  # right
        elif key == 85:
            self.config["camera"]["pitch"] -= 0.1  # page up
        elif key == 86:
            self.config["camera"]["pitch"] += 0.1  # page down
        elif ord("1") <= key <= ord("6"):
            self.seek(int(self.n_frames * (key - ord("0") - 1) / 5))
        elif key != 255:
            print("Unknown key press: [%s]" % key)
        self.show = True
        return True

    def frame_pos(self, frame_id):
        return min(self.f_w - 1, int(frame_id / self.n_frames * self.f_w))

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
            time.sleep(0.01)


def main():
    print("Arrow keys to navigate, 1-6 to teleport, s to save, ESC to quit")
    print(
        "To assign a label state as you play: g=good, r=risky, t=trash, and c to clear marker"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="recording path location")
    parser.add_argument("--scale", type=float, default=1.0, help="frame rescale ratio")
    args = parser.parse_args()
    labeler = Labeler(folder=args.path, scale=args.scale)
    labeler.run()


if __name__ == "__main__":
    main()
