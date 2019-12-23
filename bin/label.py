#!/usr/bin/env python3
"""OpenCV-based frame viewer that replays recordings and assign time-based labels"""
import argparse
from pathlib import Path
import time
import cv2
import numpy as np
from skimage.draw import line_aa
import derp.util


class Labeler:
    """OpenCV-based frame viewer that replays recordings and assign time-based labels"""

    def __init__(self, folder, scale=1, bhh=100):
        """Load the topics and existing labels from the folder, scaling up the frame"""
        self.scale = scale
        self.folder = folder
        self.config_path = self.folder / "config.yaml"
        self.config = derp.util.load_config(self.config_path)
        self.quality_colors = [(128, 128, 128), (0, 0, 255), (0, 128, 255), (0, 255, 0)]
        self.topics = derp.util.load_topics(folder)
        self.frame_id = 0
        self.n_frames = len(self.topics["camera"])
        self.seek(self.frame_id)
        self.f_h = self.frame.shape[0]
        self.f_w = self.frame.shape[1]
        self.bhh = bhh
        self.window = np.zeros([self.f_h + self.bhh * 2 + 2, self.f_w, 3], dtype=np.uint8)
        self.paused = True
        self.show = False
        self.quality = "unknown"

        # Prepare labels
        self.label_bar = np.ones((self.f_w, 3), dtype=np.uint8) * (128, 128, 128,)
        if "label" in self.topics and len(self.topics["label"]) >= self.n_frames:
            self.labels = [str(msg.quality) for msg in self.topics["label"]]
        else:
            self.labels = ["unknown" for _ in range(self.n_frames)]
        for i, l in enumerate(self.labels):
            self.update_label(i, i, l)

        # Prepare state messages
        control_times = [msg.timePublished for msg in self.topics["control"]]
        camera_times = [msg.timePublished for msg in self.topics["camera"]]
        speeds = [msg.speed for msg in self.topics["control"]]
        steers = [msg.steer for msg in self.topics["control"]]
        self.speeds = derp.util.latest_messages(camera_times, control_times, speeds)
        self.steers = derp.util.latest_messages(camera_times, control_times, steers)

    def __del__(self):
        """Deconstructor to close window"""
        cv2.destroyAllWindows()

    def update_label(self, first_index, last_index, quality="unknown"):
        """Update the label bar to the given quality"""
        if quality == "unknown":
            return False
        for index in range(first_index, last_index + 1):
            self.labels[index] = quality
        beg_pos = self.frame_pos(first_index)
        end_pos = self.frame_pos(last_index + (self.n_frames < len(self.label_bar)))
        self.label_bar[beg_pos : end_pos + 1] = self.bar_color(quality)

    def seek(self, frame_id=None):
        """Update the current frame to the given frame_id, otherwise advances by 1 frame"""
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

    def bar_color(self, quality):
        """Figure out the color for the given quality"""
        return self.quality_colors[derp.util.TOPICS["label"].QualityEnum.__dict__[quality]]

    def draw_line(self, values, color):
        """Draw the line on the screen"""
        interpolated_values = derp.util.interpolate(values, self.f_w, self.bhh)
        self.window[interpolated_values + self.f_h + self.bhh, np.arange(self.f_w), :] = color

    def display(self):
        """Blit all the status on the screen"""
        self.window[: self.frame.shape[0], :, :] = self.frame
        horizon_percent = self.config["camera"]["pitch"] / self.config["camera"]["vfov"] + 0.5
        # Horizon line
        self.window[int(self.f_h * horizon_percent), :, :] = (255, 0, 255)
        # Clear status buffer
        self.window[self.f_h :, :, :] = 0
        # Draw label bar
        self.window[self.f_h : self.f_h + int(self.bhh // 10), :, :] = self.label_bar
        # Draw current timestamp vertical line
        self.window[self.f_h :, self.frame_pos(self.frame_id), :] = self.bar_color(self.quality)
        # Draw zero line
        self.window[self.f_h + self.bhh, :, :] = (192, 192, 192)
        self.draw_line(self.speeds, color=(0, 0, 255))
        self.draw_line(self.steers, color=(255, 128, 0))
        cv2.imshow("Labeler %s" % self.folder, self.window)

    def save_labels(self):
        """Write all of our labels to the folder as messages"""
        with derp.util.topic_file_writer(self.folder, "label") as label_fd:
            for label_i, label in enumerate(self.labels):
                msg = derp.util.TOPICS["label"].new_message(
                    timeCreated=derp.util.get_timestamp(),
                    timePublished=self.topics["camera"][label_i].timePublished - 1,
                    timeWritten=derp.util.get_timestamp(),
                    quality=label,
                )
                msg.write(label_fd)
        print("Saved labels in", self.folder)

    def handle_keyboard_input(self):
        """Fetch a new keyboard input if one exists"""
        key = cv2.waitKey(1) & 0xFF
        if key == 255:
            return True
        if key == 27:
            return False  # ESC
        elif key == ord(" "):
            self.paused = not self.paused
        elif key == ord("g"):
            self.quality = "good"
        elif key == ord("r"):
            self.quality = "risky"
        elif key == ord("t"):
            self.quality = "trash"
        elif key == ord("c"):
            self.quality = "unknown"
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
                self.update_label(self.frame_id, self.frame_id, self.quality)
                self.show = self.seek()
            if self.show:
                self.display()
                self.show = False
            if not self.handle_keyboard_input():
                break
            time.sleep(0.01)


def main():
    print("Arrow keys to navigate, 1-6 to teleport, s to save, ESC to quit")
    print("To assign a label state as you play: g=good, r=risky, t=trash, and c to clear quality")
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="recording path location")
    parser.add_argument("--scale", type=float, default=1.0, help="frame rescale ratio")
    args = parser.parse_args()
    labeler = Labeler(folder=args.path, scale=args.scale)
    labeler.run()


if __name__ == "__main__":
    main()
