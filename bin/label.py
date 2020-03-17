#!/usr/bin/env python3
"""OpenCV-based frame viewer that replays recordings and assign time-based labels"""
import argparse
from pathlib import Path
import time
import cv2
import numpy as np
import derp.util


class Labeler:
    """OpenCV-based frame viewer that replays recordings and assign time-based labels"""

    def __init__(self, folder, scale=1, bhh=40):
        """Load the topics and existing labels from the folder, scaling up the frame"""
        self.folder = folder
        self.scale = scale
        self.bhh = bhh
        self.config_changed = False
        self.quality = None
        self.config_path = self.folder / "config.yaml"
        self.window_name = "Labeler %s" % self.folder
        self.config = derp.util.load_config(self.config_path)
        self.quality_colors = [(0, 0, 255), (0, 128, 255), (0, 255, 0)]
        self.topics = derp.util.load_topics(folder)
        self.frame_id = 0
        self.n_frames = len(self.topics["camera"])
        self.seek(self.frame_id)
        self.f_h = self.frame.shape[0]
        self.f_w = self.frame.shape[1]
        self.l_h = int(self.bhh // 5)
        self.window = np.zeros(
            [self.f_h + self.bhh * 2 + self.l_h + 2, self.f_w, 3], dtype=np.uint8
        )
        self.paused = True
        self.show = False

        # Prepare labels
        self.autonomous_bar = np.ones((self.f_w, 3), dtype=np.uint8) * (255, 255, 255)
        self.quality_bar = np.ones((self.f_w, 3), dtype=np.uint8) * (128, 128, 128)
        if "quality" in self.topics and len(self.topics["quality"]) >= self.n_frames:
            self.qualities = [str(msg.quality) for msg in self.topics["quality"]]
        else:
            self.qualities = ["junk" for _ in range(self.n_frames)]
        for i, quality in enumerate(self.qualities):
            self.update_quality(i, i, quality)

        # Prepare state messages
        self.camera_times = [msg.publishNS for msg in self.topics["camera"]]
        self.camera_autos = []
        auto = False
        for timestamp, topic, msg in derp.util.replay(self.topics):
            if topic == 'controller':
                auto = msg.isAutonomous
            elif topic == 'camera':
                self.camera_autos.append(auto)

        actions = derp.util.extract_car_actions(self.topics)
        self.camera_speeds = derp.util.extract_latest(self.camera_times,
                                                      actions[:, 0], actions[:, 1])
        self.camera_steers = derp.util.extract_latest(self.camera_times,
                                                      actions[:, 0], actions[:, 2])
        window_Xs = np.linspace(self.camera_times[0], self.camera_times[-1], self.f_w)
        self.window_speeds = np.array(np.interp(window_Xs, self.camera_times, self.camera_speeds)
                                      * -self.bhh, dtype=np.int)
        self.window_steers = np.array(np.interp(window_Xs, self.camera_times, self.camera_steers)
                                      * -self.bhh, dtype=np.int)
        self.autonomous_bar *= np.array(np.interp(window_Xs, self.camera_times,
                                                  self.camera_autos), dtype=np.uint8)[:, None]
        self.window_steers[self.window_steers > self.bhh] = self.bhh
        self.window_steers[self.window_steers < -self.bhh] = -self.bhh

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.click_handler)
        # Print some statistics
        duration = (self.camera_times[-1] - self.camera_times[0]) / 1e9
        fps = (len(self.camera_times) - 1) / duration
        print("Duration of %.0f seconds at %.0f fps" % (duration, fps))

    def __del__(self):
        """Deconstructor to close window"""
        cv2.destroyAllWindows()

    def click_handler(self, event, x, y, flags, param):
        """ Handle clicks on the window """
        if event == cv2.EVENT_LBUTTONDOWN:
            if y > self.f_h:
                frame_id = int((x / self.f_w) * self.n_frames)
                self.seek(frame_id)
                self.show = True

    def update_quality(self, first_index, last_index, quality=None):
        """Update the label bar to the given quality"""
        if quality is None:
            return False
        first_index, last_index = min(first_index, last_index), max(first_index, last_index)
        for index in range(first_index, last_index + 1):
            self.qualities[index] = quality
        beg_pos = self.frame_pos(first_index)
        end_pos = self.frame_pos(last_index + (self.n_frames < len(self.quality_bar)))
        self.quality_bar[beg_pos : end_pos + 1] = self.bar_color(quality)
        return True

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
        self.update_quality(self.frame_id, frame_id, self.quality)
        self.frame = cv2.resize(
            derp.util.decode_jpg(self.topics["camera"][frame_id].jpg),
            None,
            fx=self.scale,
            fy=self.scale,
            interpolation=cv2.INTER_AREA,
        )
        self.frame_id = frame_id
        return True

    def bar_color(self, quality):
        """Figure out the color for the given quality"""
        if quality is None:
            return (128, 128, 128)
        return self.quality_colors[derp.util.TOPICS["quality"].QualityEnum.__dict__[quality]]

    def display(self):
        """Blit all the status on the screen"""
        self.window[: self.frame.shape[0], :, :] = self.frame
        horizon_percent = self.config["camera"]["pitch"] / self.config["camera"]["vfov"] + 0.5
        # Horizon line
        self.window[int(self.f_h * horizon_percent), :, :] = (255, 0, 255)
        # Clear status buffer
        self.window[self.f_h :, :, :] = 0
        # Draw label bar
        self.window[self.f_h : self.f_h + self.l_h // 2, :, :] = self.autonomous_bar
        self.window[self.f_h + self.l_h // 2 : self.f_h + self.l_h, :, :] = self.quality_bar
        # Draw current timestamp vertical line
        current_x = self.frame_pos(self.frame_id)
        self.window[self.f_h + self.l_h :, current_x, :] = self.bar_color(self.quality)
        # Draw zero line
        self.window[self.f_h + self.l_h + self.bhh, :, :] = (96, 96, 96)
        offset = self.f_h + self.bhh + self.l_h
        self.window[self.window_speeds + offset, np.arange(self.f_w), :] = (255, 64, 255)
        self.window[self.window_steers + offset, np.arange(self.f_w), :] = (64, 255, 255)
        text = "%05i %07.3f %06.3f %06.3f" % (self.frame_id,
                                            (self.camera_times[self.frame_id] / 1E9) % 100,
                                            self.camera_steers[self.frame_id],
                                            self.camera_speeds[self.frame_id])
        font = cv2.FONT_HERSHEY_SIMPLEX
        pink = (255, 128, 255)
        offset = (0, int(self.scale * 30))
        cv2.putText(self.window, text, offset, font, self.scale, pink, 1, cv2.LINE_AA)
        cv2.imshow(self.window_name, self.window)

    def save_labels(self):
        """Write all of our labels to the folder as messages"""
        with derp.util.topic_file_writer(self.folder, "quality") as quality_fd:
            for quality_i, quality in enumerate(self.qualities):
                msg = derp.util.TOPICS["quality"].new_message(
                    createNS=derp.util.get_timestamp(),
                    publishNS=self.topics["camera"][quality_i].publishNS - 1,
                    writeNS=derp.util.get_timestamp(),
                    quality=quality,
                )
                msg.write(quality_fd)
        print("Saved quality labels in", self.folder)
        if self.config_changed:
            derp.util.dump_config(self.config, self.config_path)
            print("Saved changes to config")

    def handle_keyboard_input(self):
        """Fetch a new keyboard input if one exists"""
        key = cv2.waitKey(1) & 0xFF
        if key == 255:
            return True
        if key == 27:
            return False  # ESC
        if key == ord(" "):
            self.paused = not self.paused
        elif key == ord("g"):
            self.quality = "good"
        elif key == ord("r"):
            self.quality = "risk"
        elif key == ord("t"):
            self.quality = "junk"
        elif key == ord("c"):
            self.quality = None
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
            self.config_changed = True
        elif key == 86:
            self.config["camera"]["pitch"] += 0.1  # page down
            self.config_changed = True
        elif ord("1") <= key <= ord("5"):
            self.seek(int(self.n_frames * (key - ord("0") - 1) / 4))
        elif key != 255:
            print("Unknown key press: [%s]" % key)
        self.show = True
        return True

    def frame_pos(self, frame_id):
        """Position of current camera frame on the horizontal status bars"""
        return min(self.f_w - 1, int(frame_id / self.n_frames * self.f_w))

    def run(self):
        """Run the labeling program in a forever loop until the user quits"""
        self.display()
        while True:
            if not self.paused:
                self.show = self.seek()
            if self.show:
                self.display()
                self.show = False
            if not self.handle_keyboard_input():
                break
            time.sleep(0.01)


def main():
    """Initialize the labeler based on user args and run it"""
    print(
        """
This labeling tool interpolates the data based on camera frames and then lets you label each.
To exit press ESCAPE
To save press s
To navigate between frames:
    Left/Right: move in 1 frame increments
    Up/Down: move in 10 frame increments
    1: goes to beginning
    2: goes to 25% in
    3: goes to 50% in
    4  goes to 25% in
    5: goes to end
To adjust horizon line press PAGE_UP or PAGE_DOWN
To change the quality label of this frame
    g: good (use for training)
    r: risk (advanced situation not suitable for classic training)
    t: junk (don't use this part of the video, aka trash)
    c: clear, as in don't change the quality label
"""
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", type=Path, nargs="*", metavar="N", help="recording path location")
    parser.add_argument("--scale", type=float, default=1.0, help="frame rescale ratio")
    args = parser.parse_args()
    if not args.paths:
        recordings = (derp.util.DERP_ROOT / "recordings").glob("recording-*")
        args.paths = [r for r in recordings if not (r / "quality.bin").exists()]
    for path in args.paths:
        print("Labeling", path)
        labeler = Labeler(folder=path, scale=args.scale)
        labeler.run()


if __name__ == "__main__":
    main()
