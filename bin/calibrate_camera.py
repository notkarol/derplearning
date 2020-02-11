#!/usr/bin/env python3
"""
A tool to calibrate the connected camera using a checkerboard printout.
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
from derp.camera import Camera
import derp.util


def live_calibrate(camera, pattern_size, n_matches_needed=10):
    """ Find calibration parameters as the user moves a checkerboard in front of the camera """
    print("Looking for %s checkerboard" % (pattern_size,))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    example_3d = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    example_3d[:, :2] = np.mgrid[0 : pattern_size[1], 0 : pattern_size[0]].T.reshape(
        -1, 2
    )
    points_3d = []
    points_2d = []
    while len(points_3d) < n_matches_needed:
        ret, frame = camera.cap.read()
        assert ret
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findCirclesGrid(
            gray_frame, pattern_size, flags=cv2.CALIB_CB_ASYMMETRIC_GRID
        )
        if ret:
            points_3d.append(example_3d.copy())
            points_2d.append(corners)
            print("Found calibration %i of %i" % (len(points_3d), n_matches_needed))
            frame = cv2.drawChessboardCorners(frame, pattern_size, corners, ret)
        cv2.imshow("camera", frame)
        cv2.waitKey(10)
    ret, camera_matrix, distortion_coefficients, _, _ = cv2.calibrateCamera(
        points_3d, points_2d, gray_frame.shape[::-1], None, None
    )
    assert ret
    return camera_matrix, distortion_coefficients


def live_undistort(camera, camera_matrix, distortion_coefficients):
    """ Using a given calibration matrix, display the distorted, undistorted, and cropped frame"""
    scaled_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, distortion_coefficients, camera.size, 1, camera.size
    )
    while True:
        ret, frame = camera.cap.read()
        assert ret
        distorted_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        undistorted_frame = cv2.undistort(
            distorted_frame,
            camera_matrix,
            distortion_coefficients,
            None,
            scaled_camera_matrix,
        )
        roi_x, roi_y, roi_w, roi_h = roi
        cropped_frame = undistorted_frame[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]
        cv2.imshow("distorted", distorted_frame)
        cv2.imshow("undistorted", undistorted_frame)
        cv2.imshow("cropped", cropped_frame)
        cv2.waitKey(10)


def main():
    """
    Calibrate the live camera and optionally do a live display of the results
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("config", type=Path, help="camera config path")
    parser.add_argument("--pattern_width", type=int, default=5)
    parser.add_argument("--pattern_height", type=int, default=10)
    parser.add_argument("--view", action="store_true")
    args = parser.parse_args()

    config = {"camera": derp.util.load_config(args.config)}
    camera = Camera(config)
    pattern_size = (args.pattern_height, args.pattern_width)

    camera_matrix, distortion_coefficients = live_calibrate(camera, pattern_size)

    if args.view:
        live_undistort(camera, camera_matrix, distortion_coefficients)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
