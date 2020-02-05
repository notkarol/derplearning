#!/usr/bin/env python3
import cv2
import argparse
import numpy as np
from pathlib import Path
from derp.camera import Camera
import derp.util

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("config", type=Path, help='camera config path')
    parser.add_argument("--pattern_width", type=int, default=5)
    parser.add_argument("--pattern_height", type=int, default=10)
    args = parser.parse_args()
    
    config = {'camera': derp.util.load_config(args.config)}
    camera = Camera(config)

    pattern_size = (args.pattern_height, args.pattern_width)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((args.pattern_height * args.pattern_width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.pattern_height, 0:args.pattern_width].T.reshape(-1, 2)
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    while len(objpoints) < 10:
        ret, frame = camera.cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("camera", gray)
        cv2.waitKey(10)
        ret, corners = cv2.findCirclesGrid(gray, pattern_size, flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
        if ret:
            objpoints.append(objp.copy())
            imgpoints.append(corners)
            frame = cv2.drawChessboardCorners(frame, pattern_size, corners, ret)
            cv2.imshow("success", frame)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    w, h = gray.shape[::-1]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    while True:
        ret, frame = cap.read()
        distorted = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        undistorted = cv2.undistort(distorted, mtx, dist, None, newcameramtx)
        x, y, w, h = roi
        cropped = undistorted[y : y + h, x : x + w]
        cv2.imshow("distorted", distorted)
        cv2.imshow("undistorted", undistorted)
        cv2.imshow("cropped", cropped)
        cv2.waitKey(10)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
