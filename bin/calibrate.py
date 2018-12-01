import cv2
import numpy as np

cap = cv2.VideoCapture(0)
pattern_size = (4, 11)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((44, 3), np.float32)
objp[:,:2] = np.mgrid[0:4,0:11].T.reshape(-1,2)
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

while len(objpoints) < 10:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('camera', gray)
    cv2.waitKey(10)
    ret, corners = cv2.findCirclesGrid(gray, pattern_size, flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
    if ret:
        objpoints.append(objp.copy())
        imgpoints.append(corners)
        frame = cv2.drawChessboardCorners(frame, pattern_size, corners, ret)
        cv2.imshow('success', frame)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                   gray.shape[::-1], None, None)
    
w, h = gray.shape[::-1]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
while True:
    ret, frame = cap.read()
    distorted = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    undistorted = cv2.undistort(distorted, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    cropped = undistorted[y:y+h, x:x+w]
    cv2.imshow('distorted', distorted)
    cv2.imshow('undistorted', undistorted)
    cv2.imshow('cropped', cropped)
    cv2.waitKey(10)
cv2.destroyAllWindows()
