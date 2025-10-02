import cv2 as cv
import numpy as np
import pickle
import glob



with open('cameraMatrix.pkl', 'rb') as f:
    camera_matrix = pickle.load(f)

with open('dist.pkl', 'rb') as f:
    dist_coeffs = pickle.load(f)

print("camera_matrix", camera_matrix)
print("dist_coeffs", dist_coeffs)


pattern_size = (7, 7)
square_size = 0.0027


################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

chessboardSize = (9,6)
frameSize = (640,480)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

size_of_chessboard_squares_mm = 2.7
objp = objp * size_of_chessboard_squares_mm


# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


images = glob.glob('images/*.png')
# images = 'images/*.png'

for image in images:

    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

    # If found, add object points, image points (after refining them)
    if ret == True:

        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(1000)


cv.destroyAllWindows()


ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)
print("ret", ret)
print("cameraMatrix", cameraMatrix)
print("dist", dist)
print("rvecs", rvecs)
print("tvecs", tvecs)













#
# obj_points = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
# obj_points[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
# obj_points *= square_size
# image = cv2.imread("images/img0.png")
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
#
# if ret:
#     K = camera_matrix
#     dist_coeffs = dist_coeffs
#
#     # 📌 Solve PnP to find rotation & translation vectors
#     success, rvec, tvec = cv2.solvePnP(obj_points, corners, K, dist_coeffs)
#
#     if success:
#         # 📌 Convert rotation vector to rotation matrix
#         R, _ = cv2.Rodrigues(rvec)
#
#         # 📌 Construct the extrinsic matrix [R | t]
#         extrinsic_matrix = np.hstack((R, tvec))
#
#         # _, rVec, tVec = cv2.solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs)
#         # Rt = cv2.Rodrigues(rvec)
#         # R = Rt.transpose()
#         # pos = -R * tVec
#
#         # 📌 Print results
#         print("Rotation Matrix (R):\n", R)
#         print("\nTranslation Vector (t):\n", tvec)
#         print("\nExtrinsic Matrix [R | t]:\n", extrinsic_matrix)
#     else:
#         print("❌ Failed to solve PnP.")
# else:
#     print("❌ Chessboard corners not detected.")
#
# # 📌 Display detected corners
# cv2.drawChessboardCorners(image, pattern_size, corners, ret)
# cv2.imshow("Chessboard", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
