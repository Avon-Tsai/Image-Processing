# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 10:26:39 2019
@author: A40455
"""

import cv2
import numpy as np
import os
import glob

CHECKERBOARD = (6,9)
subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

'''
criteria: 這是迭代終止準則。當滿足這個準則時，算法迭代停止。
實際上，它應該是一個3個參數的元組：(type, max_iter, epsilon) 
    *3.a - 終止準則的類型： 
           有3個標誌如下：        
             cv2.TERM_CRITERIA_EPS -如果滿足了指定準確度，epsilon就停止算法迭代。        
             cv2.TERM_CRITERIA_MAX_ITER -在指定次數的迭代後就停止算法。        
             cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER - 當任何上面的條件滿足就停止迭代    
    *3.b - max_iter - 指定最大的迭代次數，整數    
    *3.c - epsilon - 需要的準確度
'''

calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW

objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
_img_shape = None
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('*.jpg')
for fname in images:
    img = cv2.imread(fname)    
    img = cv2.resize(img,(1280,960) )
    
    if _img_shape == None:
        _img_shape = img.shape[:2]
    else:
        assert _img_shape == img.shape[:2], "All images must share the same size."
         
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
        imgpoints.append(corners)
         
N_OK = len(objpoints)
K = np.zeros((3, 3))
D = np.zeros((4, 1)) # 在fisheye模型中，畸變係數主要是一個四維的向量 {k1,k2,k3,k4}
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
rms, _, _, _, _ = \
    cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        K,
        D,
        rvecs,
        tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )
 
# 攝像機的內參矩陣 K，和畸變係數 D
# 旋轉參數 rvecs，平移參數 tvecs
DIM = _img_shape[::-1]
K = np.array(K.tolist())
D = np.array(D.tolist())

print("Found " + str(N_OK) + " valid images for calibration")
print("DIM=" + str(_img_shape[::-1]))
print("K=np.array(" + str(K.tolist()) + ")")
print("D=np.array(" + str(D.tolist()) + ")")

'''
DIM， K， D是固定不變的，
因此，map1和map2也是不變的， 
當你有大量的數據需要矯正時， 
應當避免map1和map2的重複計算， 
只需要計算 校正remap 即可。
'''
def undistort(img_path):
    DIM = (640*2, 480*2)
    K=np.array([[674.2470124768595, 0.0, 644.2978957009802], [0.0, 673.4913940419676, 478.5309482863649], [0.0, 0.0, 1.0]])
    D=np.array([[0.007154823478563458], [-0.26904311051430424], [0.44441034579115585], [-0.22409297163013786]])
    
    img = cv2.imread(img_path)
    h,w = img.shape[:2]
    # initUndistortRectifyMap 用於去除鏡頭畸變的圖像拉伸
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    new_img_path = str(img_path).replace(".jpg","_calibrate.jpg")
    cv2.imwrite(new_img_path,undistorted_img)
    cv2.imshow("undistorted", undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    
    images = 'test.jpg'
    undistort(images)
