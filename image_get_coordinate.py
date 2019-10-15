# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 21:47:04 2019

@author: avon
"""

# coding: utf-8
import cv2
import numpy as np
import glob

img = cv2.imread('test_store.jpg')
img = cv2.resize(img,(640,480))

def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        print(xy)
        cv2.circle(img, (x, y), 1, (255, 100, 0), thickness = 3)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0,0,255), thickness = 1)
        cv2.imshow("image", img)

cv2.namedWindow("image")
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)

cv2.imshow('image', img)

while (1):
    cv2.imshow('image', img)
    
    if cv2.waitKey(1)&0xFF == ord('q'):#按q键退出
        break
cv2.destroyAllWindows()
