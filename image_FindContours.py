# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 10:20:31 2019

@author: A40455
"""
import cv2
import numpy as np

img = cv2.imread('test_store.jpg')


imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


print(np.size(cnts))  #   得到该图中总的轮廓数量
print(cnts)   #  打印出第一个轮廓的所有点的坐标， 更改此处的0，为0--（总轮廓数-1），可打印出相应轮廓所有点的坐标
print(hierarchy) #打印出相应轮廓之间的关系

img = cv2.drawContours(img, cnts, -1, (0,255,0), 3)  #标记处编号为0的轮廓
cv2.imshow('drawimg',img)

cv2.waitKey(0)
cv2.destroyAllWindows()
