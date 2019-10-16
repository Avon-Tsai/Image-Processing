# -*- coding: utf-8 -*-
"""
Created on Thu May 16 17:44:50 2019

@author: A40455
"""
import cv2
import glob
import ast

def calculateIoU(candidateBound, groundTruthBound):
    cx1 = candidateBound[0]
    cy1 = candidateBound[1]
    cx2 = candidateBound[2]
    cy2 = candidateBound[3]

    gx1 = groundTruthBound[0]
    gy1 = groundTruthBound[1]
    gx2 = groundTruthBound[2]
    gy2 = groundTruthBound[3]

    carea = (cx2 - cx1) * (cy2 - cy1) #C的面积
    garea = (gx2 - gx1) * (gy2 - gy1) #G的面积

    x1 = max(cx1, gx1)
    y1 = max(cy1, gy1)
    x2 = min(cx2, gx2)
    y2 = min(cy2, gy2)
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    area = w * h #C∩G的面积

    iou = area / (carea + garea - area)
    return iou


path = 'test.txt'
openpose_lines = open(path).readlines()
openpose_lines = [ast.literal_eval(openpose_lines[i]) for i in range(len(openpose_lines))]

path = 'yolo_result.txt'
yolo_lines = open(path).readlines()
yolo_lines = [ast.literal_eval(yolo_lines[i]) for i in range(len(yolo_lines))]

result = []
total_iou = 0
get_iou = 0
max_iou_5 = max_iou_6 =  max_iou_7 = max_iou_8 = max_iou_9 = max_iou_else = 0

for yolo in yolo_lines :
    idx = [i for i, s in enumerate(openpose_lines) if yolo[0].split('_')[2] in s[0].split('_')[2]]
    max_iou = 0
    for i in idx :
        max_iou = max(max_iou, calculateIoU(yolo[1], openpose_lines[i][1]) )
    
    if max_iou != 0 :
        get_iou += 1    
    total_iou += max_iou
    result.append([yolo[0], max_iou])
    #print([yolo[0], max_iou])
    
    if max_iou >= 0.5 and max_iou < 0.6 :
        max_iou_5 += 1
    elif max_iou >= 0.6 and max_iou < 0.7 :
        max_iou_6 += 1
    elif max_iou >= 0.7 and max_iou < 0.8 :
        max_iou_7 += 1
    elif max_iou >= 0.8 and max_iou < 0.9 :
        max_iou_8 += 1
    elif max_iou >= 0.9 and max_iou < 1 :
        max_iou_9 += 1
    else:
        max_iou_else += 1
    
    
print('Average IOU : ' + str(total_iou/get_iou))
print('max_iou_5 : ' + str(max_iou_5))
print('max_iou_6 : ' + str(max_iou_6))
print('max_iou_7 : ' + str(max_iou_7))
print('max_iou_8 : ' + str(max_iou_8))
print('max_iou_9 : ' + str(max_iou_9))
