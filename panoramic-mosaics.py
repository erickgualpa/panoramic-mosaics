import cv2
import os
import numpy as np
from functions import *

# 1 - LOAD IMAGES ############################################################################
img_path = "./ee/"

working_imgs = []
for root, dirs, files in os.walk(img_path):
    for filename in files[:3]:
        working_imgs.append(loadImage(img_path + filename))

img1 = resizeImage(working_imgs[0], 15)
img2 = resizeImage(working_imgs[1], 15)
img3 = resizeImage(working_imgs[2], 15)
##############################################################################################

# 2 - REMARKABLE POINTS SELECTION ############################################################
pointsA = []
pointsBtoA = []
pointsBtoC = []
pointsC = []

def getPointA(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pointsA.append((x,y))
        if len(pointsA) >= 4: cv2.destroyWindow('image')

def getPointBtoA(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pointsBtoA.append((x,y))
        if len(pointsBtoA) >= 4: cv2.destroyWindow('image')

def getPointBtoC(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pointsBtoC.append((x,y))
        if len(pointsBtoC) >= 4: cv2.destroyWindow('image')

def getPointC(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pointsC.append((x,y))
        if len(pointsC) >= 4: cv2.destroyWindow('image')

param = [img1]
imgAB = writeMessageOnImage(img1.copy(), "SELECT the POINTS in the LEFT IMAGE OVER THE CENTER ONE")
cv2.namedWindow('image')
cv2.setMouseCallback('image',getPointA)
cv2.imshow('image',imgAB) 
cv2.waitKey(0) 

param = [img2]
imgBA = writeMessageOnImage(img2.copy(), "SELECT the POINTS in the CENTER IMAGE OVER THE LEFT ONE ")
cv2.namedWindow('image')
cv2.setMouseCallback('image',getPointBtoA)
cv2.imshow('image',imgBA) 
cv2.waitKey(0) 

param = [img2]
imgBC = writeMessageOnImage(img2.copy(), "SELECT the POINTS in the CENTER IMAGE OVER THE RIGHT ONE")
cv2.namedWindow('image')
cv2.setMouseCallback('image',getPointBtoC)
cv2.imshow('image',imgBC) 
cv2.waitKey(0) 

param = [img3]
imgCB = writeMessageOnImage(img3.copy(), "SELECT the POINTS in the RIGHT IMAGE OVER THE CENTER ONE")
cv2.namedWindow('image')
cv2.setMouseCallback('image',getPointC)
cv2.imshow('image',imgCB) 
cv2.waitKey(0)
##############################################################################################

# 3 - (DLT) HOMOGRAPHY CALCULATION ###########################################################
print("\n-- IMAGE A POINTS: ", pointsA)
print("\n-- IMAGE B POINTS related to A (center): ", pointsBtoA)
print("\n-- IMAGE B POINTS related to C (center): ", pointsBtoC)
print("\n-- IMAGE C POINTS: ", pointsC)

Ha, boundingRect_A = getHomographyMatrixByDLT(img1, pointsA, pointsBtoA)
Hc, boundingRect_C = getHomographyMatrixByDLT(img1, pointsC, pointsBtoC)

canvasShape, startBlendingFrom = getCanvasShapeFromImgsBoundingRect(boundingRect_A, img2.shape[:2], boundingRect_C)

imA_H = cv2.warpPerspective(img1, Ha, (boundingRect_A[2], boundingRect_A[3]))
imC_H = cv2.warpPerspective(img3, Hc, (boundingRect_C[2], boundingRect_C[3]))

ret, imA_H_mask = cv2.threshold(cv2.cvtColor(imA_H, cv2.COLOR_BGR2GRAY), 2, 255, cv2.THRESH_BINARY)
ret, imC_H_mask = cv2.threshold(cv2.cvtColor(imC_H, cv2.COLOR_BGR2GRAY), 2, 255, cv2.THRESH_BINARY)

#showImage(imA_H)
#showImage(imC_H)
##############################################################################################

# 4 - IMAGE BLENDING (PANORAMIC IMAGE MAKING) ################################################
canvas = np.zeros((canvasShape[1],canvasShape[0],3), np.uint8)
canvas = blendImages(	Ha, 
						Hc, 
						startBlendingFrom, 
						imA_H, 
						img2, 
						imC_H, 
						canvas, 
						pointsA, 
						pointsBtoA, 
						pointsBtoC, 
						pointsC, 
						imA_H_mask, 
						imC_H_mask)
showImage(canvas)
cv2.imwrite('./results/canvas.jpg', canvas)
##############################################################################################
