import cv2
import numpy as np

# FUNCIONES #########################
def showImage(img):
	if type(img) == str:
		img = cv2.imread(img_name, cv2.IMREAD_COLOR)
	cv2.imshow('image',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def loadImage(img_name):
	img = cv2.imread(img_name, cv2.IMREAD_COLOR)
	return img

def resizeImage(img, scale):
	scale_percent = scale
	width = int(img.shape[1] * scale_percent / 100)
	height = int(img.shape[0] * scale_percent / 100)
	dim = (width, height)
	
	resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	return resized

def superposeImage(img_dst, img_src, offset):
		img_dst[offset[1]:offset[1] + img_src.shape[0],offset[0]:offset[0]+img_src.shape[1]] = img_src
		return img_dst

def writeMessageOnImage(img, message):
	RED = (0,0,255)
	font = cv2.FONT_HERSHEY_SIMPLEX
	font_size = 0.5
	font_color = RED
	font_thickness = 1
	x,y = 10,40
	img_text = cv2.putText(img, message, (x,y), font, font_size, font_color, font_thickness, cv2.LINE_AA)

	return img_text
#####################################

# GEOMETRIC TRANSFORMATIONS #########
## Cáculo de la matriz de translación para evitar el cropping de la imagen ######
def getAlignmentMatrixAfterHomography(img, H):

	height, width = img.shape[:2]
	corners = np.array([
	    [0, 0],
	    [0, height - 1],
	    [width - 1, height - 1],
	    [width - 1, 0]
	])
	corners = cv2.perspectiveTransform(np.float32([corners]), H)[0]
	bx, by, bwidth, bheight = cv2.boundingRect(corners)
	tH = np.array([
	    [ 1, 0, -bx ],
	    [ 0, 1, -by ],
	    [ 0, 0,   1 ]
	])
	return tH, (bx, by, bwidth, bheight)

def getHomographyMatrixByDLT(img_source, pointsA, pointsB):
	A = []
	for i in range(len(pointsA)):
		x, y = pointsA[i][0], pointsA[i][1]
		u, v = pointsB[i][0], pointsB[i][1]
		A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
		A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
	A = np.asarray(A)
	U, S, Vh = np.linalg.svd(A)
	L = Vh[-1,:] / Vh[-1,-1]
	H = L.reshape(3, 3)


	tH, boundingRect = getAlignmentMatrixAfterHomography(img_source, H)
	return tH.dot(H), boundingRect

def getCanvasShapeFromImgsBoundingRect(boundingRect_A, imgBshape, boundingRect_C):
	MIN_X = 0
	MIN_Y = 1
	WIDTH = 2
	HEIGHT = 3

	startBlendingFrom = "L"

	print("\n-- DIMENSIONES DE LA IMAGEN A despues de la HOMOGRAFIA: minX, minY, width, height")
	print("\n- ", boundingRect_A)
	print("\n-- DIMENSIONES DE LA IMAGEN CENTRAL B: ")
	print("\n- ", imgBshape)
	print("\n-- DIMENSIONES DE LA IMAGEN C despues de la HOMOGRAFIA: minX, minY, width, height")
	print("\n- ", boundingRect_C)

	width = abs(boundingRect_A[MIN_X]) + abs(boundingRect_C[MIN_X]) + abs(boundingRect_C[WIDTH])

	#minHeight = min([boundingRect_A[MIN_Y], boundingRect_C[MIN_Y]])

	if boundingRect_A[MIN_Y] < boundingRect_C[MIN_Y]:
		minHeight = boundingRect_A[MIN_Y]
	else:
		minHeight = boundingRect_C[MIN_Y]
		startBlendingFrom = "R"


	heightA = boundingRect_A[MIN_Y] + boundingRect_A[HEIGHT]
	heightC = boundingRect_C[MIN_Y] + boundingRect_C[HEIGHT]
	maxHeight = max([heightA, heightC])

	height = abs(minHeight) + abs(maxHeight)

	print("\n-- DIMENSIONES DE LA IMAGEN PANORAMICA: ")
	print("\n- WIDTH: ", width)
	print("\n- HEIGHT: ", height)

	return (width, height), startBlendingFrom

def getPointCoordsInCanvas(point, imgShape, canvasShape):
	WIDTH = 1
	HEIGHT = 0

	axisX = canvasShape[WIDTH] - imgShape[WIDTH] + point[0]
	axisY = canvasShape[HEIGHT] - imgShape[HEIGHT] + point[1]

	return (axisX, axisY)

def alphaBlending(canvas, img_src, img_src_alpha, offset):

	roi = canvas[offset[1]:offset[1] + img_src.shape[0], offset[0]:offset[0] + img_src.shape[1]]
	mask_inv = cv2.bitwise_not(img_src_alpha)

	#showImage(mask_inv)

	canvas_background = cv2.bitwise_and(roi, roi, mask = mask_inv)
	img_src_foreground = cv2.bitwise_and(img_src, img_src, mask = img_src_alpha)

	dst = cv2.add(canvas_background, img_src_foreground)

	canvas[offset[1]:offset[1] + img_src.shape[0], offset[0]:offset[0] + img_src.shape[1]] = dst

	return canvas

def blendImages(Ha,	Hc, 
				startBlendingFrom, 
				imgA, imgB, imgC, 
				canvas, 
				pointsA, pointsBtoA, pointsBtoC, pointsC,
				imA_H_mask, imC_H_mask):

	offset = [0,0]

	print("\n-- Comenzando blending des de... : L = Izquierda, R = Derecha")
	print("\n- ", startBlendingFrom)
	
	if startBlendingFrom == "L": # Merging from left		
		canvas = superposeImage(canvas, imgA, offset)

		## Blending de la imagen A y B ############
		npPointsA = np.array(pointsA, np.float32)
		npPointsA = npPointsA.reshape(-1,1,2).astype(np.float32)
		dstPointsA = cv2.perspectiveTransform(npPointsA, Ha)

		#workingPointA = getPointCoordsInCanvas(dstPointsA[0][0], imgA.shape[:2], canvas.shape[:2]) 
		workingPointA = dstPointsA[0][0]
		workingPointB = pointsBtoA[0]
		offset_X = (workingPointA[0] - workingPointB[0]).astype(np.int64)
		offset_Y = (workingPointA[1] - workingPointB[1]).astype(np.int64)
		print("\n-- VALOR DEL OFFSET CALCULADO de la imagen CENTRAL:")
		print("\n- offset X = ", offset_X)
		print("\n- offset Y = ", offset_Y)
		canvas = superposeImage(canvas, imgB, [offset_X, offset_Y])
		###########################################

		## Blending de la imagen C ################
		npPointsC = np.array(pointsC, np.float32)
		npPointsC = npPointsC.reshape(-1,1,2).astype(np.float32)
		dstPointsC = cv2.perspectiveTransform(npPointsC, Hc)

		workingPointC = dstPointsC[0][0]
		workingPointB = pointsBtoC[0]
		workingPointB = (offset_X + workingPointB[0], offset_Y + workingPointB[1])
		offset_X = (workingPointB[0] - workingPointC[0]).astype(np.int64)
		offset_Y = (workingPointB[1] - workingPointC[1]).astype(np.int64)

		print("\n-- VALOR DEL OFFSET CALCULADO de la imagen DERECHA:")
		print("\n- offset X = ", offset_X)
		print("\n- offset Y = ", offset_Y)
		#canvas = superposeImage(canvas, imgC, [offset_X, offset_Y])
		canvas = alphaBlending(canvas, imgC, imC_H_mask, [offset_X, offset_Y])
		###########################################

	else: # Merging from right
		offset[0] = canvas.shape[1] - imgC.shape[1] # width
		canvas = superposeImage(canvas, imgC, offset)

		## Blending de la imagen C y B ############
		npPointsC = np.array(pointsC, np.float32)
		npPointsC = npPointsC.reshape(-1,1,2).astype(np.float32)
		dstPointsC = cv2.perspectiveTransform(npPointsC, Hc)

		workingPointC = getPointCoordsInCanvas(dstPointsC[0][0], imgC.shape[:2], canvas.shape[:2]) 
		workingPointB = pointsBtoC[0]
		offset_X = (workingPointC[0] - workingPointB[0]).astype(np.int64)
		offset_Y = (workingPointC[1] - workingPointB[1]).astype(np.int64)
		print("\n-- VALOR DEL OFFSET CALCULADO de la imagen CENTRAL:")
		print("\n- offset X = ", offset_X)
		print("\n- offset Y = ", offset_Y)
		canvas = superposeImage(canvas, imgB, [offset_X, offset_Y])
		###########################################

		## Blending de la imagen A ################
		npPointsA = np.array(pointsA, np.float32)
		npPointsA = npPointsA.reshape(-1,1,2).astype(np.float32)
		dstPointsA = cv2.perspectiveTransform(npPointsA, Ha)

		workingPointA = dstPointsA[0][0]
		workingPointB = pointsBtoA[0]
		workingPointB = (offset_X + workingPointB[0], offset_Y + workingPointB[1])
		offset_X = (workingPointB[0] - workingPointA[0]).astype(np.int64)
		offset_Y = (workingPointB[1] - workingPointA[1]).astype(np.int64)
		print("\n-- VALOR DEL OFFSET CALCULADO de la imagen IZQUIERDA:")
		print("\n- offset X = ", offset_X)
		print("\n- offset Y = ", offset_Y)
		#canvas = superposeImage(canvas, imgA, [offset_X, offset_Y])
		canvas = alphaBlending(canvas, imgA, imA_H_mask, [offset_X, offset_Y])
		###########################################

	return canvas
	