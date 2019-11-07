import numpy as np
import cv2
import pandas as pd
import os
import matplotlib.pyplot as plt

# in_low_DAPI = 0
# in_high_DAPI = 0.15
# in_low_515lp = 0
# in_high_515lp = 0.05

# in_low_DAPI = 0
# in_high_DAPI = 0.5
# in_low_515lp = 0
# in_high_515lp = 0.1
# gamma = 0.5

in_low_DAPI = 0
in_high_DAPI = 0.3
in_low_515lp = 0
in_high_515lp = 0.06
gamma = 0.64

# scalingFactor = 4570.0/1428

def lowMag2highMag(x_lowMag,y_highMag,width,height,scalingFactor):
	x = round(x_lowMag*scalingFactor)
	y = round(y_highMag*scalingFactor)
	if x >= width-1:
			x = width-1
	if y >= height-1:
		y = height-1
	# return x.astype('int'),y.astype('int')
	return x,y

def adjustContrast(I,in_low,in_high):
	# takes float 0-1
	I = I - in_low
	I = I/(in_high - in_low)
	return I

def overlay(I_fluorescence,I_DIC,alpha=0.64):
	return alpha*I_fluorescence + (1-alpha)*I_DIC

def gray2RGB(I_DIC):
	return np.dstack((I_DIC,I_DIC,I_DIC))

def combineChannels(I_DAPI,I_515lp):
	height,width = I_DAPI.shape
	I_fluorescence = np.zeros((height,width,3), np.float)
	I_fluorescence[:,:,0] = I_DAPI # openCV uses BGR
	I_fluorescence[:,:,1] = I_515lp 
	return I_fluorescence

def addCenteredBox(I,R,r,color=[0,1,1]):
	if r > R:
		r = R
	r = int(r)
	for i in range(3):
		x_min = R - r
		x_max = R + r
		y_min = R - r
		y_max = R + r
		I[y_min,x_min:x_max+1,i] = color[i]
		I[y_max,x_min:x_max+1,i] = color[i]
		I[y_min:y_max+1,x_min,i] = color[i]
		I[y_min:y_max+1,x_max,i] = color[i]

def addBoundingBox(I,x,y,r,extension=2,color=[0,0,0.6]):
	print('adding box ...')
	ny, nx, nc = I.shape
	x_min = max(x - r - extension,0)
	y_min = max(y - r - extension,0)
	x_max = min(x + r + extension,nx-1)
	y_max = min(y + r + extension,ny-1)
	for i in range(3):
		I[y_min,x_min:x_max+1,i] = color[i]
		I[y_max,x_min:x_max+1,i] = color[i]
		I[y_min:y_max+1,x_min,i] = color[i]
		I[y_min:y_max+1,x_max,i] = color[i]
	return I

def highlightSpots(bgremoved_fluorescence,spotList,contrastBoost=1.6):
	# bgremoved_fluorescence_spotBoxed = np.copy(bgremoved_fluorescence)
	bgremoved_fluorescence_spotBoxed = bgremoved_fluorescence.astype('float')/255 # this copies the image
	bgremoved_fluorescence_spotBoxed = bgremoved_fluorescence_spotBoxed*contrastBoost # enhance contrast
	for s in spotList:
		addBoundingBox(bgremoved_fluorescence_spotBoxed,int(s[0]),int(s[1]),int(s[2]))
	return bgremoved_fluorescence_spotBoxed

def cropAndSave(I,spotList,dir_spot_images,fileID,r=6,scaling=5,contrastBoost=1.2):
	# for low mag scan
	ny, nx, nc = I.shape
	I = I.astype('float')/255 * contrastBoost
	i = 0
	for s in spotList:
		x = int(s[0])
		y = int(s[1])
		# r = int(s[2])
		x_min = max(x-r,0)
		y_min = max(y-r,0)
		x_max = min(x+r,nx-1)
		y_max = min(y+r,ny-1)
		cropped = I[y_min:y_max+1,x_min:x_max+1,:]
		cropped = cv2.resize(cropped,None,fx=scaling,fy=scaling,interpolation=cv2.INTER_NEAREST)
		cv2.imwrite(dir_spot_images + '/' + fileID + '_' + str(i).zfill(4) + '.png',cropped*255)
		i = i + 1

def cropAddBoxAndSave(fileID,dir_in_spotData,dir_in_highMagImages,dir_in_highMagImages_DIC,dir_in_lowMagImages,dir_in_correspondenceCheckImages,dir_out,r=30,scale=5,lowMagFOVSize=1428,scalingFactor=4570.0/1428):
# this function is for generating tiled images for manual label assignment

	# check if aligned FOV exist or not
	if not os.path.exists(dir_in_highMagImages + '/' + fileID + '_ch1.tif'):
		return

	if not os.path.exists(dir_out):
			os.mkdir(dir_out)

	# parse fileID
	rowIdx,colIdx = fileID.split('_')
	rowIdx = int(rowIdx)
	colIdx = int(colIdx)

	# load aligned and background-removed high mag images
	I_DAPI = cv2.imread(dir_in_highMagImages + '/' + fileID + '_ch1.tif',cv2.IMREAD_UNCHANGED)
	I_515lp = cv2.imread(dir_in_highMagImages + '/' + fileID + '_ch0.tif',cv2.IMREAD_UNCHANGED)
	I_DIC = cv2.imread(dir_in_highMagImages_DIC + '/' + fileID + '_ch2.tif',cv2.IMREAD_UNCHANGED)
	I_DIC = gray2RGB(I_DIC)
	
	# load low mag bright field and correspondence check
	I_lowMag_bf = cv2.imread(dir_in_lowMagImages + '/' + '_' + str(rowIdx).zfill(2) + '_' + str(colIdx).zfill(2) + '_bf.png', cv2.IMREAD_UNCHANGED)
	I_lowMag_bf = gray2RGB(I_lowMag_bf)
	I_correspondence = cv2.imread(dir_in_correspondenceCheckImages + '/' + fileID + '.png',cv2.IMREAD_UNCHANGED)

	# convert to float
	I_DAPI = I_DAPI.astype('float')/65535
	I_515lp = I_515lp.astype('float')/65535
	I_DIC = I_DIC.astype('float')/65535
	I_lowMag_bf = I_lowMag_bf.astype('float')/255
	I_lowMag_bf = adjustContrast(I_lowMag_bf,220.0/255,1)
	I_correspondence = I_correspondence.astype('float')/255
	
	# get the image dimension
	height,width = I_DAPI.shape

	# adjust contrast and create overlay images
	I_DAPI_1 = adjustContrast(I_DAPI,in_low_DAPI,in_high_DAPI)
	I_515lp_1 = adjustContrast(I_515lp,in_low_515lp,in_high_515lp)
	I_fluorescence_1 = combineChannels(I_DAPI_1,I_515lp_1)
	I_overlay_1 = overlay(I_fluorescence_1,I_DIC)

	I_DAPI_2 = I_DAPI_1*1.5
	I_515lp_2 = I_515lp_1*1.5
	I_fluorescence_2 = combineChannels(I_DAPI_2,I_515lp_2)
	I_overlay_2 = overlay(I_fluorescence_2,I_DIC)

	I_DAPI_3 = I_DAPI_1*2
	I_515lp_3 = I_515lp_1*2
	I_fluorescence_3 = combineChannels(I_DAPI_3,I_515lp_3)
	I_overlay_3 = overlay(I_fluorescence_3,I_DIC)

	# convert grayscale images to RGB images
	I_DAPI_1 = gray2RGB(I_DAPI_1)
	I_515lp_1 = gray2RGB(I_515lp_1)
	I_DAPI_2 = gray2RGB(I_DAPI_2)
	I_515lp_2 = gray2RGB(I_515lp_2)
	I_DAPI_3 = gray2RGB(I_DAPI_3)
	I_515lp_3 = gray2RGB(I_515lp_3)

	'''
	cv2.imshow('DAPI',I_DAPI[100:400,100:400])
	cv2.imshow('515lp',I_515lp[100:400,100:400])
	cv2.imshow('DAPI + 515lp',I_fluorescence[100:400,100:400,:])
	cv2.imshow('overlay',I_overlay[100:400,100:400,:])
	cv2.waitKey(0)
	'''

	# load spots detected from the low mag scan
	spotData_FOV = np.loadtxt(dir_in_spotData + '/' + fileID + '.csv', delimiter=',',skiprows=1)
	
	# do the work
	for spot in spotData_FOV:

		# initialize arrays to hold the image
		I_DIC_cropped = np.zeros((2*r+1,2*r+1,3), np.float)
		I_lowMag_bf_cropped = np.zeros((2*r+1,2*r+1,3), np.float)
		I_correspondence_cropped = np.zeros((2*r+1,2*r+1,3), np.float)

		I_DAPI_1_cropped = np.zeros((2*r+1,2*r+1,3), np.float)
		I_515lp_1_cropped = np.zeros((2*r+1,2*r+1,3), np.float)
		I_fluorescence_1_cropped = np.zeros((2*r+1,2*r+1,3), np.float)
		I_overlay_1_cropped = np.zeros((2*r+1,2*r+1,3), np.float)

		I_DAPI_2_cropped = np.zeros((2*r+1,2*r+1,3), np.float)
		I_515lp_2_cropped = np.zeros((2*r+1,2*r+1,3), np.float)
		I_fluorescence_2_cropped = np.zeros((2*r+1,2*r+1,3), np.float)
		I_overlay_2_cropped = np.zeros((2*r+1,2*r+1,3), np.float)

		I_DAPI_3_cropped = np.zeros((2*r+1,2*r+1,3), np.float)
		I_515lp_3_cropped = np.zeros((2*r+1,2*r+1,3), np.float)
		I_fluorescence_3_cropped = np.zeros((2*r+1,2*r+1,3), np.float)
		I_overlay_3_cropped = np.zeros((2*r+1,2*r+1,3), np.float)

		# get the spot coordinate
		x_lowMag = spot[2].astype('int')
		y_lowMag = spot[3].astype('int')
		r_spot = spot[4]*2
		x,y = lowMag2highMag(x_lowMag,y_lowMag,width,height,scalingFactor)

		# identify cropping region in the FOV and in the cropped image (high mag)
		x_start = max(0,x-r)
		x_end = min(x+r,width-1)
		y_start = max(0,y-r)
		y_end = min(y+r,height-1)
		x_idx_FOV = slice(x_start,x_end+1)
		y_idx_FOV = slice(y_start,y_end+1)

		x_cropped_start = x_start - (x-r)
		x_cropped_end = (2*r+1-1) - ((x+r)-x_end)
		y_cropped_start = y_start - (y-r)
		y_cropped_end = (2*r+1-1) - ((y+r)-y_end)
		x_idx_cropped = slice(x_cropped_start,x_cropped_end+1)
		y_idx_cropped = slice(y_cropped_start,y_cropped_end+1)

		# do the cropping (high mag)
		I_DIC_cropped[y_idx_cropped,x_idx_cropped,:] = I_DIC[y_idx_FOV,x_idx_FOV,:]
		I_DAPI_1_cropped[y_idx_cropped,x_idx_cropped,:] = I_DAPI_1[y_idx_FOV,x_idx_FOV,:]
		I_DAPI_2_cropped[y_idx_cropped,x_idx_cropped,:] = I_DAPI_2[y_idx_FOV,x_idx_FOV,:]
		I_DAPI_3_cropped[y_idx_cropped,x_idx_cropped,:] = I_DAPI_3[y_idx_FOV,x_idx_FOV,:]
		I_515lp_1_cropped[y_idx_cropped,x_idx_cropped,:] = I_515lp_1[y_idx_FOV,x_idx_FOV,:]
		I_515lp_2_cropped[y_idx_cropped,x_idx_cropped,:] = I_515lp_2[y_idx_FOV,x_idx_FOV,:]
		I_515lp_3_cropped[y_idx_cropped,x_idx_cropped,:] = I_515lp_3[y_idx_FOV,x_idx_FOV,:]
		I_fluorescence_1_cropped[y_idx_cropped,x_idx_cropped,:] = I_fluorescence_1[y_idx_FOV,x_idx_FOV,:]
		I_fluorescence_2_cropped[y_idx_cropped,x_idx_cropped,:] = I_fluorescence_2[y_idx_FOV,x_idx_FOV,:]
		I_fluorescence_3_cropped[y_idx_cropped,x_idx_cropped,:] = I_fluorescence_3[y_idx_FOV,x_idx_FOV,:]
		I_overlay_1_cropped[y_idx_cropped,x_idx_cropped,:] = I_overlay_1[y_idx_FOV,x_idx_FOV,:]
		I_overlay_2_cropped[y_idx_cropped,x_idx_cropped,:] = I_overlay_2[y_idx_FOV,x_idx_FOV,:]
		I_overlay_3_cropped[y_idx_cropped,x_idx_cropped,:] = I_overlay_3[y_idx_FOV,x_idx_FOV,:]

		# check if the center is outside the highMag scan
		if I_DIC_cropped[r,r,0]==0:
			continue

		# check if the spot is at the edge of the low mag FOV
		r_lowMag = round(r/scalingFactor)
		if (x_lowMag - r_lowMag < 0) or (x_lowMag + r_lowMag >= lowMagFOVSize):
			continue
		if (y_lowMag - r_lowMag < 0) or (y_lowMag + r_lowMag >= lowMagFOVSize):
			continue

		# crop and resize low mag images
		x_lowMag_idx_cropped = slice(x_lowMag-r_lowMag,x_lowMag+r_lowMag+1)
		y_lowMag_idx_cropped = slice(y_lowMag-r_lowMag,y_lowMag+r_lowMag+1)
		I_lowMag_bf_cropped = I_lowMag_bf[y_lowMag_idx_cropped,x_lowMag_idx_cropped,:]
		I_correspondence_cropped = I_correspondence[y_lowMag_idx_cropped,x_lowMag_idx_cropped,:]
		I_lowMag_bf_cropped = cv2.resize(I_lowMag_bf_cropped,(2*r+1,2*r+1),cv2.INTER_NEAREST)
		I_correspondence_cropped = cv2.resize(I_correspondence_cropped,(2*r+1,2*r+1),cv2.INTER_NEAREST)


def createTiledImageForVisualization_base(x_lowMag,y_lowMag,I_lowMag_fluorescence,I_lowMag_bf,I_lowMag_overlay,I_highMag_DIC,I_highMag_DAPI,I_highMag_515lp,r=30,scale=5,lowMagFOVSize=1428,scalingFactor=4570.0/1428,addBox=False,boxColor=[0,1,1]):
	# each cropped image (in high mag) has length and width of 2*r+1
	# initialize arrays to hold the image

	I_DIC_cropped = np.zeros((2*r+1,2*r+1,3), np.float)
	I_DAPI_cropped = np.zeros((2*r+1,2*r+1,3), np.float)
	I_515lp_cropped = np.zeros((2*r+1,2*r+1,3), np.float)

	# get the spot coordinate in high mag images
	# get the image dimension
	height,width,channels = I_highMag_DAPI.shape
	x,y = lowMag2highMag(x_lowMag,y_lowMag,width,height,scalingFactor)

	# identify cropping region in the FOV and in the cropped image (high mag)
	x_start = max(0,x-r)
	x_end = min(x+r,width-1)
	y_start = max(0,y-r)
	y_end = min(y+r,height-1)
	x_idx_FOV = slice(x_start,x_end+1)
	y_idx_FOV = slice(y_start,y_end+1)

	x_cropped_start = x_start - (x-r)
	x_cropped_end = (2*r+1-1) - ((x+r)-x_end)
	y_cropped_start = y_start - (y-r)
	y_cropped_end = (2*r+1-1) - ((y+r)-y_end)
	x_idx_cropped = slice(x_cropped_start,x_cropped_end+1)
	y_idx_cropped = slice(y_cropped_start,y_cropped_end+1)

	# do the cropping (high mag)
	I_DIC_cropped[y_idx_cropped,x_idx_cropped,:] = I_highMag_DIC[y_idx_FOV,x_idx_FOV,:]
	I_DAPI_cropped[y_idx_cropped,x_idx_cropped,:] = I_highMag_DAPI[y_idx_FOV,x_idx_FOV,:]
	I_515lp_cropped[y_idx_cropped,x_idx_cropped,:] = I_highMag_515lp[y_idx_FOV,x_idx_FOV,:]


	# identify cropping region in the FOV and in the cropped image (low mag)
	r_lowMag = round(r/scalingFactor)
	height_lowMag,width_lowMag,channels = I_lowMag_fluorescence.shape

	I_lowMag_bf_cropped = np.zeros((2*r_lowMag+1,2*r_lowMag+1,3), np.float)
	I_lowMag_fluorescence_cropped = np.zeros((2*r_lowMag+1,2*r_lowMag+1,3), np.float)
	I_lowMag_overlay_cropped = np.zeros((2*r_lowMag+1,2*r_lowMag+1,3), np.float)

	x_lowMag_start = max(0,x_lowMag-r_lowMag)
	x_lowMag_end = min(x_lowMag+r_lowMag,width_lowMag-1)
	y_lowMag_start = max(0,y_lowMag-r_lowMag)
	y_lowMag_end = min(y_lowMag+r_lowMag,height_lowMag-1)
	x_lowMag_idx_FOV = slice(x_lowMag_start,x_lowMag_end+1)
	y_lowMag_idx_FOV = slice(y_lowMag_start,y_lowMag_end+1)

	x_lowMag_cropped_start = x_lowMag_start - (x_lowMag-r_lowMag)
	x_lowMag_cropped_end = (2*r_lowMag+1-1) - ((x_lowMag+r_lowMag)-x_lowMag_end)
	y_lowMag_cropped_start = y_lowMag_start - (y_lowMag-r_lowMag)
	y_lowMag_cropped_end = (2*r_lowMag+1-1) - ((y_lowMag+r_lowMag)-y_lowMag_end)
	x_lowMag_idx_cropped = slice(x_lowMag_cropped_start,x_lowMag_cropped_end+1)
	y_lowMag_idx_cropped = slice(y_lowMag_cropped_start,y_lowMag_cropped_end+1)

	I_lowMag_bf_cropped[y_lowMag_idx_cropped,x_lowMag_idx_cropped,:] = I_lowMag_bf[y_lowMag_idx_FOV,x_lowMag_idx_FOV,:]
	I_lowMag_fluorescence_cropped[y_lowMag_idx_cropped,x_lowMag_idx_cropped,:] = I_lowMag_fluorescence[y_lowMag_idx_FOV,x_lowMag_idx_FOV,:]
	I_lowMag_overlay_cropped[y_lowMag_idx_cropped,x_lowMag_idx_cropped,:] = I_lowMag_overlay[y_lowMag_idx_FOV,x_lowMag_idx_FOV,:]

	I_lowMag_bf_cropped = cv2.resize(I_lowMag_bf_cropped,(2*r+1,2*r+1),cv2.INTER_NEAREST)
	I_lowMag_fluorescence_cropped = cv2.resize(I_lowMag_fluorescence_cropped,(2*r+1,2*r+1),cv2.INTER_NEAREST)
	I_lowMag_overlay_cropped = cv2.resize(I_lowMag_overlay_cropped,(2*r+1,2*r+1),cv2.INTER_NEAREST)

	if addBox:
		addCenteredBox(I_lowMag_bf_cropped,r,round((3+2)*scalingFactor),boxColor)
		addCenteredBox(I_lowMag_fluorescence_cropped,r,round((3+2)*scalingFactor),boxColor)
		addCenteredBox(I_lowMag_overlay_cropped,r,round((3+2)*scalingFactor),boxColor)

	# tile the images
	# row1 = np.hstack((I_lowMag_fluorescence_cropped,I_lowMag_bf_cropped,I_lowMag_overlay_cropped))
	row1 = np.hstack((I_lowMag_fluorescence_cropped,I_lowMag_overlay_cropped,I_lowMag_bf_cropped))
	row2 = np.hstack((I_DAPI_cropped,I_515lp_cropped,I_DIC_cropped))
	tiled = np.vstack((row1,row2))

	# scale up the image
	tiled = cv2.resize(tiled,None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST)
	return tiled

def createTiledImageForVisualization(x_lowMag,y_lowMag,I_lowMag_fluorescence,I_lowMag_bf,I_lowMag_overlay,I_highMag_DIC,I_highMag_DAPI,I_highMag_515lp,r=30,scale=5,addBox=False,withScatterPlot=False,scatterPlotBackground=None,R=None,G=None,B=None,color=None,lowMagFOVSize=1428,scalingFactor=4570.0/1428):

	tiled = createTiledImageForVisualization_base(x_lowMag,y_lowMag,I_lowMag_fluorescence,I_lowMag_bf,I_lowMag_overlay,I_highMag_DIC,I_highMag_DAPI,I_highMag_515lp,r=r,scale=scale,addBox=addBox)

	if withScatterPlot==False:
		pass
	else:
		h,w,c = scatterPlotBackground.shape
		plt.figure(figsize=(h/100,w/100),dpi=100)
		# plt.figure(figsize=(6.1,6.1),dpi=100)
		ax = plt.subplot(111)
		plt.scatter(R/B,G/B,s=G/16,c=color)
		plt.xlabel("R/B")
		plt.ylabel("G/B")
		plt.xlim(0,1.2)
		plt.ylim(0.3,2.5)
		plt.savefig('tmp.png')
		plt.close()
		scatterPlot = cv2.imread('tmp.png').astype('float')/255
		scatterPlot = overlay(scatterPlot,scatterPlotBackground,alpha=0.75)
		# inserted two lines
		h,w,c = tiled.shape
		scatterPlot = cv2.resize(scatterPlot,(h,h),cv2.INTER_NEAREST)
		tiled = np.hstack((tiled,scatterPlot))

	return tiled


def generateSpotVisualizationsForTheGivenFOV(rowIdx,colIdx,spotData,dir_in_highMagImages,dir_in_highMagImages_DIC,dir_in_lowMagImages,dir_out,withScatterPlot=False,scatterPlotBackground=None,r=30,scale=5,addBox=False,lowMagFOVSize=1428,scalingFactor=4570.0/1428):
# this function creates tiled images for visualization
# do it FOV by FOV

# default [zoomed-in view]: 		r=30,scale=5,addBox=False
# alternative [zoomed-out view]:	r=120,scale=1,addBox=True

	fileID = str(rowIdx).zfill(4) + '_' + str(colIdx).zfill(4)

	# load aligned and background-removed high mag images
	I_DAPI = cv2.imread(dir_in_highMagImages + '/' + fileID + '_ch1.tif',cv2.IMREAD_UNCHANGED)
	I_515lp = cv2.imread(dir_in_highMagImages + '/' + fileID + '_ch0.tif',cv2.IMREAD_UNCHANGED)
	I_DIC = cv2.imread(dir_in_highMagImages_DIC + '/' + fileID + '_ch2.tif',cv2.IMREAD_UNCHANGED)
	
	# load low mag bright field and correspondence check
	I_lowMag_bf = cv2.imread(dir_in_lowMagImages + '/' + '_' + str(rowIdx).zfill(2) + '_' + str(colIdx).zfill(2) + '_bf.png', cv2.IMREAD_UNCHANGED)
	I_lowMag_bf = gray2RGB(I_lowMag_bf)
	I_lowMag_fluorescence = cv2.imread(dir_in_lowMagImages + '/' + '_' + str(rowIdx).zfill(2) + '_' + str(colIdx).zfill(2) + '_fluorescent_linearRGB.png', cv2.IMREAD_UNCHANGED)

	# print(dir_in_lowMagImages + '/' + '_' + str(rowIdx).zfill(2) + '_' + str(colIdx).zfill(2) + '_fluorescence.png')

	# convert to float
	I_DAPI = I_DAPI.astype('float')/65535
	I_515lp = I_515lp.astype('float')/65535
	I_DIC = I_DIC.astype('float')/65535
	I_lowMag_bf = I_lowMag_bf.astype('float')/255
	I_lowMag_bf = adjustContrast(I_lowMag_bf,220.0/255,1)
	I_lowMag_fluorescence = I_lowMag_fluorescence.astype('float')/255
	I_lowMag_overlay = overlay(I_lowMag_fluorescence,I_lowMag_bf)

	# adjust contrast/gamma and create overlay images
	gamma = 0.64
	I_DAPI = adjustContrast(I_DAPI,in_low_DAPI,in_high_DAPI)**gamma
	I_515lp = adjustContrast(I_515lp,in_low_515lp,in_high_515lp)**gamma
	#I_fluorescence = combineChannels(I_DAPI_1,I_515lp_1)
	#I_overlay = overlay(I_fluorescence_1,I_DIC)

	# convert grayscale images to RGB images
	I_DAPI = gray2RGB(I_DAPI)
	I_515lp = gray2RGB(I_515lp)
	I_DIC = gray2RGB(I_DIC)
	
	if withScatterPlot:
		# with scatterplot
		for idx, entry in spotData.iterrows():
			x = entry['x']
			y = entry['y']
			R = entry['R']
			G = entry['G']
			B = entry['B']
			if entry['Annotation']=='Parasite':
				color = '#ff7f0e'
			elif entry['Annotation']=='Platelet':
				color = '#1f77b4'
			else:
				color = '#2ca02c'
			tiled = createTiledImageForVisualization(x,y,I_lowMag_fluorescence,I_lowMag_bf,I_lowMag_overlay,I_DIC,I_DAPI,I_515lp,withScatterPlot=True,scatterPlotBackground=scatterPlotBackground,R=R,G=G,B=B,color=color,r=r,scale=scale,addBox=addBox)
			cv2.imwrite(dir_out + '/' + fileID + '_' + str(x) + '_' + str(y) + '_' + entry['Annotation'] + '.png',tiled*255)

	else:
		# without scatter plot
		for idx, entry in spotData.iterrows():
			x = entry['x']
			y = entry['y']
			tiled = createTiledImageForVisualization(x,y,I_lowMag_fluorescence,I_lowMag_bf,I_lowMag_overlay,I_DIC,I_DAPI,I_515lp,withScatterPlot=False,scatterPlotBackground=None,R=None,G=None,B=None,color=None,r=r,scale=scale,addBox=addBox)
			cv2.imwrite(dir_out + '/' + fileID + '_' + str(x) + '_' + str(y) + '_' + entry['Annotation'] + '.png',tiled*255)

def highLightSpotsInLowMag(rowIdx,colIdx,spotData,dir_in_lowMagImages,dir_out,color=[0,0.6,0.6]):
# highlight all the spots in the supplied pd file with the specified color
	
	fileID = str(rowIdx).zfill(4) + '_' + str(colIdx).zfill(4)

	# load low mag images
	I_lowMag_bf = cv2.imread(dir_in_lowMagImages + '/' + '_' + str(rowIdx).zfill(2) + '_' + str(colIdx).zfill(2) + '_bf.png', cv2.IMREAD_UNCHANGED)
	I_lowMag_bf = gray2RGB(I_lowMag_bf)
	I_lowMag_fluorescence = cv2.imread(dir_in_lowMagImages + '/' + '_' + str(rowIdx).zfill(2) + '_' + str(colIdx).zfill(2) + '_fluorescent_linearRGB.png', cv2.IMREAD_UNCHANGED)

	# convert to float
	I_lowMag_bf = I_lowMag_bf.astype('float')/255
	I_lowMag_bf = adjustContrast(I_lowMag_bf,220.0/255,1)
	I_lowMag_fluorescence = I_lowMag_fluorescence.astype('float')/255
	I_lowMag_overlay = overlay(I_lowMag_fluorescence,I_lowMag_bf)

	# add box
	for idx, entry in spotData.iterrows():
		x = entry['x']
		y = entry['y']
		r = entry['r']
		I_lowMag_overlay = addBoundingBox(I_lowMag_overlay,x,y,r,extension=2,color=color)
		I_lowMag_bf = addBoundingBox(I_lowMag_bf,x,y,r,extension=2,color=color)
		I_lowMag_fluorescence = addBoundingBox(I_lowMag_fluorescence,x,y,r,extension=2,color=color)

	# write image
	cv2.imwrite(dir_out + '/' + fileID + '_overlay.png',I_lowMag_overlay*255)
	cv2.imwrite(dir_out + '/' + fileID + '_bf.png',I_lowMag_bf*255)
	cv2.imwrite(dir_out + '/' + fileID + '_fluorescence.png',I_lowMag_fluorescence*255)


'''
# merge into generateSpotVisualizationsForTheGivenFOV
def generateSpotVisualizationsForTheGivenFOV_withScatterPlot(rowIdx,colIdx,spotData,dir_in_highMagImages,dir_in_highMagImages_DIC,dir_in_lowMagImages,dir_out,scatterPlotBackground,r=30,scale=5,addBox=False,lowMagFOVSize=1428,scalingFactor=4570.0/1428,):
# this function creates tiled images for visualization
# do it FOV by FOV

# default [zoomed-in view]: 		r=30,scale=5,addBox=False
# alternative [zoomed-out view]:	r=120,scale=1,addBox=True

	fileID = str(rowIdx).zfill(4) + '_' + str(colIdx).zfill(4)

	# load aligned and background-removed high mag images
	I_DAPI = cv2.imread(dir_in_highMagImages + '/' + fileID + '_ch1.tif',cv2.IMREAD_UNCHANGED)
	I_515lp = cv2.imread(dir_in_highMagImages + '/' + fileID + '_ch0.tif',cv2.IMREAD_UNCHANGED)
	I_DIC = cv2.imread(dir_in_highMagImages_DIC + '/' + fileID + '_ch2.tif',cv2.IMREAD_UNCHANGED)
	
	# load low mag bright field and correspondence check
	I_lowMag_bf = cv2.imread(dir_in_lowMagImages + '/' + '_' + str(rowIdx).zfill(2) + '_' + str(colIdx).zfill(2) + '_bf.png', cv2.IMREAD_UNCHANGED)
	I_lowMag_bf = gray2RGB(I_lowMag_bf)
	I_lowMag_fluorescence = cv2.imread(dir_in_lowMagImages + '/' + '_' + str(rowIdx).zfill(2) + '_' + str(colIdx).zfill(2) + '_fluorescent_linearRGB.png', cv2.IMREAD_UNCHANGED)

	# print(dir_in_lowMagImages + '/' + '_' + str(rowIdx).zfill(2) + '_' + str(colIdx).zfill(2) + '_fluorescence.png')

	# convert to float
	I_DAPI = I_DAPI.astype('float')/65535
	I_515lp = I_515lp.astype('float')/65535
	I_DIC = I_DIC.astype('float')/65535
	I_lowMag_bf = I_lowMag_bf.astype('float')/255
	I_lowMag_bf = adjustContrast(I_lowMag_bf,220.0/255,1)
	I_lowMag_fluorescence = I_lowMag_fluorescence.astype('float')/255
	I_lowMag_overlay = overlay(I_lowMag_fluorescence,I_lowMag_bf)

	# adjust contrast/gamma and create overlay images
	gamma = 0.64
	I_DAPI = adjustContrast(I_DAPI,in_low_DAPI,in_high_DAPI)**gamma
	I_515lp = adjustContrast(I_515lp,in_low_515lp,in_high_515lp)**gamma
	#I_fluorescence = combineChannels(I_DAPI_1,I_515lp_1)
	#I_overlay = overlay(I_fluorescence_1,I_DIC)

	# convert grayscale images to RGB images
	I_DAPI = gray2RGB(I_DAPI)
	I_515lp = gray2RGB(I_515lp)
	I_DIC = gray2RGB(I_DIC)
	
	# do the work
	for idx, entry in spotData.iterrows():
		x = entry['x']
		y = entry['y']
		R = entry['R']
		G = entry['G']
		B = entry['B']
		if entry['Annotation']=='Parasite':
			color = '#ff7f0e'
		else:
			color = '#1f77b4'
		tiled = createTiledImageForVisualization_withScatterPlot(x,y,I_lowMag_fluorescence,I_lowMag_bf,I_lowMag_overlay,I_DIC,I_DAPI,I_515lp,R,G,B,color,scatterPlotBackground,r=r,scale=scale,addBox=addBox)
		cv2.imwrite(dir_out + '/' + fileID + '_' + str(x) + '_' + str(y) + '_' + entry['Annotation'] + '.png',tiled*255)
'''


'''
# merge into generateSpotVisualizationsForTheGivenFOV_withScatterPlot
def generateSpotVisualizationsForTheGivenFOV_withScatterPlot_others(rowIdx,colIdx,spotData,dir_in_highMagImages,dir_in_highMagImages_DIC,dir_in_lowMagImages,dir_out,scatterPlotBackground,r=30,scale=5,lowMagFOVSize=1428,scalingFactor=4570.0/1428,):
# this function creates tiled images for visualization
# do it FOV by FOV

	fileID = str(rowIdx).zfill(4) + '_' + str(colIdx).zfill(4)

	# load aligned and background-removed high mag images
	I_DAPI = cv2.imread(dir_in_highMagImages + '/' + fileID + '_ch1.tif',cv2.IMREAD_UNCHANGED)
	I_515lp = cv2.imread(dir_in_highMagImages + '/' + fileID + '_ch0.tif',cv2.IMREAD_UNCHANGED)
	I_DIC = cv2.imread(dir_in_highMagImages_DIC + '/' + fileID + '_ch2.tif',cv2.IMREAD_UNCHANGED)
	
	# load low mag bright field and correspondence check
	I_lowMag_bf = cv2.imread(dir_in_lowMagImages + '/' + '_' + str(rowIdx).zfill(2) + '_' + str(colIdx).zfill(2) + '_bf.png', cv2.IMREAD_UNCHANGED)
	I_lowMag_bf = gray2RGB(I_lowMag_bf)
	I_lowMag_fluorescence = cv2.imread(dir_in_lowMagImages + '/' + '_' + str(rowIdx).zfill(2) + '_' + str(colIdx).zfill(2) + '_fluorescent_linearRGB.png', cv2.IMREAD_UNCHANGED)

	# print(dir_in_lowMagImages + '/' + '_' + str(rowIdx).zfill(2) + '_' + str(colIdx).zfill(2) + '_fluorescence.png')

	# convert to float
	I_DAPI = I_DAPI.astype('float')/65535
	I_515lp = I_515lp.astype('float')/65535
	I_DIC = I_DIC.astype('float')/65535
	I_lowMag_bf = I_lowMag_bf.astype('float')/255
	I_lowMag_bf = adjustContrast(I_lowMag_bf,220.0/255,1)
	I_lowMag_fluorescence = I_lowMag_fluorescence.astype('float')/255
	I_lowMag_overlay = overlay(I_lowMag_fluorescence,I_lowMag_bf)

	# adjust contrast/gamma and create overlay images
	gamma = 0.64
	I_DAPI = adjustContrast(I_DAPI,in_low_DAPI,in_high_DAPI)**gamma
	I_515lp = adjustContrast(I_515lp,in_low_515lp,in_high_515lp)**gamma
	#I_fluorescence = combineChannels(I_DAPI_1,I_515lp_1)
	#I_overlay = overlay(I_fluorescence_1,I_DIC)

	# convert grayscale images to RGB images
	I_DAPI = gray2RGB(I_DAPI)
	I_515lp = gray2RGB(I_515lp)
	I_DIC = gray2RGB(I_DIC)
	
	# do the work
	for idx, entry in spotData.iterrows():
		x = entry['x']
		y = entry['y']
		R = entry['R']
		G = entry['G']
		B = entry['B']
		# color = '#ff7f0e'
		color = '#1f77b4'
		tiled = createTiledImageForVisualization_withScatterPlot(x,y,I_lowMag_fluorescence,I_lowMag_bf,I_lowMag_overlay,I_DIC,I_DAPI,I_515lp,R,G,B,color,scatterPlotBackground,r=120,scale=1,addBox=True)
		cv2.imwrite(dir_out + '/' + fileID + '_' + str(x) + '_' + str(y) + '_' + entry['Annotation'] + '.png',tiled*255)
'''