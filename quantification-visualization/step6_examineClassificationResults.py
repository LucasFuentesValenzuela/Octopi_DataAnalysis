import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import re
import os
import utils

dir_FOV_highMag = '../Octopi high mag//3C_step2_removeBGfromalignedHighMag/alignedHighMag_bgRemoved'
dir_FOV_highMag_DIC = '../Octopi high mag/3C_step1_alignHighMagImages/Registered'
dir_in_lowMagImages = '../Octopi high mag/3C_step1_alignHighMagImages/Octopi_BGremoved'

dir_out_falsePositives = 'classificationResult_falsePositives'
dir_out_falseNegatives = 'classificationResult_falseNegatives'

# create folders
if not os.path.exists(dir_out_falsePositives):
	os.mkdir(dir_out_falsePositives)
if not os.path.exists(dir_out_falseNegatives):
	os.mkdir(dir_out_falseNegatives)


# load the classification result
result_pd = pd.read_csv('X_test.csv', index_col=None, header=0)
result_pd['FOV_ID'] = result_pd['FOV_row'].map(str).str.zfill(4) + '_' + result_pd['FOV_col'].map(str).str.zfill(4)
# print(result_pd)

# parasites misidentified as platelets/others
parasite_falseNegatives_pd = result_pd[ (result_pd['pred']==0) & (result_pd['Annotation']=='Parasite') ]
print(parasite_falseNegatives_pd)

# nonparasites misidentified as parasites
parasite_falsePositives_pd = result_pd[ (result_pd['pred']==1) & (result_pd['Annotation']!='Parasite') ]
print(parasite_falsePositives_pd)

# create images for examination [falseNegatives]
FOVs = parasite_falseNegatives_pd.FOV_ID.unique()
scatterPlotBackground = cv2.imread('scatter plot background.png')
scatterPlotBackground = scatterPlotBackground.astype('float')/255
for fov in FOVs:
	rowIdx,colIdx = fov.split('_')
	rowIdx = int(rowIdx)
	colIdx = int(colIdx)
	spotData_singleFOV = parasite_falseNegatives_pd[parasite_falseNegatives_pd['FOV_ID']==fov]
	utils.generateSpotVisualizationsForTheGivenFOV(rowIdx,colIdx,spotData_singleFOV,dir_FOV_highMag,dir_FOV_highMag_DIC,dir_in_lowMagImages,dir_out_falseNegatives,withScatterPlot=True,scatterPlotBackground=scatterPlotBackground,r=120,scale=1,addBox=True,lowMagFOVSize=1428,scalingFactor=4570.0/1428)
	print(spotData_singleFOV)

# create images for examination [falsePositives]
FOVs = parasite_falsePositives_pd.FOV_ID.unique()
scatterPlotBackground = cv2.imread('scatter plot background.png')
scatterPlotBackground = scatterPlotBackground.astype('float')/255
for fov in FOVs:
	rowIdx,colIdx = fov.split('_')
	rowIdx = int(rowIdx)
	colIdx = int(colIdx)
	spotData_singleFOV = parasite_falsePositives_pd[parasite_falsePositives_pd['FOV_ID']==fov]
	utils.generateSpotVisualizationsForTheGivenFOV(rowIdx,colIdx,spotData_singleFOV,dir_FOV_highMag,dir_FOV_highMag_DIC,dir_in_lowMagImages,dir_out_falsePositives,withScatterPlot=True,scatterPlotBackground=scatterPlotBackground,r=120,scale=1,addBox=True,lowMagFOVSize=1428,scalingFactor=4570.0/1428)
	print(spotData_singleFOV)