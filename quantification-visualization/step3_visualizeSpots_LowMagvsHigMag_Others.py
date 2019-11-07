# this script shows spots with assigned class labels in the low mag FOVs

import numpy as np
import cv2
import utils
import multiprocessing as mp
import os
import pandas as pd

dir_spot_data = 'spot_data'
dir_visualizations_withScatterPlot_others = 'visualization_withScatterPlot_others'
dir_FOV_highMag = '../Octopi high mag//3C_step2_removeBGfromalignedHighMag/alignedHighMag_bgRemoved'
dir_FOV_highMag_DIC = '../Octopi high mag/3C_step1_alignHighMagImages/Registered'
dir_in_lowMagImages = '../Octopi high mag/3C_step1_alignHighMagImages/Octopi_BGremoved'

URLPrefix = 'https://octopi201910.s3-us-west-1.amazonaws.com/3C/'
fileExtension = '.jpeg'

# load spot data
spotData_all_pd = pd.read_csv('spotData.csv', index_col=None, header=0)
spotData_all_pd['URL'] = URLPrefix + spotData_all_pd['FOV_row'].map(str).str.zfill(4) + '_' + spotData_all_pd['FOV_col'].map(str).str.zfill(4) + '/' + spotData_all_pd['x'].map(str) + '_' + spotData_all_pd['y'].map(str) + '.jpeg'

# create a new data frame to hold annotated entries
spotData_annotated_pd = pd.DataFrame(columns=spotData_all_pd.columns)
spotData_annotated_pd['Annotation'] = None

# parse jason file and populate the dataframe with annotation
annotation = pd.read_json('annotation.json',lines=True)
for idx, entry in annotation.iterrows():

	if entry['annotation'] is None:
		continue
	if 'Parasite' in entry['annotation']['labels']:
		continue
		# entry_spotData = spotData_all_pd[spotData_all_pd['URL'] == entry['content']].copy()
		# entry_spotData['Annotation'] = 'Parasite'
		# spotData_annotated_pd = spotData_annotated_pd.append(entry_spotData, ignore_index=True, sort=False)
	if 'Platelet' in entry['annotation']['labels']:
		continue
		# entry_spotData = spotData_all_pd[spotData_all_pd['URL'] == entry['content']].copy()
		# entry_spotData['Annotation'] = 'Platelet'
		# spotData_annotated_pd = spotData_annotated_pd.append(entry_spotData, ignore_index=True, sort=False)
	else:
		entry_spotData = spotData_all_pd[spotData_all_pd['URL'] == entry['content']].copy()
		entry_spotData['Annotation'] = 'non-parasite, non-platelet'
		spotData_annotated_pd = spotData_annotated_pd.append(entry_spotData, ignore_index=True, sort=False)

spotData_pd = spotData_annotated_pd
spotData_pd['FOV_ID'] = spotData_pd['FOV_row'].map(str).str.zfill(4) + '_' + spotData_pd['FOV_col'].map(str).str.zfill(4)

# remove spots with saturated pixels
idx_spotWithSaturatedPixels = spotData_pd['numSaturatedPixels']>0
spotData_pd = spotData_pd[~idx_spotWithSaturatedPixels]

# with scatter plot
# create folders
if not os.path.exists(dir_visualizations_withScatterPlot_others):
	os.mkdir(dir_visualizations_withScatterPlot_others)
FOVs = spotData_pd.FOV_ID.unique()
scatterPlotBackground = cv2.imread('scatter plot background.png')
scatterPlotBackground = scatterPlotBackground.astype('float')/255
for fov in FOVs:
	rowIdx,colIdx = fov.split('_')
	rowIdx = int(rowIdx)
	colIdx = int(colIdx)
	spotData_singleFOV = spotData_pd[spotData_pd['FOV_ID']==fov]
	utils.generateSpotVisualizationsForTheGivenFOV(rowIdx,colIdx,spotData_singleFOV,dir_FOV_highMag,dir_FOV_highMag_DIC,dir_in_lowMagImages,dir_visualizations_withScatterPlot_others,withScatterPlot=True,scatterPlotBackground=scatterPlotBackground,r=120,scale=1,addBox=True,lowMagFOVSize=1428,scalingFactor=4570.0/1428)
