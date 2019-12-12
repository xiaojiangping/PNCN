# -*- coding: UTF-8 -*-
import operator
import sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


# Import data file.
# In our experiment, we importe data files in CSV format.
# Extract features and labels of data.
csvData = np.loadtxt("LOTAAS_1_Lyon_Features (8).csv", dtype=np.str, delimiter=",")
featuresNum = csvData.shape[1] - 1
data = csvData[0:,0:featuresNum].astype(np.float)
label = csvData[0:,featuresNum].astype(np.int)


# Some common functions, including normalization, distance.
def autoNorm(dataSet):
    # Obtain the maximum, minimum values and change range of data.
    maxValues = dataSet.max(0)
    minValues = dataSet.min(0)
    ranges = maxValues - minValues

    # Number of rows and columns of dataset.
    lines = dataSet.shape[0]
    rows = dataSet.shape[1]

    # Subtract the minimum value from the original value and divide it by the difference between the maximum value and the minimum value to get the normalized data.
    normDataSet = np.zeros((lines, rows))
    normDataSet = dataSet - np.tile(minValues, (lines, 1))
    normDataSet = normDataSet / np.tile(ranges, (lines, 1))
    
    return normDataSet
	
def distances(intX, dataSet):
    # Number of rows in the dataset.
    lines = dataSet.shape[0]
    
    # Repeat intX once (horizontally) in column vector direction, intX times (vertically) in row vector direction, and square after feature subtraction.
    diffMat = np.tile(intX, (lines, 1)) - dataSet
    sqDiffMat = diffMat**2
    
    # Calculate distance.
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    
    return distances
	
	
# The PNCN algorithm.
# Some defined functions, including nearest neighbor, centroid and pseudo nearest neighbor.
def nearestNeighbor(intX, dataSet):
    distance = distances(intX, dataSet)
    sortedDistIndices = distance.argsort()
    nearestNeighbor = dataSet[sortedDistIndices[0]]

    return nearestNeighbor

def centroid(intX, nonCentroidDataSet, centroidDataSet):
    # Calculate the mean value of the sum of the existing centroids and other points.
    CentroidDataSet = centroidDataSet.sum(axis=0)
    NewDataSet = (nonCentroidDataSet + CentroidDataSet) / (len(centroidDataSet) + 1)
    
    # Calculate the distance and return the sorted index value of the elements in the distance.
    distance = distances(intX, NewDataSet)    
    sortedDistIndices = distance.argsort()

    # Return to centroid.
    centroid = nonCentroidDataSet[sortedDistIndices[0]]
            
    return centroid

def pncn(intX, nonCentroidDataSet, k):
    centroidDataSet = []
           
    # Looking for k centroids.
    for i in range(k):   
        centroidDataSet = np.array(centroidDataSet)        
        NewCentroid = centroid(intX, nonCentroidDataSet, centroidDataSet)
        nonCentroidDataSet = nonCentroidDataSet.tolist() 
        NewCentroid = NewCentroid.tolist()      
        centroidDataSet = centroidDataSet.tolist() 
        
        nonCentroidDataSet.remove(NewCentroid)
        centroidDataSet.append(NewCentroid)
 
        nonCentroidDataSet = np.array(nonCentroidDataSet)   

    centroidDataSet = np.array(centroidDataSet) 
    
    # Computing pseudo near center neighbor 
    distance = distances(intX, centroidDataSet)
    sortedDistIndices = distance.argsort()
    
    ncn = centroidDataSet[sortedDistIndices]
    localMeanVector = np.zeros((k,8))
    for i in range(k):
        localMeanVector[i] = np.mean(ncn[0:i+1], axis=0) 
        
    lmvDistance = distances(intX, localMeanVector)
    sortedLmvDistIndices = lmvDistance.argsort()        
              
    pncn = 0.0
    
    for i in range(k):
        weight = 1 / (i + 1)        
        t = sortedLmvDistIndices[i]              
        pncn= pncn + weight * lmvDistance[t]   
		
    return pncn

# Pncn classifier
def pncnClassifier(intX, dataTrain, labelTrain, k):
    
    pulse = np.where(labelTrain == 1)
    nonPulse = np.where(labelTrain == 0)
    
    pulseDataTrain = dataTrain[pulse]
    pulseLabelTrain = labelTrain[pulse]        
    nonPulseDataTrain = dataTrain[nonPulse]
    nonPulseLabelTrain = labelTrain[nonPulse]
    
    pulse = pncn(intX, pulseDataTrain, k)
    nonPulse = pncn(intX, nonPulseDataTrain, k)

    result = 0
    if nonPulse >= pulse:
         result = 1
    else:
         result = 0

    return result

# Pncn classifier's performance evaluation
def pncnPerformance(dataTrain, dataTest, labelTrain, labelTest, k):

    # Confusion matrix count
    tpCount = 0
    fpCount = 0
    tnCount = 0
    fnCount = 0
    
    for i in range(len(labelTest)):       
        result = pncnClassifier(dataTest[i,:], dataTrain, labelTrain, k)

        if result == 1 and labelTest[i] == 1:
            tpCount = tpCount + 1            
        if result == 1 and labelTest[i] == 0:
            fpCount = fpCount + 1
        if result == 0 and labelTest[i] == 0:
            tnCount = tnCount + 1
        if result == 0 and labelTest[i] == 1:
            fnCount = fnCount + 1  
        
    accuracy = (tpCount + tnCount) / float(len(labelTest))          
    precision = tpCount / float(tpCount + fpCount) 
    recall = tpCount / float(tpCount + fnCount)     
    FPR = fpCount / float(fpCount + tnCount)     
    
    return accuracy, precision, recall, FPR
	
# Data normalization, stratified sampling.
# n this experiment, we adopt 5 fold stratification cross validation.
normDataMat = autoNorm(data)
sfolder = StratifiedKFold(n_splits=5, shuffle=True)


Accuracy = np.zeros((5,15))
Precision = np.zeros((5,15))
Recall = np.zeros((5,15))
FPR = np.zeros((5,15))
GMean = np.zeros((5,15))
Fscore = np.zeros((5,15))

foldCount = 0
for trainIndex, testIndex in sfolder.split(normDataMat, label):
    
    dataTrain = normDataMat[trainIndex]
    dataTest = normDataMat[testIndex]
    labelTrain = label[trainIndex]
    labelTest = label[testIndex]
    
    for k in range(15):
        k = k + 1
        pncnAccuracy, pncnPrecision, pncnRecall, pncnFPR = pncnPerformance(dataTrain, dataTest, labelTrain, labelTest, k)              
        
        Accuracy[foldCount, k] = pncnAccuracy 
        Precision[foldCount, k] = pncnPrecision
        Recall[foldCount, k] = pncnRecall
        FPR[foldCount, k] = pncnFPR  
        GMean[foldCount, k] = (pncnRecall*(1-pncnFPR))**0.5
        Fscore[foldCount, k] = 2*((pncnPrecision * pncnRecall)/(pncnPrecision + pncnRecall))

    foldCount = foldCount + 1

avgAccuracy = np.mean(Accuracy, axis=0)
avgPrecision = np.mean(Precision, axis=0)
avgRecall = np.mean(Recall, axis=0)
avgFPR = np.mean(FPR, axis=0)
avgGMean = np.mean(GMean, axis=0)
avgFscore = np.mean(Fscore, axis=0) 