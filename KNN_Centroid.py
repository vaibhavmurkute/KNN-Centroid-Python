# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 20:22:33 2018

@author: Vaibhav Murkute
"""
import pandas as pd
import numpy as np
import math
import operator
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.linear_model import LinearRegression

def Main():
    picked_data = pickDataClass("HandWrittenLetters.txt", [1,22,19,5,6,7,2,3])
    np.savetxt("Picked_dataset.csv", picked_data.T, delimiter=',', fmt='%d')
    print("File Saved!")
    
    trainX, trainY, testX, testY = splitData2TestTrain("HandWrittenLetters.txt",39, 25)
    saveTrainTestFile(trainX, trainY, testX, testY)
    print("Files saved!")
    
    result=[]
    k = 1   # K is set to 1 for 1-KNN
    result = KNearestNeighbors(trainX, trainY, testX, k)
    np.savetxt("KNN-3-Predictions.txt", result, delimiter=',', fmt='%d')
    print(getAccuracy(testY.tolist(), result))
    
    
    result=[]
    result = NearestCentroidClf(trainX, trainY, testX)
    np.savetxt("myCentroid-Predictions.txt", result, delimiter=',', fmt='%d')
    print(getAccuracy(testY.tolist(), result))
    
    #=======================================================
    svm_clf = svm.SVC()
    svm_clf.fit(trainX, trainY)
    result = svm_clf.predict(testX)
    np.savetxt("SVM-Predictions.txt", result, delimiter=',', fmt='%d')
    print(getAccuracy(testY.tolist(), result))
    #=======================================================
    
def rotateData(file_name):
    #file_data = pd.read_csv(file_name)
    file_data = np.genfromtxt(file_name,delimiter=',')
    return file_data.T

def pickDataClass1(rotated_array, class_ids):
    picked_data = []
    for row in rotated_array:
        if(row[0] in class_ids):
            picked_data.append(row)
    return picked_data

def pickDataClass(file_name, class_ids):
    if type(file_name) == np.ndarray:
        file_data = file_name
    else:
        file_data = np.genfromtxt(file_name, delimiter=',')
    picked_data = np.array([])
    for i in range(len(file_data[0])):
        if(file_data[0][i] in class_ids):
            if(picked_data.size == 0):
                picked_data = file_data[:,i]
            else:
                picked_data = np.vstack((picked_data, file_data[:,i]))
    
    return picked_data


def splitData2TestTrain(filename, num_per_class,  test_instances):
    test_percent = (test_instances/num_per_class)
    test_x = np.array([])
    test_y = np.array([])
    train_x = np.array([])
    train_y = np.array([])
    #class_labels = set(file_data[0,:])
    class_labels = set([1,22,19,5,6,7,2,3])
    for label in class_labels:
        label_data = pickDataClass(filename, [label])
        
        y = label_data[:,0]
        x = label_data[:,1:]
        
        trainX, testX, trainY, testY = train_test_split(x, y, test_size=test_percent, random_state=0)
        
        if train_x.size == 0:
            train_x = trainX
        else:
            train_x = np.vstack((train_x, trainX))
            
        if train_y.size == 0:
            train_y = trainY
        else:
            train_y = np.hstack((train_y, trainY))
            
        if test_x.size == 0:
            test_x = testX
        else:
            test_x = np.vstack((test_x, testX))
            
        if test_y.size == 0:
            test_y = testY
        else:
            test_y = np.hstack((test_y, testY))
    
    return train_x, train_y, test_x, test_y

def splitData2TestTrain1(filename, num_per_class,  test_instances):
    dataset = pd.read_csv(filename)
    dataset = dataset.T
    x = dataset.iloc[:,0].values
    y = dataset.iloc[:,1:].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_instances, random_state=0)
    
    return x_train, y_train, x_test, y_test

def saveTrainTestFile(trainX, trainY, testX, testY):
    train_x = np.array(trainX)
    train_y = np.array(trainY)
    test_x = np.array(testX)
    test_y = np.array(testY)
    
    np.savetxt('trainX.csv', train_x, delimiter=',', fmt='%d')
    np.savetxt('trainY.csv', train_y, delimiter=',', fmt='%d')
    np.savetxt('testX.csv', test_x, delimiter=',', fmt='%d')
    np.savetxt('testY.csv', test_y, delimiter=',', fmt='%d')
    
def letter_2_digit_convert(word):
    num_list = [(ord(character) - (ord('a')-1)) for character in (word).strip().lower()]
    return num_list

def euclideanDistance(data1, data2, feature_length):
	distance = 0
	for x in range(feature_length):
		distance += pow((data1[x] - data2[x]), 2)
	return math.sqrt(distance)

def getNearestNeighbor(trainingSet, trainingLabels, testInstance, k):
    distances = []
    length = len(testInstance)
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingLabels[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = {}
    for x in range(k):
        if(bool(neighbors) and distances[x][0] in neighbors.keys()):
            neighbors[distances[x][0]] += 1
        else:
            neighbors[distances[x][0]] = 1
    max_key = max(neighbors.items(), key=operator.itemgetter(1))[0]
    return max_key

def KNearestNeighbors(trainX, trainY, testX, k):
    predictions = []
    for i in range(len(testX)):
        predictions.append(getNearestNeighbor(trainX, trainY, testX[i], k))
    return predictions

def getAccuracy(testSet, predictions):
    correct_predictions = 0
    testset_size = len(testSet)
    for x in range(testset_size):
        if testSet[x] == predictions[x]:
            correct_predictions += 1
    return 100.0 * (correct_predictions/testset_size)

def getCentroids(trainX, trainY):
    file_data = np.vstack((trainY, trainX.T))
    all_centroids = {}
    class_labels = set(trainY)
    for label in class_labels:
        label_data = pickDataClass(file_data, [label])
        centroid = np.array([])
        for i in range(len(label_data[0])):
            column = label_data[:,i]
            column_sum = np.sum(column)
            column_centroid = 1.0*(column_sum / len(column))
            if centroid.size == 0:
                centroid = np.array([column_centroid])
            else:
                centroid = np.hstack((centroid, np.array([column_centroid])))
        all_centroids[label] = centroid
    
    return all_centroids

def getNearestCentroid(all_centroids, testInstance):
    distances = []
    length = len(testInstance)
    labels = all_centroids.keys()
    for key in labels:
        dist = euclideanDistance(testInstance, all_centroids[key], length)
        distances.append((key, dist))
    distances.sort(key=operator.itemgetter(1))
    
    if(len(distances) > 0):
        return distances[0][0]
    else:
        return None
    
def NearestCentroidClf(trainX, trainY, testX):
    all_centroids = getCentroids(trainX, trainY)
    predictions = []
    for i in range(len(testX)):
        predictions.append(getNearestCentroid(all_centroids, testX[i]))
    return predictions

def Cross_Validation(clf, x, y, num_folds):
    if isinstance(clf, LinearRegression):
        cv_accuracy = cross_val_score(clf, x, y, cv=num_folds, scoring='neg_mean_squared_error')
    else:
        cv_accuracy = cross_val_score(clf, x, y, cv=num_folds, scoring='accuracy')
    print("Fold \t Accuracy")
    for i in range(len(cv_accuracy)):
        print("\n{} \t {}".format((i+1),cv_accuracy[i]))

    print("\n")
    print("Average Accuracy : {}".format(cv_accuracy.mean()))

if(__name__ == '__main__'):
    Main()