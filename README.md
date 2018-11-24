# K-Nearest Neighbors and Nearest-Centroid Classification Algorithms in Python

##### Implementation of classification algorithms: K-Nearest Neighbors and Centroid Classification method in Python. Implemented algorithms are used to classify handwritten-characters and ATNT Face-Dataset.
=================================================================================
### Usage:
- ###### >> python KNN_Centroid.py
-  ###### Datasets: ATNTFaceImages400.txt, HandWrittenLetters.txt
### Helper Subroutines written:

  - ##### pickDataClass(filename, class_ids) : 
  - - ##### filename: char_string specifing the data file to read. For example, 'ATNT_face_image.txt'
  - - ##### class_ids:  array that contains the classes to be pick. For example: (3, 5, 8, 9)
  - - ##### Returns: an multi-dimension array containing the data (both attribute vectors and class labels) of the selected classes
  - - ##### We can use this subroutine to pick a small part of the data for analysis. For example for handwrittenletter data, we can pick classes "C" and "F" for a 2-class experiment. Or we pick "A,B,C,D,E" for a 5-class experiment. 
  #####
  
  - ##### splitData2TestTrain(filename, num_per_class,  test_instances) : 
  - - ##### filename: char_string specifing the data file to read. This can also be an array containing input data.
  - - ##### num_per_class: number of data instances in each class (we assume every class has the same number of data instances)
  - - ##### test_instances: the data instances in each class to be used as test data.
  - - ##### Return/output: Training_attributeVector(trainX), Training_labels(trainY), Test_attributeVectors(testX), Test_labels(testY)
  - - ##### Example: splitData2TestTrain('Handwrittenletters.txt', 39, 1:20). Use entire 26-class handwrittenletters data. Each class has 39 instances. In every class, first 20 images for testing, remaining 19 images for training
  #####
  
  - ##### splitData2TestTrain(filename, num_per_class,  test_instances) : 
  - - ##### filename: char_string specifing the data file to read. This can also be an array containing input data.
  - - ##### num_per_class: number of data instances in each class (we assume every class has the same number of data instances)
  - - ##### test_instances: the data instances in each class to be used as test data.
  - - ##### Return/output: Training_attributeVector(trainX), Training_labels(trainY), Test_attributeVectors(testX), Test_labels(testY)
  - - ##### Example: splitData2TestTrain('Handwrittenletters.txt', 39, 1:20). Use entire 26-class handwrittenletters data. Each class has 39 instances. In every class, first 20 images for testing, remaining 19 images for training
  #####
### Classifiers:
- ##### K-Nearest-Neighbors: KNearestNeighbors(trainX, trainY, testX, k)
- ##### Centroid: NearestCentroidClf(trainX, trainY, testX)
- -   ##### Accuracy achieved with this implemented KNN model is 97.567% for ATNTFaceImages400.txt.
