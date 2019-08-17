#import pickle
#import json
import pandas as pd   #for csv files
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.utils import to_categorical
from keras.models import load_model
from keras.models import Sequential
from keras import optimizers

# defining a window function for segmentation purposes
def windows(data,size):
    start = 0
    while start < data.count():
        yield int(start), int(start + size)
        start += (size/2)   # (size/2) for double dataset

def segment_signal(data, window_size = 24):
    segments = np.empty((0,window_size,3))
    labels= np.empty((0))
    for (start, end) in windows(data['time-step'],window_size):
        x = data['x-axis'][start:end]
        y = data['y-axis'][start:end]
        z = data['z-axis'][start:end]
        if(len(data['time-step'][start:end])==window_size):
            segments = np.vstack([segments,np.dstack([x,y,z])])
            labels = np.append(labels,stats.mode(data['user_id'][start:end])[0][0])
    return segments, labels

def loadLocalDataset(path):
    try:
        columnNames = ['user_id','time-step','x-axis','y-axis','z-axis']
        data = pd.read_csv(path, header=None, names=columnNames, na_values=';')
        return data
    except Exception as e:
        print("\n", e, "Error trying to Load data from", path)
        return -1

def featureNormalize(dataset):
    mu = np.mean(dataset,axis=0)
    sigma = np.std(dataset,axis=0)
    return (dataset-mu)/sigma

def cnnModel(numFilters, kernelSize, numOfRows, numOfColumns, poolingWindowSz, dropOutRatio, numNueronsFCL1, numNueronsFCL2, numClasses):
    model = Sequential()
    model.add(Conv2D(numFilters, (kernelSize,kernelSize),input_shape=(numOfRows, numOfColumns,1),activation='relu'))
    model.add(MaxPooling2D(pool_size=(poolingWindowSz,poolingWindowSz),padding='valid'))
    model.add(Dropout(dropOutRatio))
    model.add(Flatten())
    model.add(Dense(numNueronsFCL1, activation='relu'))
    model.add(Dense(numNueronsFCL2, activation='relu'))
    model.add(Dense(numClasses, activation='softmax'))

    adam = optimizers.Adam(lr=0.001, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model

# # # #   MAIN   # # # #
random_seed = 456
np.random.seed(random_seed)

dataset = loadLocalDataset("dataset.csv")
dataset["x-axis"] = featureNormalize(dataset["x-axis"])
dataset["y-axis"] = featureNormalize(dataset["y-axis"])
dataset["z-axis"] = featureNormalize(dataset["z-axis"])

#each segmeent as a 2D image (24 X 3)
segments, labels = segment_signal(dataset)
labels = np.asarray(pd.get_dummies(labels), dtype=np.int8)

numOfRows = segments.shape[1]
numOfColumns = segments.shape[2]

numChannels = 1
numFilters = 128 # number of filters in Conv2D layer
kernelSize = 2 # kernel size of the Conv2D layer
poolingWindowSz = 2 # max pooling window size

# number of filters in fully connected layers
numNueronsFCL1 = 128
numNueronsFCL2 = 128

trainSplitRatio = 0.8 # split ratio for test and validation
Epochs = 30
batchSize = 10
numClasses = labels.shape[1] # number of total clases
dropOutRatio = 0.2 # dropout ratio for dropout layer

# reshaping the data for network input
reshapedSegments = segments.reshape(segments.shape[0], numOfRows, numOfColumns,1)


# splitting in training and testing data
trainSplit = np.random.rand(len(reshapedSegments)) < trainSplitRatio
trainX = reshapedSegments[trainSplit]
testX = reshapedSegments[~trainSplit]
trainX = np.nan_to_num(trainX)
testX = np.nan_to_num(testX)
trainY = labels[trainSplit]
testY = labels[~trainSplit]

# Building the model
model = cnnModel(numFilters, kernelSize, numOfRows, numOfColumns, poolingWindowSz, dropOutRatio, numNueronsFCL1, numNueronsFCL2, numClasses)
print("Model:")
model.summary()

print("Start training ...\n")
model.fit(trainX,trainY, validation_split=1-trainSplitRatio, epochs=Epochs, batch_size=batchSize, verbose=2)

#print("\nStart evaluating ...")
#score = model.evaluate(testX, testY, verbose=1)
#print("\nLoss:", score[0], "\nAccuracy:", score[1])
#print('Baseline Error: %.2f%%' %(100-score[1]*100))

model.save('model.h5')
np.save('testData.npy', testX)
np.save('groundTruth.npy', testY)
