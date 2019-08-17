# -*- coding: utf-8 -*-
"""
This script is written to evaluate a pretrained model saved as  model.h5 using 'testData.npy'
and 'groundTruth.npy'. This script reports the error as the cross entropy loss in percentage
and also generated a png file for the confusion matrix.
"""

from keras.models import load_model
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import os

os.environ['QT_PLUGIN_PATH'] = ''
def plot_cm(cM, labels, title):
    # normalizing the confusionMatrix
    cmNormalized = np.around((cM/cM.sum(axis=1)[:,None])*100, 2)

    fig = plt.figure()
    plt.imshow(cmNormalized, interpolation=None, cmap=plt.cm.Blues)
    plt.colorbar()
    plt.clim(0,100)

    # assiging the title, x and y labels
    plt.xlabel('Predicted Values')
    plt.ylabel('Ground Truth')
    plt.title(title + '\n%age confidence')

    # defining the ticks for the x and y axis
    plt.xticks(range(len(labels)), labels, rotation = 60)
    plt.yticks(range(len(labels)), labels)

    width, height = cM.shape
    print('Accuracy for each class is given below.')
    for predicted in range(width):
        for real in range(height):
            color = 'black'
            if(predicted == real):
                color = 'white'
                print(labels[predicted].ljust(12)+ ':', cmNormalized[predicted,real], '%')
            plt.gca().annotate(
                    '{:d}'.format(int(cmNormalized[predicted,real])), xy=(real, predicted),
                    horizontalalignment='center', verticalalignment='center', color=color)

    plt.tight_layout()
    fig.savefig(title +'.png')


# # # #   MAIN   # # # #
model = load_model('model.h5')
test_x = np.load('testData.npy')
groundTruth = np.load('groundTruth.npy')

# EVALUATION SECTION
print("\nStart evaluating ...")
score = model.evaluate(test_x, groundTruth, verbose=1)
print("\nLoss:", score[0], "\nAccuracy:", score[1])
print('Baseline Error: %.2f%%' %(100-score[1]*100))

# CONFUSION MATRIX SECTION
labels = ['A','B','C','D','E','F','G','H','I','J','K',
          'L','M','N','O','P','Q','R','S','T','U','V']

predictions = model.predict(test_x, verbose=2)
predictedClass = np.zeros((predictions.shape[0]))
groundTruthClass = np.zeros((groundTruth.shape[0]))

for instance in range (groundTruth.shape[0]):
    predictedClass[instance] = np.argmax(predictions[instance,:])
    groundTruthClass[instance] = np.argmax(groundTruth[instance,:])

cm = metrics.confusion_matrix(groundTruthClass, predictedClass)
plot_cm(cm, labels, 'Confusion Matrix')
