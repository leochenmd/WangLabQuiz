import pandas as pd
import os
import numpy as np

dir = '/Users/Leo/Documents/Stanford Sensitive/Jenn/BrainTumor/retinanet_output'

files = [f for f in os.listdir(dir) if 'png' in f]

matrix = np.zeros([5, 5])
print(matrix.shape)
print(matrix[0, 0])

def convtoindex(tumortype):
    if tumortype == 'none':
        return 0
    elif tumortype == 'dipg':
        return 1
    elif tumortype == 'ependymoma':
        return 2
    elif tumortype == 'mb':
        return 3
    elif tumortype == 'pilocytic':
        return 4

for i, file in enumerate(files):
    actual = file.split('_')[1]
    predicted = file.split('_')[2]
    predicted = predicted.split('.')[0]
    print(actual, predicted)

    actualindex = int(convtoindex(actual))
    predictedindex = int(convtoindex(predicted))

    print(actualindex)
    print(predictedindex)
    matrix[actualindex, predictedindex] += 1

print(matrix)
print(matrix[1,0])
