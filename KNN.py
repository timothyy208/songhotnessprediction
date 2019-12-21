import sklearn.neighbors as skn
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.neural_network import MLPClassifier
import sklearn.metrics as skm
import sklearn.preprocessing as skp
import random
import matplotlib.pyplot as plt
np.seterr(divide='ignore', invalid='ignore')
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


# lists to hold the data and targets for the unbalanced dataset
unbalancedData = []
unbalancedTargets = []

# lists to hold the data and targets for the resampled dataset
resampledData = []
resampledTargets = []

#populating the unbalanced lists
f = open("datamaxfeatures.txt","r")
for line in f:
    temp = line.split(",")
    tempdata = [float(x) for x in temp[1:-1]]
    temptarget = float(temp[-1])
    unbalancedData.append(tempdata)
    unbalancedTargets.append(temptarget)
f.close()

#populating the resampled lists
f = open("resample.txt","r")
for line in f:
    temp = line.split(",")

    tempdata = [float(x) for x in temp[:-1]]
    temptarget = float(temp[-1])
    resampledData.append(tempdata)
    resampledTargets.append(temptarget)





# lists to help calculate the average accuracy measures
eins = []
eouts = []
precisions = []
recalls = []
f1s = []

for z in range(100):

    unbalancedData = np.nan_to_num(unbalancedData)
    resampledData = np.nan_to_num(resampledData)

    # selecting features
    kdata = SelectKBest(f_classif,k=4)
    kUnbalancedData = kdata.fit_transform(unbalancedData,unbalancedTargets)
    kResampledData = kdata.fit_transform(resampledData,resampledTargets)


    #splitdata = train_test_split(newdata,targets,test_size=0.2,train_size=0.8)

    # fitting the classifier
    knn = skn.KNeighborsClassifier(n_neighbors=9)
    knn.fit(kResampledData,resampledTargets)

    # calculating error
    ein = 1-cross_val_score(knn, kResampledData, resampledTargets, cv=10).mean()
    eout = 1-cross_val_score(knn, kUnbalancedData, unbalancedTargets, cv=10).mean()

    eins.append(ein)
    eouts.append(eout)
    
    # predicting labels for unbalanced data
    pred = knn.predict(kUnbalancedData)
    
    # calculating accuracy measures
    precision = skm.precision_score(unbalancedTargets,pred)
    recall = skm.recall_score(unbalancedTargets,pred)
    f1 = precision * recall * 2 / (precision + recall)

    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)



print("Number of songs predicted to be +1:",list(pred).count(1))
print("Number of songs predicted to be -1:",list(pred).count(-1))
print('ein:',sum(eins)/len(eins))
print('eout:',sum(eouts)/len(eouts))
print('precision:',sum(precisions)/len(precisions))
print('recall:',sum(recalls)/len(recalls))
print('f1:',sum(f1s)/len(f1s))


f.close()