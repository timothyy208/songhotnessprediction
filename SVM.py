import numpy as np
from sys import argv
from numpy import mean, sqrt, std
import math
import random
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import sklearn.metrics as skm
import sklearn.preprocessing as skp
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

# Load data set
if len(argv)==1:
   datafilename = 'max.txt'
   #datafilename = 'filtered.txt'
   datafilename1 = 'resample.txt'
else:
    datafilename = argv[-1]
    datafilename1 = argv[-1]
#print('Loading', datafilename);
dataset = np.loadtxt(datafilename, delimiter=',')
(numSamples, numFeatures) = dataset.shape

dataset1 = np.loadtxt(datafilename1, delimiter=',')
(numSamples1, numFeatures1) = dataset1.shape

Einmeans = []
Eoutmeans = []
premeans = []
recallmeans = []
fmeans = []

#data is the original dataset
#data1 is the resample dataset
fea = 13
data = dataset[:,range(fea)].reshape((numSamples, fea))
data1 = dataset1[:,range(fea)].reshape((numSamples1, fea))

data = np.nan_to_num(data)
data1 = np.nan_to_num(data1)

labels = dataset[:, 15].reshape((numSamples,))
relabels = dataset1[:, 15].reshape((numSamples1,))

kdata = SelectKBest(f_classif,k=13)
resamples = kdata.fit_transform(data1,relabels)
odata = kdata.fit_transform(data,labels)

for fea in range(1, 16):

#for fea in range(13,14):
      #traindata,testdata,traintargets,testtargets = train_test_split(newdata, targets, test_size=0.2,train_size=0.8)
      #[ 0  1  2  3  4  5  6  7  8  9 10 11 12]
      #print(kdata.get_support(indices=True))
      
      labelleng = len(labels)
      nohit = []
      hit = []
      for i in range(0, labelleng):
          if labels[i] == -1:
              datum = data[i].tolist()
              nohit.append(datum)
          else:
              datum1 = data[i].tolist()
              hit.append(datum1)
                 
      nohitlength = len(nohit)
          
      hitlength = len(hit)
      hitlabels = []
      for o in range(0, hitlength):
          hitlabels.append(1)
       
      resam = 612
      #resam = 100
      renohit = random.sample(nohit,k=resam)
      nohitlabels = []
      for l in range(0, resam):
          nohitlabels.append(-1)
      
      #This is for hyperparameter selection
      #resams = 200
      #rehit = random.sample(hit,k=resams)
      #rehitlabels = []
      #for m in range(0, resams):
        # rehitlabels.append(1)
      
      resamples = hit + renohit
      relabels = hitlabels + nohitlabels
      
      #This is for hyperparameter selection
      #resamples = rehit + renohit
      #relabels = rehitlabels + nohitlabels
      
      c = list(zip(resamples, relabels))
      random.shuffle(c)
      resamples, relabels = zip(*c)
      X_train, X_test, y_train, y_test = train_test_split(resamples, relabels, test_size=0.2, train_size=0.8)
      #X_train, X_test, y_train, y_test = train_test_split(resamples, relabels, test_size=0.5, train_size=0.5)
   
      Eins = []
      Eouts = []
      pres = []
      recalls = []
      fs = []   
      for k in range(0,100):

         clf = SVC(gamma='auto')
   
         #This is for training the splitted dataset
         #clf.fit(X_train, y_train)
         #Ein = 1-cross_val_score(clf, X_train, y_train, cv=10).mean()
         #Eout = 1-cross_val_score(clf, X_test, y_test, cv=10).mean()
         #pred = clf.predict(X_test)
         #mtx = confusion_matrix(y_test,pred)
         #precision = mtx[1][1] / (mtx[1][1]+mtx[0][1])
         #recall = mtx[1][1] / (mtx[1][1]+mtx[1][0])
         
        # This is for training the original dataset
         
         SVM = clf.fit(resamples, relabels)
         Ein = 1-cross_val_score(SVM, resamples, relabels, cv=10).mean()
         Eout = 1-cross_val_score(SVM, odata, labels, cv=10).mean()
         pred = clf.predict(odata)
         precision = skm.precision_score(labels,pred)
         recall = skm.recall_score(labels,pred)
         
         f1 = precision * recall * 2 / (precision + recall)
         Eins.append(Ein)
         Eouts.append(Eout)
         pres.append(precision)
         recalls.append(recall)
         fs.append(f1)
       
      meanEin = sum(Eins)/len(Eins)
      meanEout = sum(Eouts)/len(Eouts)
      meanpre = sum(pres)/len(pres)
      meanrecall = sum(recalls)/len(recalls)
      meanf = sum(fs)/len(fs)

#print("E_in:",meanEin)
#print("E_out:",meanEout)
#print("Precision:",meanpre)
#print("Recall:",meanrecall)
#print("F1:",meanf)
      Einmeans.append(meanEin)
      Eoutmeans.append(meanEout)
      premeans.append(meanpre)
      recallmeans.append(meanrecall)
      fmeans.append(meanf)
      
      #This is for hyperparameter selection of 13 features
      #tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'degree':[1,2,4],
       #              'C': [0.5,1, 2, 10, 100, 1000]},
       #             {'kernel': ['linear'], 'degree':[1,2,4],'C': [0.5,1, 2,10, 100, 1000]}]

     # scores = ['precision', 'recall']
      
      #for score in scores:
      #    print("# Tuning hyper-parameters for %s" % score)
      #    print()
      
      #    clf = GridSearchCV(
      #        SVC(), tuned_parameters, scoring='%s_macro' % score)
      #    clf.fit(X_train, y_train)
      
       #   print("Best parameters set found on development set:")
       #   print()
       #   print(clf.best_params_)
       #   print()
       #   print("Grid scores on development set:")
       #   print()
       #   means = clf.cv_results_['mean_test_score']
       #   stds = clf.cv_results_['std_test_score']
       #   for mean, std, params in zip(means, stds, clf.cv_results_['params']):
       #       print("%0.3f (+/-%0.03f) for %r"
       #             % (mean, std * 2, params))
       #   print()
      
       #   print("Detailed classification report:")
       #   print()
       #   print("The model is trained on the full development set.")
       #   print("The scores are computed on the full evaluation set.")
       #   print()
       #   y_true, y_pred = y_test, clf.predict(X_test)
       #   print(classification_report(y_true, y_pred))
       #   print()

#print("Einmeans",Einmeans)
#print("Eoutmeans",Eoutmeans)
#print("precision",premeans)
#print("recall",recallmeans)
#print("F1",fmeans)
plt.figure()
plt.plot(range(1,16), Einmeans, label='Ein')
plt.plot(range(1,16), Eoutmeans, label = 'Eout')
plt.xlabel('Number of Features')
plt.ylabel('Errors')
plt.legend()

plt.figure()
plt.plot(range(1,16), premeans, label='Precision')
plt.xlabel('Number of Features')
plt.ylabel('Precision')
plt.legend()


plt.figure()
plt.plot(range(1,16), recallmeans, label = 'Recall')
plt.xlabel('Number of Features')
plt.ylabel('Recall')
plt.legend()

plt.figure()
plt.plot(range(1,16), fmeans, label='F1')
plt.xlabel('Number of Features')
plt.ylabel('F1')

plt.legend()

plt.show()
