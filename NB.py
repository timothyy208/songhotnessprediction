import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import sklearn.metrics as skm
import sklearn.preprocessing as skp
import random
import math
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

np.seterr(divide='ignore', invalid='ignore')
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

ogdata = []
ogtargets = []
names = []
tdata = []
ttargets = []
data = []
targets = []
f = open("datamaxfeatures.txt","r")
for line in f:
    temp = line.split(",")
    tempdata = [float(x) for x in temp[1:-1]]
    temptarget = int(temp[-1])
    ogdata.append(tempdata)
    ogtargets.append(temptarget)
    if temptarget == 1:
        data.append(tempdata)
        targets.append(temptarget)
   
    else:
        names.append(temp[0])
        tdata.append(tempdata)
        ttargets.append(temptarget)








toadd = set()
while len(toadd) != 612:
    toadd.add(random.randint(0,len(tdata)-1))
for i in toadd:
    r=random.randint(0, len(data))
    data.insert(r,tdata[i])
    targets.insert(r,ttargets[i])
    #data.append(tdata[i])
    #targets.append(ttargets[i])





eins=[]
eouts=[]
precisions=[]
recalls=[]
f1s=[]
for i in range(100):
    k = 1
    data = np.nan_to_num(data)
    ogdata = np.nan_to_num(ogdata)
    kdata=SelectKBest(f_classif,k=k)
    newdata = kdata.fit_transform(data,targets)
    print(kdata.get_support(indices=True))
    
    newogdata = SelectKBest(f_classif,k=k).fit_transform(ogdata,ogtargets)
    #X_train=data
    #y_train=targets
    X_train, X_test, y_train, y_test = train_test_split(newdata,targets,test_size=0.2,train_size=0.8)

    clf = GaussianNB()
    clf.fit(X_train,y_train)

    ein = 1-cross_val_score(clf, X_train, y_train, cv=10).mean()
    #eout = 1-cross_val_score(clf, X_test, y_test, cv=10).mean()
    eout = 1-cross_val_score(clf, newogdata, ogtargets, cv=10).mean()
    eins.append(ein)
    eouts.append(eout)
    
    y_pred = clf.predict(newogdata)
    #mtx = confusion_matrix(y_test,y_pred)
    precision = skm.precision_score(ogtargets,y_pred)
    #precision=mtx[1][1]/(mtx[1][1]+mtx[0][1])
    recall = skm.recall_score(ogtargets,y_pred)
    #recall=mtx[1][1]/(mtx[1][1]+mtx[1][0])
    f1= 2*((precision*recall)/(precision+recall))


    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)

    
print("Number of songs predicted to be +1:",list(y_pred).count(1))
print("Number of songs predicted to be -1:",list(y_pred).count(-1))

print('ein:',sum(eins)/len(eins))
print('eout:',sum(eouts)/len(eouts))
print('precision:',sum(precisions)/len(precisions))
print('recall:',sum(recalls)/len(recalls))
print('f1:',sum(f1s)/len(f1s))
    
##Y=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
##plt.plot(Y,recallplot,'b-')
##
##plt.title('Recall')
##
##plt.xlabel('Number of Features')
##
##plt.show()
    
   
      
f.close()
