from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import sklearn.metrics as skm


np.seterr(divide='ignore', invalid='ignore')
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
resin = np.zeros((10,12))
resout = np.zeros((10,12))

data = []
targets = []
f = open("resample.txt","r")
for line in f:
    temp = line.split(",")
    tempdata = [float(x) for x in temp[0:-1]]
    temptarget = float(temp[-1])
    data.append(tempdata)
    targets.append(temptarget)

odata = []
ogtargets = []
names=[]
g = open("datamaxfeatures.txt","r")
for line in g:
    temp = line.split(",")
    tempdata = [float(x) for x in temp[1:-1]]
    temptarget = float(temp[-1])
    odata.append(tempdata)
    ogtargets.append(temptarget)
    names.append(temp[0])

    
odata = np.nan_to_num(odata)
kdata=SelectKBest(f_classif,k=4)
newdata = kdata.fit_transform(data,targets)
print(kdata.get_support(indices=True))

ogdata = kdata.fit_transform(odata,ogtargets)
print(kdata.get_support(indices=True))


acc_l=[]
pre=[]
rec=[]
f_1=[]
eout=[]
for _ in range(100):       
    lr = LogisticRegression(solver='liblinear',
                       multi_class='ovr')   

    lr.fit(newdata,targets)
    y_pred=lr.predict(ogdata)
    precision = skm.precision_score(ogtargets,y_pred)
    recall = skm.recall_score(ogtargets,y_pred)
    f1 = precision * recall * 2 / (precision + recall)
    pre.append(precision)
    rec.append(recall)
    f_1.append(f1)
    
    acc=np.mean(cross_val_score(lr,newdata,targets,cv=10))
    acc_l.append(acc)
    
    score=np.mean(cross_val_score(lr,ogdata,ogtargets,cv=10))
    eout.append(1-score)
    
accm=np.mean(acc_l)
pre_mean=np.mean(pre)
rec_mean=np.mean(rec)
f_mean=np.mean(f_1)
eout_mean=np.mean(eout)
 


print('E_in:',1-accm)
print('E_out:',eout_mean)
print('Precision:',pre_mean)
print('Recall:',rec_mean)
print('F1:',f_mean)

f.close()
g.close()