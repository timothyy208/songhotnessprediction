from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

#load data
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
    
kdata = SelectKBest(f_classif,k=4)
newdata = kdata.fit_transform(data,targets)
print(kdata.get_support(indices=True)) #indices of k best features
traindata,testdata,traintargets,testtargets = train_test_split(
        newdata,targets,test_size=0.2,train_size=0.8)

C_l=[]
acc_mean=[]
Eout=[]
pre_final=[]
rec_final=[]
f_final=[] 
for n in range(-10,11):
    C_n=float(2**(0.5*n))
    C_l.append(np.log2(C_n))
    acc_l=[]
    pre=[]
    rec=[]
    f_1=[]
    for _ in range(100):       
        lr = LogisticRegression(solver='liblinear',C=C_n,
                           multi_class='ovr')   
    
        lr.fit(traindata,traintargets)
        y_pred=lr.predict(testdata)
        mtx = confusion_matrix(testtargets,y_pred)
        precision = mtx[1][1] / (mtx[1][1]+mtx[0][1])
        recall = mtx[1][1] / (mtx[1][1]+mtx[1][0])
        f1 = precision * recall * 2 / (precision + recall)
        pre.append(precision)
        rec.append(recall)
        f_1.append(f1)
        
        acc=np.mean(cross_val_score(lr,traindata,traintargets,cv=10))
        acc_l.append(acc)
    accm=np.mean(acc_l)
    acc_mean.append(accm)
    pre_mean=np.mean(pre)
    pre_final.append(pre_mean)
    rec_mean=np.mean(rec)
    rec_final.append(rec_mean)
    f_mean=np.mean(f_1)
    f_final.append(f_mean)

bn=range(-10,11)[rec_final.index(max(rec_final))]
precision=pre_final[rec_final.index(max(rec_final))]
recall=max(rec_final)
f1=f_final[rec_final.index(max(rec_final))]
ein=1-acc_mean[rec_final.index(max(rec_final))]
best_lr = LogisticRegression(solver='liblinear',C=float(2**(0.5*bn)),
                           multi_class='ovr').fit(traindata,traintargets)
score=np.mean(cross_val_score(best_lr,testdata,testtargets,cv=10))    
#y_pred=best_lr.predict(testdata)
#mtx = confusion_matrix(testtargets,y_pred)
#precision = mtx[1][1] / (mtx[1][1]+mtx[0][1])
#recall = mtx[1][1] / (mtx[1][1]+mtx[1][0])
#f1 = precision * recall * 2 / (precision + recall)

plt.figure()
plt.plot(C_l,acc_mean)
plt.xlabel('Accuracy')
plt.xlabel('Log_2(C)')

plt.figure()
plt.plot(C_l,f_final)
plt.ylabel('F1')
plt.xlabel('Log_2(C)')

print('Log_2(C):',0.5*bn)
print('E_in:',ein)
print('E_out:',1-score)
print('Precision:',precision)
print('Recall:',recall)
print('F1:',f1)

f.close()