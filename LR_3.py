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

In=[]
Out=[]
pre_final=[]
rec_final=[]
f_final=[]    
  
for k in range(1,16):
    kdata= SelectKBest(f_classif,k=k)
    newdata = kdata.fit_transform(data,targets)
    traindata,testdata,traintargets,testtargets = train_test_split(
            newdata,targets,test_size=0.2,train_size=0.8)
    
    print(kdata.get_support(indices=True))
    
    E_in=[]
    E_out=[] 
    pre=[]
    rec=[]
    f_1=[]
    for _ in range(100):
        lr = LogisticRegression(solver='liblinear',
                               multi_class='ovr')       
        lr.fit(traindata,traintargets)
        acc=np.mean(cross_val_score(lr,traindata,traintargets,cv=10))
        E_in.append(1-acc) 
        
        score = np.mean(cross_val_score(lr,testdata,testtargets,cv=10))   
        E_out.append(1-score)

        
        y_pred=lr.predict(testdata)
        mtx = confusion_matrix(testtargets,y_pred)
        precision = mtx[1][1] / (mtx[1][1]+mtx[0][1])
        recall = mtx[1][1] / (mtx[1][1]+mtx[1][0])
        f1 = precision * recall * 2 / (precision + recall)
        pre.append(precision)
        rec.append(recall)
        f_1.append(f1)

        
    ein=np.mean(E_in)
    In.append(ein)
    eout=np.mean(E_out)
    Out.append(eout)
    pre_mean=np.mean(pre)
    pre_final.append(pre_mean)
    rec_mean=np.mean(rec)
    rec_final.append(rec_mean)
    f_mean=np.mean(f_1)
    f_final.append(f_mean)

plt.title('E_in and E_out')
plt.plot(range(1,16),In,label='E_in')
plt.plot(range(1,16),Out,label='E_out')
plt.xlabel('number of features')
plt.legend()

plt.figure()
plt.title('Precision')
plt.plot(range(1,16),pre_final,label='Precision')
plt.xlabel('number of features')
plt.legend()

plt.figure()
plt.title('Recall')
plt.plot(range(1,16),rec_final,label='Recall')
plt.xlabel('number of features')
plt.legend()

plt.figure()
plt.title('F1')
plt.plot(range(1,16),f_final,label='F1')
plt.xlabel('number of features')
plt.legend()

precision=pre_final[f_final.index(max(f_final))]
f1=max(f_final)
recall=rec_final[f_final.index(max(f_final))]
EIN=In[f_final.index(max(f_final))]
EOUT=Out[f_final.index(max(f_final))]

print('number of features:',f_final.index(max(f_final))+1)
print('E_in:',EIN)
print('E_out:',EOUT)
print('Precision:',precision)
print('Recall:',recall)
print('F1:',f1)

#min_Eout=min(Out)
#max_recall=max(rec_final)
#k_recall=range(1,16)[rec_final.index(max_recall)]
#k_num=range(1,16)[Out.index(min_Eout)]
#print('E_out:',min_Eout,'features:',k_num,'Recall:',max_recall,'feature_rec',k_recall)
f.close()